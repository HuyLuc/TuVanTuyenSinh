"""Hàm gọi Gemini dùng chung cho toàn dự án.

- Chọn mô hình theo vai trò ("router" dùng bản nhẹ, "answer" dùng bản trả lời),
  tên mô hình nằm trong cấu hình nên đổi mô hình chỉ cần sửa .env.
- Tự thử lại khi bị giới hạn lượt gọi (429) hoặc lỗi phía server (5xx).
- Cache kết quả: cùng mô hình + nội dung + cấu hình thì trả kết quả cũ, không gọi API.
- Hỗ trợ gọi hàm (function calling, không tự thực thi), JSON schema và streaming.
"""

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import chain
from typing import Any, Literal

from google import genai
from google.genai import errors, types
from pydantic import BaseModel
from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

from tuyensinh.config import Settings, get_settings
from tuyensinh.llm.cache import ResponseCache, make_cache_key

logger = logging.getLogger(__name__)

Role = Literal["answer", "router"]
Contents = str | list[str | types.Content]

RETRYABLE_CODES = {429, 500, 502, 503, 504}


def is_retryable(exc: BaseException) -> bool:
    return isinstance(exc, errors.APIError) and exc.code in RETRYABLE_CODES


@dataclass
class LLMResult:
    response: types.GenerateContentResponse
    cached: bool

    @property
    def text(self) -> str:
        return self.response.text or ""

    @property
    def function_calls(self) -> list[types.FunctionCall]:
        return self.response.function_calls or []

    def json(self) -> Any:
        return json.loads(self.text)


def _to_jsonable(contents: Contents) -> Any:
    if isinstance(contents, str):
        return contents
    return [
        c.model_dump(mode="json", exclude_none=True) if isinstance(c, BaseModel) else c
        for c in contents
    ]


class LLMClient:
    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
        cache: ResponseCache | None = None,
    ):
        self.settings = settings or get_settings()
        self._client = client
        if cache is None and self.settings.llm_cache_enabled:
            cache = ResponseCache(self.settings.cache_path)
        self._cache = cache

    @property
    def client(self) -> Any:
        if self._client is None:
            if not self.settings.gemini_api_key:
                raise RuntimeError(
                    "Thiếu GEMINI_API_KEY. Sao chép .env.example thành .env và điền key."
                )
            self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client

    def model_for(self, role: Role) -> str:
        if role == "router":
            return self.settings.gemini_model_router
        return self.settings.gemini_model_answer

    def build_config(
        self,
        *,
        system: str | None = None,
        tools: list[types.FunctionDeclaration] | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        force_tool: bool = False,
    ) -> types.GenerateContentConfig:
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            # Hệ thống tự thực thi hàm và kiểm tra tham số, không để SDK tự gọi.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        if tools:
            config.tools = [types.Tool(function_declarations=tools)]
            if force_tool:
                # Bắt buộc mô hình gọi một trong các hàm thay vì trả lời bằng chữ.
                config.tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="ANY")
                )
        if json_schema is not None:
            config.response_mime_type = "application/json"
            config.response_json_schema = json_schema
        return config

    def _retrying(self) -> Retrying:
        return Retrying(
            retry=retry_if_exception(is_retryable),
            wait=wait_random_exponential(multiplier=2, max=self.settings.llm_retry_max_wait),
            stop=stop_after_attempt(self.settings.llm_max_retries),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )

    def _key(self, model: str, contents: Contents, config: types.GenerateContentConfig, **extra):
        cfg = config.model_dump(mode="json", exclude_none=True) | extra
        return make_cache_key(model, _to_jsonable(contents), cfg)

    def generate(
        self,
        contents: Contents,
        *,
        role: Role = "answer",
        system: str | None = None,
        tools: list[types.FunctionDeclaration] | None = None,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.0,
        force_tool: bool = False,
        use_cache: bool = True,
    ) -> LLMResult:
        model = self.model_for(role)
        config = self.build_config(
            system=system,
            tools=tools,
            json_schema=json_schema,
            temperature=temperature,
            force_tool=force_tool,
        )
        cache = self._cache if use_cache else None
        key = self._key(model, contents, config) if cache else None

        if cache and (hit := cache.get(key)) is not None:
            response = types.GenerateContentResponse.model_validate_json(hit)
            return LLMResult(response=response, cached=True)

        response = self._retrying()(
            self.client.models.generate_content, model=model, contents=contents, config=config
        )
        if cache:
            cache.set(key, model, response.model_dump_json(exclude_none=True))
        return LLMResult(response=response, cached=False)

    def stream(
        self,
        contents: Contents,
        *,
        role: Role = "answer",
        system: str | None = None,
        temperature: float = 0.0,
        use_cache: bool = True,
    ) -> Iterator[str]:
        """Trả về từng đoạn chữ. Cache hit thì trả toàn bộ câu trả lời một lần."""
        model = self.model_for(role)
        config = self.build_config(system=system, temperature=temperature)
        cache = self._cache if use_cache else None
        key = self._key(model, contents, config, stream=True) if cache else None

        if cache and (hit := cache.get(key)) is not None:
            yield json.loads(hit)["text"]
            return

        def open_stream():
            # Chỉ thử lại khi chưa nhận được đoạn nào, tránh lặp chữ cho người dùng.
            it = iter(
                self.client.models.generate_content_stream(
                    model=model, contents=contents, config=config
                )
            )
            return next(it, None), it

        first, it = self._retrying()(open_stream)
        parts: list[str] = []
        head = [first] if first is not None else []
        for chunk in chain(head, it):
            if chunk.text:
                parts.append(chunk.text)
                yield chunk.text

        if cache and parts:
            cache.set(key, model, json.dumps({"text": "".join(parts)}, ensure_ascii=False))


_default: LLMClient | None = None


def get_llm() -> LLMClient:
    global _default
    if _default is None:
        _default = LLMClient()
    return _default
