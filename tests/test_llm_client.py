import pytest
from google.genai import errors, types

from tuyensinh.config import Settings
from tuyensinh.llm.cache import ResponseCache, make_cache_key
from tuyensinh.llm.client import LLMClient


def _response(text: str) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))
        ]
    )


def _rate_limited() -> errors.ClientError:
    return errors.ClientError(
        429, {"error": {"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED"}}
    )


class FakeModels:
    def __init__(self, failures: int = 0):
        self.calls: list[dict] = []
        self.failures = failures

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.failures:
            self.failures -= 1
            raise _rate_limited()
        return _response(f"trả lời từ {model}")

    def generate_content_stream(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        if self.failures:
            self.failures -= 1
            raise _rate_limited()
        return iter([_response("Xin "), _response("chào")])


class FakeClient:
    def __init__(self, failures: int = 0):
        self.models = FakeModels(failures)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        gemini_api_key="test",
        data_dir=tmp_path,
        llm_retry_max_wait=0,
        llm_max_retries=3,
    )


def make_llm(settings, failures: int = 0) -> tuple[LLMClient, FakeModels]:
    fake = FakeClient(failures)
    return LLMClient(
        settings=settings, client=fake, cache=ResponseCache(settings.cache_path)
    ), fake.models


def test_second_identical_call_is_served_from_cache(settings):
    llm, models = make_llm(settings)
    first = llm.generate("Điểm chuẩn là gì?")
    second = llm.generate("Điểm chuẩn là gì?")
    assert not first.cached and second.cached
    assert second.text == first.text
    assert len(models.calls) == 1


def test_cache_survives_new_client_instance(settings):
    llm, _ = make_llm(settings)
    llm.generate("câu hỏi")
    llm2, models2 = make_llm(settings)
    assert llm2.generate("câu hỏi").cached
    assert models2.calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"system": "prompt khác"},
        {"temperature": 0.7},
        {"role": "router"},
        {"json_schema": {"type": "object", "properties": {"loai": {"type": "string"}}}},
        {
            "tools": [
                types.FunctionDeclaration(
                    name="tra_diem_chuan",
                    description="Tra điểm chuẩn",
                    parameters_json_schema={"type": "object", "properties": {}},
                )
            ]
        },
    ],
)
def test_changing_config_misses_cache(settings, kwargs):
    llm, models = make_llm(settings)
    llm.generate("câu hỏi")
    assert not llm.generate("câu hỏi", **kwargs).cached
    assert len(models.calls) == 2


def test_role_selects_model(settings):
    llm, models = make_llm(settings)
    llm.generate("a", role="router")
    llm.generate("b", role="answer")
    assert [c["model"] for c in models.calls] == [
        settings.gemini_model_router,
        settings.gemini_model_answer,
    ]


def test_tools_disable_automatic_function_calling(settings):
    llm, _ = make_llm(settings)
    decl = types.FunctionDeclaration(name="f", parameters_json_schema={"type": "object"})
    config = llm.build_config(tools=[decl])
    assert config.automatic_function_calling.disable is True
    assert config.tools[0].function_declarations[0].name == "f"
    assert config.tool_config is None
    forced = llm.build_config(tools=[decl], force_tool=True)
    assert forced.tool_config.function_calling_config.mode == "ANY"


def test_retries_on_rate_limit(settings):
    llm, models = make_llm(settings, failures=2)
    assert llm.generate("câu hỏi").text.startswith("trả lời")
    assert len(models.calls) == 3


def test_gives_up_after_max_retries(settings):
    llm, models = make_llm(settings, failures=10)
    with pytest.raises(errors.ClientError):
        llm.generate("câu hỏi")
    assert len(models.calls) == settings.llm_max_retries


def test_does_not_retry_bad_request(settings):
    class BadModels(FakeModels):
        def generate_content(self, **kw):
            self.calls.append(kw)
            raise errors.ClientError(400, {"error": {"code": 400, "message": "bad"}})

    fake = FakeClient()
    fake.models = BadModels()
    llm = LLMClient(settings=settings, client=fake, cache=ResponseCache(settings.cache_path))
    with pytest.raises(errors.ClientError):
        llm.generate("x")
    assert len(fake.models.calls) == 1


def test_stream_yields_chunks_then_caches_full_text(settings):
    llm, models = make_llm(settings, failures=1)
    assert list(llm.stream("chào")) == ["Xin ", "chào"]
    assert list(llm.stream("chào")) == ["Xin chào"]
    assert len(models.calls) == 2  # 1 lần lỗi 429 + 1 lần thành công, lần thứ ba lấy từ cache


def test_use_cache_false_always_calls_api(settings):
    llm, models = make_llm(settings)
    llm.generate("x", use_cache=False)
    llm.generate("x", use_cache=False)
    assert len(models.calls) == 2


def test_cache_key_is_order_independent_for_config():
    assert make_cache_key("m", "q", {"a": 1, "b": 2}) == make_cache_key("m", "q", {"b": 2, "a": 1})
