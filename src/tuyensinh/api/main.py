"""FastAPI kèm giao diện Gradio, chạy trong một tiến trình.

Giai đoạn 0 mới chỉ là khung: câu hỏi được gửi thẳng cho Gemini, chưa có
bộ định tuyến, tìm kiếm văn bản hay tra cứu số liệu.
"""

import logging
from collections.abc import Iterator
from typing import Any

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from google.genai import types
from pydantic import BaseModel, Field

from tuyensinh.llm import get_llm
from tuyensinh.telemetry.query_log import get_query_logger

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Bạn là trợ lý tư vấn tuyển sinh đại học ở Việt Nam. Hệ thống hiện CHƯA có dữ liệu "
    "của các trường, nên không được đưa ra con số cụ thể (điểm chuẩn, chỉ tiêu, học phí). "
    "Nếu người dùng hỏi số liệu, hãy nói rõ là chưa có dữ liệu."
)
DISCLAIMER = (
    "Thông tin chỉ mang tính tham khảo, cần đối chiếu với thông báo chính thức của trường. "
    "Câu hỏi được lưu lại để cải thiện hệ thống, vui lòng không nhập thông tin cá nhân."
)
MAX_HISTORY_MESSAGES = 6


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list)


def _message_text(content: Any) -> str:
    # Gradio 6 có thể trả nội dung dạng chuỗi hoặc danh sách phần tử {"type": "text", ...}.
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def build_contents(question: str, history: list[Turn]) -> list[types.Content]:
    contents = [
        types.Content(
            role="model" if t.role == "assistant" else "user",
            parts=[types.Part.from_text(text=t.content)],
        )
        for t in history[-MAX_HISTORY_MESSAGES:]
        if t.content
    ]
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
    return contents


def answer_stream(question: str, history: list[Turn]) -> Iterator[str]:
    parts: list[str] = []
    try:
        for piece in get_llm().stream(build_contents(question, history), system=SYSTEM_PROMPT):
            parts.append(piece)
            yield piece
    except Exception:
        logger.exception("Lỗi khi gọi LLM")
        msg = "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại sau."
        parts.append(msg)
        yield msg
    finally:
        get_query_logger().log(question, "".join(parts), stage=0)


app = FastAPI(title="Tư vấn tuyển sinh")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        answer_stream(req.question, req.history), media_type="text/plain; charset=utf-8"
    )


def _gradio_chat(message: str, history: list[dict[str, Any]]) -> Iterator[str]:
    turns = [
        Turn(role=m["role"], content=_message_text(m.get("content")))
        for m in history
        if m.get("role") in ("user", "assistant")
    ]
    text = ""
    for piece in answer_stream(message, turns):
        text += piece
        yield text


demo = gr.ChatInterface(
    fn=_gradio_chat,
    title="Tư vấn tuyển sinh đại học",
    description=DISCLAIMER,
    examples=[
        "Phương thức xét tuyển đại học phổ biến hiện nay là gì?",
        "Điểm chuẩn ngành CNTT Bách khoa năm 2026 là bao nhiêu?",
    ],
)

app = gr.mount_gradio_app(app, demo, path="/")
