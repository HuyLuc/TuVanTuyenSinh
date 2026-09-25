"""Kiểm tra nhanh Giai đoạn 0 với API thật.

    uv run python scripts/smoke_test.py               # Gemini + cache + JSON + gọi hàm + Qdrant
    uv run python scripts/smoke_test.py --list-models # liệt kê mô hình để chọn tên cho .env

Lần chạy thứ hai sẽ lấy toàn bộ kết quả từ cache, không tốn lượt gọi.
"""

import argparse
import sys
import time

from google.genai import types

from tuyensinh.config import get_settings
from tuyensinh.llm import get_llm
from tuyensinh.retrieval.store import ensure_collection, get_qdrant


def list_models() -> None:
    for m in get_llm().client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions:
            print(m.name)


def check_llm() -> None:
    llm = get_llm()
    s = get_settings()
    print(f"Mô hình trả lời: {s.gemini_model_answer} | định tuyến: {s.gemini_model_router}")

    q = "Trong một câu, xét tuyển đại học bằng học bạ là gì?"
    for i in (1, 2):
        t = time.perf_counter()
        r = llm.generate(q)
        dt = time.perf_counter() - t
        print(f"[generate #{i}] cached={r.cached} {dt:.2f}s -> {r.text[:80]!r}")

    schema = {
        "type": "object",
        "properties": {
            "loai": {"type": "string", "enum": ["so_lieu", "van_ban", "ca_hai", "ngoai_pham_vi"]},
            "truong": {"type": "array", "items": {"type": "string"}},
            "nam": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["loai", "truong", "nam"],
    }
    r = llm.generate(
        "Điểm chuẩn ngành CNTT Bách khoa năm 2025 là bao nhiêu?",
        role="router",
        system="Phân loại câu hỏi tuyển sinh. Trích nguyên văn tên trường người dùng nhắc đến.",
        json_schema=schema,
    )
    print(f"[router JSON] cached={r.cached} -> {r.json()}")

    tool = types.FunctionDeclaration(
        name="tra_diem_chuan",
        description="Tra điểm chuẩn theo trường, ngành, năm.",
        parameters_json_schema={
            "type": "object",
            "properties": {
                "truong": {"type": "string"},
                "nganh": {"type": "string"},
                "nam": {"type": "integer"},
            },
        },
    )
    r = llm.generate("Điểm chuẩn CNTT Bách khoa 2025?", tools=[tool], force_tool=True)
    calls = [(c.name, dict(c.args or {})) for c in r.function_calls]
    print(f"[function call] cached={r.cached} -> {calls}")

    print("[stream] ", end="")
    for piece in llm.stream("Kể tên 3 tổ hợp xét tuyển khối kỹ thuật, một dòng."):
        print(piece, end="", flush=True)
    print()


def check_qdrant() -> None:
    client = get_qdrant()
    created = ensure_collection(client, "smoke_test", dense_dim=4)
    print(f"[qdrant local] {get_settings().qdrant_path} collection smoke_test created={created}")
    client.delete_collection("smoke_test")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()
    if args.list_models:
        list_models()
        return 0
    check_qdrant()
    check_llm()
    return 0


if __name__ == "__main__":
    sys.exit(main())
