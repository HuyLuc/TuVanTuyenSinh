"""Lưu log câu hỏi thật của người dùng, dùng làm nguồn bổ sung cho bộ đánh giá.

Hai cách lưu, chọn qua QUERY_LOG_BACKEND:
- "local": ghi file JSONL trong data/logs (khi chạy Docker thì nằm trong volume).
- "hf_dataset": vẫn ghi JSONL, và CommitScheduler định kỳ đẩy thư mục lên một
  Hugging Face Dataset riêng tư, vì ổ đĩa của Space bị xóa khi khởi động lại.
"""

import json
import threading
import uuid
from datetime import UTC, datetime
from typing import Any

from tuyensinh.config import Settings, get_settings


class QueryLogger:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        log_dir = self.settings.log_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        # Mỗi tiến trình một file riêng để không ghi đè nhau khi đẩy lên Dataset.
        self.path = log_dir / f"queries-{datetime.now(UTC):%Y%m%d}-{uuid.uuid4().hex[:8]}.jsonl"
        self._scheduler = None
        self._lock: Any = threading.Lock()

        if self.settings.query_log_backend == "hf_dataset":
            from huggingface_hub import CommitScheduler

            if not self.settings.hf_log_repo_id:
                raise RuntimeError("QUERY_LOG_BACKEND=hf_dataset cần HF_LOG_REPO_ID.")
            self._scheduler = CommitScheduler(
                repo_id=self.settings.hf_log_repo_id,
                repo_type="dataset",
                folder_path=log_dir,
                path_in_repo="logs",
                every=10,
                private=True,
                token=self.settings.hf_token or None,
            )
            self._lock = self._scheduler.lock

    def log(self, question: str, answer: str, **meta: Any) -> None:
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "question": question,
            "answer": answer,
            **meta,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


_default: QueryLogger | None = None


def get_query_logger() -> QueryLogger:
    global _default
    if _default is None:
        _default = QueryLogger()
    return _default
