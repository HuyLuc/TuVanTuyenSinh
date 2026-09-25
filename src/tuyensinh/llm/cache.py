"""Cache kết quả gọi LLM trong SQLite.

Khóa cache là mã băm SHA-256 của mô hình, nội dung gửi đi và toàn bộ cấu hình
(prompt hệ thống, tham số sinh, danh sách hàm, JSON schema). Sửa bất kỳ thứ gì
trong số đó thì khóa đổi, nên không bao giờ lấy nhầm kết quả cũ.
"""

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


def make_cache_key(model: str, contents: Any, config: Any) -> str:
    payload = {"model": model, "contents": contents, "config": config}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                " key TEXT PRIMARY KEY,"
                " model TEXT NOT NULL,"
                " value TEXT NOT NULL,"
                " created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
            )
            self._conn.commit()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, key: str, model: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, model, value) VALUES (?, ?, ?)",
                (key, model, value),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
