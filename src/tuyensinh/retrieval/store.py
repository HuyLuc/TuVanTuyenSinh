"""Qdrant chế độ local: dữ liệu lưu thẳng ra thư mục, không cần server.

Chế độ local khóa thư mục, mỗi lúc chỉ một tiến trình mở được. Vì vậy script
nhập dữ liệu phải chạy khi API đang tắt, và trong một tiến trình luôn dùng
chung một client qua get_qdrant().
"""

import atexit
from functools import lru_cache
from pathlib import Path

from qdrant_client import QdrantClient, models

from tuyensinh.config import get_settings

DENSE = "dense"
SPARSE = "sparse"
BGE_M3_DIM = 1024


@lru_cache
def get_qdrant(path: Path | None = None) -> QdrantClient:
    path = path or get_settings().qdrant_path
    path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=str(path))
    # Đóng trước khi Python tắt để nhả khóa thư mục và tránh lỗi trong __del__.
    atexit.register(client.close)
    return client


def ensure_collection(client: QdrantClient, name: str, dense_dim: int = BGE_M3_DIM) -> bool:
    """Tạo collection có cả dense và sparse vector nếu chưa có. Trả về True nếu vừa tạo."""
    if client.collection_exists(name):
        return False
    client.create_collection(
        collection_name=name,
        vectors_config={
            DENSE: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={SPARSE: models.SparseVectorParams()},
    )
    return True
