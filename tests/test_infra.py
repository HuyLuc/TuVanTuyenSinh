import json

from qdrant_client import models

from tuyensinh.config import Settings
from tuyensinh.retrieval.store import DENSE, SPARSE, ensure_collection, get_qdrant
from tuyensinh.telemetry.query_log import QueryLogger


def test_query_logger_writes_jsonl(tmp_path):
    logger = QueryLogger(Settings(_env_file=None, data_dir=tmp_path))
    logger.log("Điểm chuẩn Bách khoa?", "Chưa có dữ liệu", stage=0)
    [line] = logger.path.read_text(encoding="utf-8").splitlines()
    record = json.loads(line)
    assert record["question"] == "Điểm chuẩn Bách khoa?"
    assert record["stage"] == 0


def test_qdrant_local_hybrid_collection(tmp_path):
    client = get_qdrant(tmp_path / "qdrant")
    assert ensure_collection(client, "demo", dense_dim=4)
    assert not ensure_collection(client, "demo", dense_dim=4)

    client.upsert(
        "demo",
        points=[
            models.PointStruct(
                id=1,
                vector={
                    DENSE: [1.0, 0.0, 0.0, 0.0],
                    SPARSE: models.SparseVector(indices=[3, 7], values=[0.5, 0.2]),
                },
                payload={"truong": "BKA", "nam": 2026},
            )
        ],
    )
    hits = client.query_points(
        "demo",
        query=[1.0, 0.0, 0.0, 0.0],
        using=DENSE,
        query_filter=models.Filter(
            must=[models.FieldCondition(key="truong", match=models.MatchValue(value="BKA"))]
        ),
    ).points
    assert hits[0].payload["nam"] == 2026
