from pathlib import Path
from src.utils.cache_utils import PredictionCache


def test_prediction_cache_roundtrip(tmp_path: Path) -> None:
    cache = PredictionCache(namespace="test_ns", cache_dir=tmp_path)
    payload = {"features": [5600, 30, 30, 30, 80, 110, 1.1, 1.8], "version": 1}
    key = cache.key_for(payload)

    # Initially empty
    assert cache.get(key) is None

    cache.set(key, {"score": 0.85})
    got = cache.get(key)
    assert got is not None
    assert got.get("score") == 0.85
