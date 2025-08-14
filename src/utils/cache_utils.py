"""On-disk caching helpers for expensive predictions.

Design goals:
- Deterministic, file-based cache under user cache dir
- Safe for concurrent read; best-effort write
- JSON-serializable payloads with small metadata header
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

_CACHE_LOCK = threading.Lock()


def _default_cache_dir() -> Path:
    # Respect OS cache locations
    base = os.getenv("DDR5_CACHE_DIR")
    if base:
        return Path(base)
    # Windows: %LOCALAPPDATA%\DDR5-AI\cache
    local = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
    if local:
        return Path(local) / "DDR5-AI" / "cache"
    # Fallback to home
    return Path.home() / ".cache" / "ddr5-ai"


def _stable_hash(obj: Any) -> str:
    try:
        data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except TypeError:
        # Best-effort fallback using repr
        data = repr(obj).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


class PredictionCache:
    """Simple JSON file cache keyed by a stable hash.

    Each entry is one file named <key>.json with structure:
    {
      "version": 1,
      "payload": {...}
    }
    """

    def __init__(
        self,
        namespace: str = "predictions",
        cache_dir: Optional[Path] = None,
    ):
        self.base_dir = (cache_dir or _default_cache_dir()) / namespace
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def key_for(self, payload: Dict[str, Any]) -> str:
        return _stable_hash(payload)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self.base_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "payload" in data:
                return data["payload"]
            return None
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def set(self, key: str, payload: Dict[str, Any]) -> None:
        path = self.base_dir / f"{key}.json"
        tmp = self.base_dir / f".{key}.json.tmp"
        doc = {"version": 1, "payload": payload}
        try:
            with _CACHE_LOCK:
                with tmp.open("w", encoding="utf-8") as f:
                    json.dump(doc, f, separators=(",", ":"))
                tmp.replace(path)
        except OSError:
            # Best-effort; ignore failures
            try:
                if tmp.exists():
                    try:
                        tmp.unlink()  # type: ignore[call-arg]
                    except FileNotFoundError:
                        pass
            except OSError:
                pass
