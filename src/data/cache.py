"""
Tiny on-disk cache with a time-to-live.

API responses are pickled to ``data/cache`` and re-used within
``CACHE_TTL_HOURS``. This keeps the dashboard responsive and avoids hitting
free-tier rate limits on every interaction. The cache fails open: any error is
treated as a cache miss and never reaches the user.
"""
from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path
from typing import Any, Callable

import config


def _cache_path(key: str) -> Path:
    digest = hashlib.md5(key.encode()).hexdigest()
    return config.CACHE_DIR / f"{digest}.pkl"


def load(key: str, ttl_hours: float = config.CACHE_TTL_HOURS) -> Any | None:
    """Return a cached object for ``key`` if it exists and is fresh, else None."""
    path = _cache_path(key)
    try:
        if not path.exists():
            return None
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > ttl_hours:
            return None
        with path.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def save(key: str, obj: Any) -> None:
    """Persist ``obj`` under ``key``. Silently ignores write failures."""
    try:
        with _cache_path(key).open("wb") as fh:
            pickle.dump(obj, fh)
    except Exception:
        pass


def cached(key: str, producer: Callable[[], Any], ttl_hours: float = config.CACHE_TTL_HOURS) -> Any:
    """Return the cached value for ``key``, or compute, store, and return it."""
    hit = load(key, ttl_hours)
    if hit is not None:
        return hit
    value = producer()
    save(key, value)
    return value


def clear_all() -> int:
    """
    Delete every cached response and return how many files were removed.

    This is what the dashboard's refresh control calls. Clearing Streamlit's
    in-memory cache alone is not enough: a failed fetch stores its sample-data
    fallback on disk under the same six-hour TTL, so without this the app would
    keep serving that fallback for six hours even after the network recovered.
    """
    removed = 0
    try:
        for path in config.CACHE_DIR.glob("*.pkl"):
            try:
                path.unlink()
                removed += 1
            except Exception:
                continue
    except Exception:
        pass
    return removed


def entry_count() -> int:
    """How many cached responses are currently on disk."""
    try:
        return len(list(config.CACHE_DIR.glob("*.pkl")))
    except Exception:
        return 0
