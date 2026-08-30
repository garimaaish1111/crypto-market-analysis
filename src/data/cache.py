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


def is_live(value: Any) -> bool:
    """
    Predicate for ``cached``: true when a loader returned real fetched data.

    Loaders return ``(dataframe, source)`` where ``source`` is ``"live"`` on
    success and ``"sample"`` when the fetch failed and generated data stood in.
    Only the first is worth keeping.

    The length check is ``>= 2`` rather than ``== 2`` deliberately. The crypto
    loader returns a third element carrying *why* a coin fell back, and an exact
    length check silently classified every one of its results as not-worth-
    caching. It happens to pass its own predicate today, so nothing was broken —
    but the next loader to grow a field would have had its cache quietly stop
    working, with no error to notice.
    """
    return isinstance(value, tuple) and len(value) >= 2 and value[1] == "live"


def cached(
    key: str,
    producer: Callable[[], Any],
    ttl_hours: float = config.CACHE_TTL_HOURS,
    should_cache: Callable[[Any], bool] | None = None,
) -> Any:
    """
    Return the cached value for ``key``, or compute, store, and return it.

    ``should_cache`` decides whether a freshly computed value is worth writing
    to disk. Without it, a loader that failed and fell back to generated data
    would store that fallback under the full six-hour TTL — so a momentary drop
    in connectivity kept the dashboard on sample data long after the network
    came back, and nothing short of the refresh button could dislodge it. Pass
    ``is_live`` to persist successes only; failures are then retried on the very
    next run.

    **Stale beats synthetic.** If the producer fails and an *expired* response
    for the same key is still on disk, that expired response is served instead
    of the generated fallback. This project ships with a populated cache and is
    read at an unknown later date, possibly on a machine with no network. Given
    the choice between real data carrying an older date and invented data
    carrying today's, real is the more honest thing to put in front of a reader
    — and the Datasets tab states the date range of every feed, so nothing is
    passed off as more current than it is.
    """
    hit = load(key, ttl_hours)
    if hit is not None:
        return hit

    value = producer()
    if should_cache is None or should_cache(value):
        save(key, value)
        return value

    # The producer fell back. Prefer a real-but-expired response if one exists.
    stale = load(key, ttl_hours=float("inf"))
    if stale is not None:
        return stale
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
