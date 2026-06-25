"""
Disk-based cache with TTL support.
Prevents redundant API calls across pipeline stages.
"""

from pathlib import Path
import json
import pickle
import hashlib
import logging
from datetime import datetime
from typing import Any, Optional

import config

logger = logging.getLogger(__name__)


def _cache_key(namespace: str, key: str) -> str:
    h = hashlib.md5(f"{namespace}:{key}".encode()).hexdigest()
    return h


def _cache_path(namespace: str, key: str, ext: str = "pkl") -> Path:
    ck = _cache_key(namespace, key)
    return config.CACHE_DIR / f"{namespace}_{ck}.{ext}"


def get(namespace: str, key: str, ttl: int) -> Optional[Any]:
    """
    Retrieve a value from cache.
    Returns None if missing or expired.
    
    Args:
        namespace: e.g. "fundamentals", "news"
        key: unique key for the cached item 
        ttl: max age in seconds
    """
    path = _cache_path(namespace, key)
    if not path.exists():
        return None
    
    age = (datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)).total_seconds()
    if age > ttl:
        logger.debug(f"Cache MISS (expired {age:.0f}s old > {ttl}s): {namespace}:{key}")
        return None
    
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        logger.debug(f"Cache HIT (age {age:.0f}s): {namespace}:{key}")
        return data
    except Exception as e:
        logger.warning(f"Cache read error for {namespace}:{key}: {e}")
        return None


def set(namespace: str, key: str, value: Any) -> None:
    """Store a value in cache."""
    path = _cache_path(namespace, key)
    try:
        with open(path, "wb") as f:
            pickle.dump(value, f)
        logger.debug(f"Cache SET: {namespace}:{key} → {path.name}")
    except Exception as e:
        logger.warning(f"Cache write error for {namespace}:{key}: {e}")


def clear(namespace: Optional[str] = None) -> int:
    """Clear cache entries. If namespace given, only clear that namespace."""
    count = 0
    for f in config.CACHE_DIR.glob("*.pkl"):
        if namespace is None or f.name.startswith(namespace + "_"):
            f.unlink()
            count += 1
    logger.info(f"Cleared {count} cache entries" + (f" for namespace '{namespace}'" if namespace else ""))
    return count


def stats() -> dict:
    """Return cache statistics."""
    files = list(config.CACHE_DIR.glob("*.pkl"))
    total_size = sum(f.stat().st_size for f in files)
    return {
        "entries": len(files),
        "total_size_mb": total_size / 1_000_000,
        "cache_dir": str(config.CACHE_DIR),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    set("test", "hello", {"value": 42})
    result = get("test", "hello", ttl=60)
    print("Cache test:", result)
    print("Stats:", stats())
