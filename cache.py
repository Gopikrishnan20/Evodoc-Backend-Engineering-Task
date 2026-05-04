from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Optional

TTL_SECONDS: int = 3_600  # 1 hour


# ---------------------------------------------------------------------------
# Shared key builder
# ---------------------------------------------------------------------------

def _build_cache_key(
    proposed_medicines: list[str],
    current_medications: list[str],
    known_allergies: list[str],
    conditions: list[str],
) -> str:
    """
    Produce a deterministic SHA-256 key from all four patient-specific fields.

    ALL four lists are normalised (stripped, lowercased) and sorted so that
    order never matters:
        ['Aspirin', 'Warfarin'] == ['warfarin', 'aspirin']  → same key

    Why allergies AND conditions must be in the key:
      - Patient A: Warfarin + Aspirin, no allergies → result: safe_to_prescribe=True
      - Patient B: Warfarin + Aspirin, allergy=penicillin, kidney disease → very different result
      Caching on medicines alone would return Patient A's result to Patient B.
      That is a patient safety bug, not just a logic error.
    """
    norm = lambda lst: sorted(s.strip().lower() for s in lst if s.strip())
    payload = json.dumps(
        {
            "proposed": norm(proposed_medicines),
            "current": norm(current_medications),
            "allergies": norm(known_allergies),
            "conditions": norm(conditions),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# In-memory backend
# ---------------------------------------------------------------------------

class _InMemoryCache:
    """Thread-safe enough for single-process FastAPI (asyncio single thread)."""

    def __init__(self, ttl: int = TTL_SECONDS) -> None:
        self._store: dict[str, tuple[Any, float]] = {}
        self.ttl = ttl
        self.backend = "memory"

    # ---- public API --------------------------------------------------------

    def get(
        self,
        proposed_medicines: list[str],
        current_medications: list[str],
        known_allergies: list[str],
        conditions: list[str],
    ) -> Optional[Any]:
        key = _build_cache_key(proposed_medicines, current_medications, known_allergies, conditions)
        entry = self._store.get(key)
        if entry is None:
            return None
        value, ts = entry
        if time.monotonic() - ts > self.ttl:
            del self._store[key]
            return None
        return value

    def set(
        self,
        proposed_medicines: list[str],
        current_medications: list[str],
        known_allergies: list[str],
        conditions: list[str],
        value: Any,
    ) -> None:
        key = _build_cache_key(proposed_medicines, current_medications, known_allergies, conditions)
        self._store[key] = (value, time.monotonic())
        self._evict_expired()

    # ---- housekeeping ------------------------------------------------------

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, (_, ts) in self._store.items() if now - ts > self.ttl]
        for k in expired:
            del self._store[k]

    def stats(self) -> dict:
        return {"backend": self.backend, "entries": len(self._store)}


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------

class _RedisCache:
    def __init__(self, url: str, ttl: int = TTL_SECONDS) -> None:
        import redis  # lazy import so missing redis doesn't break anything

        self.client = redis.from_url(url, decode_responses=True)
        self.ttl = ttl
        self.backend = "redis"
        self._prefix = "evodoc:drug_safety:"

    def _key(
        self,
        proposed_medicines: list[str],
        current_medications: list[str],
        known_allergies: list[str],
        conditions: list[str],
    ) -> str:
        return self._prefix + _build_cache_key(
            proposed_medicines, current_medications, known_allergies, conditions
        )

    def get(
        self,
        proposed_medicines: list[str],
        current_medications: list[str],
        known_allergies: list[str],
        conditions: list[str],
    ) -> Optional[Any]:
        raw = self.client.get(self._key(proposed_medicines, current_medications, known_allergies, conditions))
        if raw is None:
            return None
        return json.loads(raw)

    def set(
        self,
        proposed_medicines: list[str],
        current_medications: list[str],
        known_allergies: list[str],
        conditions: list[str],
        value: Any,
    ) -> None:
        self.client.setex(
            self._key(proposed_medicines, current_medications, known_allergies, conditions),
            self.ttl,
            json.dumps(value, default=str),
        )

    def stats(self) -> dict:
        return {"backend": self.backend}


# ---------------------------------------------------------------------------
# Factory — try Redis, fall back to memory
# ---------------------------------------------------------------------------

def _build_cache() -> _InMemoryCache | _RedisCache:
    redis_url = os.getenv("REDIS_URL", "")
    if redis_url:
        try:
            instance = _RedisCache(redis_url)
            instance.client.ping()
            print(f"[cache] Connected to Redis at {redis_url}")
            return instance
        except Exception as exc:
            print(f"[cache] Redis unavailable ({exc}); falling back to in-memory cache.")
    return _InMemoryCache()


cache = _build_cache()
