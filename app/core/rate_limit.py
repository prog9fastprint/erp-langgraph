"""
In-process sliding-window rate limiter for FastPrint ERP AI Orchestrator endpoints.
Uses stdlib primitives only with bounded LRU bucket eviction to prevent RAM exhaustion.
"""
from collections import OrderedDict, deque
import hashlib
import math
import threading
import time
from typing import Callable, Deque, Optional, Tuple


class SlidingWindowRateLimiter:
    """
    Thread-safe in-process sliding-window rate limiter with bounded bucket storage.
    """

    def __init__(
        self,
        max_buckets: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_buckets = max(1, int(max_buckets))
        self._clock = clock
        self._lock = threading.Lock()
        self._buckets: "OrderedDict[str, Deque[float]]" = OrderedDict()

    def check(
        self,
        key: str,
        limit: int,
        window_seconds: int,
    ) -> Tuple[bool, int]:
        """
        Record a request attempt for ``key`` under a sliding window.

        Returns:
            Tuple of ``(allowed, retry_after_seconds)``.
            When ``allowed`` is True, ``retry_after_seconds`` is 0.
        """
        safe_limit = max(1, int(limit))
        safe_window = max(1, int(window_seconds))
        bucket_key = (key or "ip:unknown").strip() or "ip:unknown"
        now = float(self._clock())
        cutoff = now - float(safe_window)

        with self._lock:
            timestamps = self._buckets.get(bucket_key)
            if timestamps is None:
                timestamps = deque()
                self._buckets[bucket_key] = timestamps
            else:
                self._buckets.move_to_end(bucket_key, last=True)

            while timestamps and timestamps[0] <= cutoff:
                timestamps.popleft()

            if len(timestamps) >= safe_limit:
                oldest = timestamps[0]
                remaining = float(safe_window) - (now - oldest)
                retry_after = max(1, int(math.ceil(remaining)))
                return False, retry_after

            timestamps.append(now)

            while len(self._buckets) > self.max_buckets:
                self._buckets.popitem(last=False)

            return True, 0

    def bucket_count(self) -> int:
        """Return current number of tracked buckets (useful for tests/diagnostics)."""
        with self._lock:
            return len(self._buckets)

    def reset(self) -> None:
        """Clear all tracked rate-limit buckets."""
        with self._lock:
            self._buckets.clear()


def build_rate_limit_key(
    *,
    user_id: Optional[str] = None,
    effective_token: Optional[str] = None,
    client_host: Optional[str] = None,
    forwarded_for: Optional[str] = None,
    service_key_verified: bool = False,
    trust_proxy_headers: bool = False,
) -> str:
    """
    Derive an anti-spoofing rate-limit bucket key with priority:
      1. Verified user_id (trusted only when X-Internal-Service-Key is verified)
      2. SHA-256 fingerprint of effective end-user token
      3. Client IP (X-Forwarded-For leftmost IP only when proxy is trusted)
    """
    if service_key_verified:
        clean_uid = (user_id or "").strip()
        if clean_uid:
            return f"uid:{clean_uid}"

    clean_token = (effective_token or "").strip()
    if clean_token:
        digest = hashlib.sha256(clean_token.encode("utf-8")).hexdigest()[:16]
        return f"tok:{digest}"

    if (service_key_verified or trust_proxy_headers) and forwarded_for:
        first_hop = forwarded_for.split(",")[0].strip()
        if first_hop:
            return f"ip:{first_hop}"

    clean_host = (client_host or "").strip() or "unknown"
    return f"ip:{clean_host}"


chat_rate_limiter = SlidingWindowRateLimiter()
