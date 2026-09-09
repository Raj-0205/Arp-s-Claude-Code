"""Rate limiters for API requests with provider isolation."""

import asyncio
import random
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, ClassVar, TypeVar

import openai
from loguru import logger

T = TypeVar("T")


class BaseRateLimiter:
    """Base rate limiter implementing rolling-window throttling,

    reactive 429 backoff, and max concurrency control.
    """

    def __init__(
        self,
        rate_limit: int = 40,
        rate_window: float = 60.0,
        max_concurrency: int = 5,
        name: str = "default",
    ):
        if rate_limit <= 0:
            raise ValueError("rate_limit must be > 0")
        if rate_window <= 0:
            raise ValueError("rate_window must be > 0")
        if max_concurrency <= 0:
            raise ValueError("max_concurrency must be > 0")

        self._name = name
        self._rate_limit = rate_limit
        self._rate_window = float(rate_window)
        # Monotonic timestamps of the last granted slots.
        self._request_times: deque[float] = deque()
        self._blocked_until: float = 0.0
        self._lock = asyncio.Lock()
        self._concurrency_sem = asyncio.Semaphore(max_concurrency)

    @property
    def name(self) -> str:
        """Provider or limiter identifier."""
        return self._name

    @property
    def rate_limit(self) -> int:
        return self._rate_limit

    @property
    def rate_window(self) -> float:
        return self._rate_window

    async def wait_if_blocked(self) -> bool:
        """Wait if currently rate limited or throttle to meet quota.

        Returns:
            True if was reactively blocked and waited, False otherwise.
        """
        # 1. Reactive check: Wait if someone hit a 429
        waited_reactively = False
        now = time.monotonic()
        if now < self._blocked_until:
            wait_time = self._blocked_until - now
            logger.warning(
                f"Rate limit active for '{self._name}' (reactive), waiting {wait_time:.1f}s..."
            )
            await asyncio.sleep(wait_time)
            waited_reactively = True

        # 2. Proactive check: strict rolling window (no bursts beyond N in last W seconds)
        await self._acquire_proactive_slot()
        return waited_reactively

    async def _acquire_proactive_slot(self) -> None:
        """Acquire a proactive slot enforcing a strict rolling window.

        Guarantees: at most `self._rate_limit` acquisitions in any interval of length
        `self._rate_window` (seconds).
        """
        while True:
            wait_time = 0.0
            async with self._lock:
                now = time.monotonic()
                cutoff = now - self._rate_window

                while self._request_times and self._request_times[0] <= cutoff:
                    self._request_times.popleft()

                if len(self._request_times) < self._rate_limit:
                    self._request_times.append(now)
                    return

                oldest = self._request_times[0]
                wait_time = max(0.0, (oldest + self._rate_window) - now)

            # Sleep outside the lock so other tasks can continue to queue.
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(0)

    def set_blocked(self, seconds: float = 60) -> None:
        """Set block for specified seconds (reactive).

        Args:
            seconds: How long to block (default 60s)
        """
        self._blocked_until = time.monotonic() + seconds
        logger.warning(
            f"Rate limit for '{self._name}' set for {seconds:.1f}s (reactive)"
        )

    def is_blocked(self) -> bool:
        """Check if currently reactively blocked."""
        return time.monotonic() < self._blocked_until

    def remaining_wait(self) -> float:
        """Get remaining reactive wait time in seconds."""
        return max(0.0, self._blocked_until - time.monotonic())

    @asynccontextmanager
    async def concurrency_slot(self) -> AsyncIterator[None]:
        """Async context manager that holds one concurrency slot for a stream.

        Blocks until a slot is available (controlled by max_concurrency).
        """
        await self._concurrency_sem.acquire()
        try:
            yield
        finally:
            self._concurrency_sem.release()

    async def execute_with_retry(
        self,
        fn: Callable[..., Any],
        *args: Any,
        max_retries: int = 3,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        jitter: float = 1.0,
        **kwargs: Any,
    ) -> Any:
        """Execute an async callable with rate limiting and retry on 429.

        Waits for the proactive limiter before each attempt. On 429, applies
        exponential backoff with jitter before retrying.

        Args:
            fn: Async callable to execute.
            max_retries: Maximum number of retry attempts after the first failure.
            base_delay: Base delay in seconds for exponential backoff.
            max_delay: Maximum delay cap in seconds.
            jitter: Maximum random jitter in seconds added to each delay.

        Returns:
            The result of the callable.

        Raises:
            The last exception if all retries are exhausted.
        """
        last_exc: Exception | None = None

        for attempt in range(1 + max_retries):
            await self.wait_if_blocked()

            try:
                return await fn(*args, **kwargs)
            except openai.RateLimitError as e:
                last_exc = e
                if attempt >= max_retries:
                    logger.warning(
                        f"[{self._name}] Rate limit retry exhausted after {max_retries} retries"
                    )
                    break

                delay = min(base_delay * (2**attempt), max_delay)
                delay += random.uniform(0, jitter)
                logger.warning(
                    f"[{self._name}] Rate limited (429), attempt {attempt + 1}/{max_retries + 1}. "
                    f"Retrying in {delay:.1f}s..."
                )
                self.set_blocked(delay)
                await asyncio.sleep(delay)

        assert last_exc is not None
        raise last_exc


class ProviderRateLimiter(BaseRateLimiter):
    """Provider-specific rate limiter maintaining isolated buckets per provider.

    NVIDIA NIM, OpenRouter, and LM Studio traffic each use their own instance,
    ensuring a 429 or throttling on one provider does not affect another.
    Note: Local limiter instances prevent unnecessary shared bottlenecks in the proxy;
    they do not bypass upstream provider limits or quotas.
    """

    _instances: ClassVar[dict[str, ProviderRateLimiter]] = {}

    def __init__(
        self,
        provider_name: str = "default",
        rate_limit: int = 40,
        rate_window: float = 60.0,
        max_concurrency: int = 5,
    ):
        norm_name = self.normalize_provider_name(provider_name)
        super().__init__(
            rate_limit=rate_limit,
            rate_window=rate_window,
            max_concurrency=max_concurrency,
            name=norm_name,
        )
        logger.info(
            f"ProviderRateLimiter ({norm_name}) initialized "
            f"({rate_limit} req / {rate_window}s, max_concurrency={max_concurrency})"
        )

    @staticmethod
    def normalize_provider_name(provider_name: str) -> str:
        """Normalize provider names to canonical keys."""
        cleaned = provider_name.strip().lower()
        if cleaned in ("nvidia", "nim", "nvidia_nim"):
            return "nvidia_nim"
        if cleaned in ("openrouter", "open_router"):
            return "open_router"
        if cleaned in ("lmstudio", "lm_studio", "lm-studio"):
            return "lmstudio"
        return cleaned or "default"

    @classmethod
    def get_instance(
        cls,
        provider_name: str = "default",
        rate_limit: int | None = None,
        rate_window: float | None = None,
        max_concurrency: int = 5,
    ) -> ProviderRateLimiter:
        """Get or create the rate limiter for a specific provider.

        Args:
            provider_name: Name of provider (e.g. "nvidia_nim", "open_router", "lmstudio")
            rate_limit: Requests per window (used on first creation)
            rate_window: Window in seconds (used on first creation)
            max_concurrency: Max simultaneous open streams (used on first creation)
        """
        key = cls.normalize_provider_name(provider_name)
        if key not in cls._instances:
            cls._instances[key] = cls(
                provider_name=key,
                rate_limit=rate_limit or 40,
                rate_window=rate_window or 60.0,
                max_concurrency=max_concurrency,
            )
        return cls._instances[key]

    @classmethod
    def reset_instance(cls, provider_name: str | None = None) -> None:
        """Reset rate limiter instance(s) (for testing).

        If provider_name is None, all instances are cleared.
        """
        if provider_name is None:
            cls._instances.clear()
        else:
            cls._instances.pop(cls.normalize_provider_name(provider_name), None)

    @classmethod
    def reset_all(cls) -> None:
        """Reset all provider limiter instances."""
        cls._instances.clear()


class GlobalRateLimiter(BaseRateLimiter):
    """Global singleton rate limiter that blocks all requests

    when a rate limit error is encountered (reactive) and
    throttles requests (proactive) using a strict rolling window.

    Maintained for full backward compatibility with existing tests and call sites.
    """

    _instance: ClassVar[GlobalRateLimiter | None] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> GlobalRateLimiter:
        if cls._instance is not None:
            return cls._instance
        instance = super().__new__(cls)
        return instance

    def __init__(
        self,
        rate_limit: int = 40,
        rate_window: float = 60.0,
        max_concurrency: int = 5,
    ):
        # Prevent re-initialization on singleton reuse
        if hasattr(self, "_initialized"):
            return

        super().__init__(
            rate_limit=rate_limit,
            rate_window=rate_window,
            max_concurrency=max_concurrency,
            name="global",
        )
        self._initialized = True

        logger.info(
            f"GlobalRateLimiter (Provider) initialized ({rate_limit} req / {rate_window}s, max_concurrency={max_concurrency})"
        )

    @classmethod
    def get_instance(
        cls,
        rate_limit: int | None = None,
        rate_window: float | None = None,
        max_concurrency: int = 5,
    ) -> GlobalRateLimiter:
        """Get or create the singleton instance.

        Args:
            rate_limit: Requests per window (only used on first creation)
            rate_window: Window in seconds (only used on first creation)
            max_concurrency: Max simultaneous open streams (only used on first creation)
        """
        if cls._instance is None:
            cls._instance = cls(
                rate_limit=rate_limit or 40,
                rate_window=rate_window or 60.0,
                max_concurrency=max_concurrency,
            )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None


__all__ = [
    "BaseRateLimiter",
    "GlobalRateLimiter",
    "ProviderRateLimiter",
]
