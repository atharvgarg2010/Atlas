"""
core/decorators.py
==================
Project Atlas — Reusable Function Decorators

Purpose:
    Provide battle-tested decorators for retry logic and execution timing.
    All external API calls (yfinance, news APIs) MUST use @retry to ensure
    the scheduler never crashes on transient network errors.

Dependencies:
    core.logging — get_logger

Failure Scenarios:
    - If all retry attempts are exhausted, the original exception is re-raised.
    - The decorator does NOT swallow exceptions; it only delays and logs them.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

from core.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def retry(
    max_attempts: int = 3,
    backoff_seconds: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Retry a function on failure with exponential backoff.

    Args:
        max_attempts:    Total number of attempts before giving up (default: 3).
        backoff_seconds: Base wait time between attempts in seconds (default: 2.0).
                         Wait time doubles on each retry: 2s → 4s → 8s ...
        exceptions:      Tuple of exception types to catch and retry on.
                         Defaults to (Exception,) — catches all exceptions.

    Returns:
        The decorated function.

    Raises:
        The last caught exception if all attempts are exhausted.

    Example:
        @retry(max_attempts=3, backoff_seconds=1.5, exceptions=(IOError,))
        def fetch_data(symbol: str) -> dict:
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc

                    if attempt < max_attempts:
                        wait = backoff_seconds * (2 ** (attempt - 1))
                        logger.warning(
                            f"[retry] {func.__qualname__} — attempt {attempt}/{max_attempts} failed. "
                            f"Retrying in {wait:.1f}s. Error: {exc}"
                        )
                        time.sleep(wait)
                    else:
                        logger.error(
                            f"[retry] {func.__qualname__} — all {max_attempts} attempts exhausted. "
                            f"Final error: {exc}"
                        )

            raise last_exc  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


def timed(func: F) -> F:
    """
    Log the wall-clock execution time of a function.

    Useful on scheduler jobs to detect performance regressions.

    Example:
        @timed
        def run_scoring_job() -> None:
            ...
        # Logs: "[timed] run_scoring_job completed in 4.23s"
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(
                f"[timed] {func.__qualname__} completed in {elapsed:.3f}s"
            )
            return result
        except Exception:
            elapsed = time.perf_counter() - start
            logger.warning(
                f"[timed] {func.__qualname__} FAILED after {elapsed:.3f}s"
            )
            raise

    return wrapper  # type: ignore[return-value]
