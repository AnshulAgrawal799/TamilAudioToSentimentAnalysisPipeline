import functools
import logging
from time import perf_counter

logger = logging.getLogger(__name__)


def trace(level: int = logging.INFO):
    """Decorator to trace function entry/exit with duration in ms.

    Usage:
        @trace()
        def my_func(...):
            ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = perf_counter()
            try:
                logger.log(level, f"→ {func.__module__}.{func.__name__} args={args[:1]} kwargs_keys={list(kwargs.keys())}")
                return func(*args, **kwargs)
            finally:
                dur_ms = (perf_counter() - start) * 1000.0
                logger.log(level, f"← {func.__module__}.{func.__name__} took {dur_ms:.1f} ms")

        return wrapper

    return decorator


