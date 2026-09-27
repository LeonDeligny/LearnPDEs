"""Customed decorators."""

# ======= Imports =======
import time as t
from functools import (
    lru_cache,
    wraps,
)
from typing import (
    Callable,
    cast,
)

from pydantic import validate_call

# ======= Functions =======


def time[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to log the execution time of a function."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start_time = t.time()
        result = func(*args, **kwargs)
        end_time = t.time()
        execution_time = end_time - start_time
        print(f'Function {func.__name__} executed in {execution_time:.4f} seconds.')
        return result

    return cast(Callable[P, R], wrapper)


def validate[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Base decorator for validating function arguments.

    Objective:
        Redefines the @validate_call decorator to cache validation results.

    Uniqueness:
        Ensures validation is performed only once per unique set of arguments.
    """
    validated_func = validate_call(func)

    @wraps(func)
    @lru_cache(maxsize=None)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return validated_func(*args, **kwargs)

    return cast(Callable[P, R], wrapper)
