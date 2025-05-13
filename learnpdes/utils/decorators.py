'''
Customed decorators.
'''

# ======= Imports =======

import time as t

from pydantic import validate_call
from functools import (
    wraps,
    lru_cache,
)

from typing import (
    Tuple,
    Callable,
)

from typing import Any

# ======= Functions =======


def time(func: Callable) -> Callable:
    '''
    Decorator to log the execution time of a function.
    '''

    @wraps(func)
    def wrapper(*args: Tuple[Any, ...], **kwargs: dict[str, Any]) -> Any:
        start_time = t.time()
        result = func(*args, **kwargs)
        end_time = t.time()
        execution_time = end_time - start_time
        print(
            f'Function {func.__name__} '
            f'executed in {execution_time:.4f} seconds.'
        )
        return result

    return wrapper


def validate(func: Callable) -> Callable:
    '''
    Base decorator for validating function arguments.

    Objective:
        Redefines the @validate_call decorator to cache validation results.

    Uniqueness:
        Ensures validation is performed only once per unique set of arguments.
    '''
    validated_func = validate_call(func)

    @wraps(func)
    @lru_cache(maxsize=None)
    def wrapper(*args: Tuple[Any, ...], **kwargs: dict[str, Any]) -> Any:
        return validated_func(*args, **kwargs)

    return wrapper
