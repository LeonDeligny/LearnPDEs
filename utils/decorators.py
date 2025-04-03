'''
Customed decorators.
'''

# ======= Imports =======

from functools import lru_cache
from pydantic import validate_call

from typing import (
    Dict,
    Tuple,
    Callable,
)

from typing import Any

# ======= Functions =======


def validate(func: Callable) -> Callable:
    '''
    Redefines the @validate_call decorator to cache validation results.
    Ensures validation is performed only once per unique set of arguments.
    '''
    validated_func = validate_call(func)

    @lru_cache(maxsize=None)
    def wrapper(*args: Tuple[Any, ...], **kwargs: Dict[str, Any]) -> Any:
        return validated_func(*args, **kwargs)

    return wrapper

# ======= Main =======


if __name__ == '__main__':
    print('Nothing to execute.')
