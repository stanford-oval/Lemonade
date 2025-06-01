import asyncio
import functools
import pickle

from diskcache import Cache

cache = Cache(directory="./.diskcache")


def diskcache_cache(func):
    """
    Decorator that caches the results of the decorated function using disk-based storage.

    This decorator works with both synchronous and asynchronous functions. It creates a
    cache key based on the function's module, qualified name, arguments, and keyword
    arguments. The function results are serialized using pickle and stored in a disk cache.

    Args:
        func (callable): The function to be decorated. Can be either a regular function
                        or an async coroutine function.

    Returns:
        callable: A wrapped version of the original function that includes caching logic.
                 For async functions, returns an async wrapper; for sync functions,
                 returns a sync wrapper.

    Note:
        - Requires a global 'cache' object to be available in the module scope
        - Uses pickle for serialization, so all arguments and return values must be picklable
        - Cache keys are generated from function metadata and all provided arguments
        - Cached results persist across function calls until the cache is cleared

    Example:
        @diskcache_cache
        def expensive_computation(x, y):
            return x ** y

        @diskcache_cache
        async def async_api_call(url):
            # expensive async operation
            return await fetch_data(url)
    """
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            key = pickle.dumps(
                (func.__module__, func.__qualname__, args, tuple(kwargs.items()))
            )
            if key in cache:
                return pickle.loads(cache[key])
            result = await func(*args, **kwargs)
            cache[key] = pickle.dumps(result)
            return result

        return async_wrapper
    else:

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            key = pickle.dumps(
                (func.__module__, func.__qualname__, args, tuple(kwargs.items()))
            )
            if key in cache:
                return pickle.loads(cache[key])
            result = func(*args, **kwargs)
            cache[key] = pickle.dumps(result)
            return result

        return sync_wrapper
