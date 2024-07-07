import functools
import hashlib
import json
import os
from typing import Any, Callable

def disk_cache(cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = hashlib.md5(json.dumps((args, kwargs)).encode()).hexdigest()
            cache_file = os.path.join(cache_dir, f"{func.__name__}_{key}.json")
            
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    return json.load(f)
            
            result = await func(*args, **kwargs)
            with open(cache_file, 'w') as f:
                json.dump(result, f)
            
            return result
        return wrapper
    return decorator