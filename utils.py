import os
import pickle

CACHE_DIR = "/tmp"


def file_cache():
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache_key = pickle.dumps({"args": args, "kwargs": kwargs})
            filename = f"{func.__name__}_{hash(cache_key)}.json"
            filepath = os.path.join(CACHE_DIR, filename)

            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    return pickle.load(f)

            result = func(*args, **kwargs)
            with open(filepath, "wb") as f:
                pickle.dump(result, f)

            return result

        return wrapper

    return decorator
