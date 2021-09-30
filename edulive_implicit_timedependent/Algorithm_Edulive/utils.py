import time

from functools import wraps
import datetime

from .__version__ import __version__


__all__ = [
    'get_version',
    '_timer'
]


def get_version():
    return __version__


def _timer(text=''):
    """Decorator, prints execution time of the decorated function.

    Parameters
    ----------
    text : str
        Text to print before time display.

    Examples
    --------
    >>> @_timer(text='Greetings took ')
    ... def say_hi():
    ...    time.sleep(1)
    ...    print("Hey! What's up!")
    ...
    >>> say_hi()
    Hey! What's up!
    Greetings took 1 sec
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(text + str(datetime.timedelta(seconds=end - start)))
            return result
        return wrapper

    return decorator
