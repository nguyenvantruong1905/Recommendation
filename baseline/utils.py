import time

from functools import wraps
from math import trunc

from .__version__ import __version__


__all__ = ["get_version", "_timer"]


def get_version():
    return __version__


def _timer(text=""):
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
            start = time.process_time()
            result = func(*args, **kwargs)
            end = time.process_time()

            hours = trunc((end - start) / 3600)
            minutes = trunc((end - start - hours * 3600) / 60)
            seconds = round((end - start) % 60)

            if hours > 1:
                print(
                    text + "{} hours {} min and {} sec".format(hours, minutes, seconds)
                )
            elif hours == 1:
                print(
                    text + "{} hour {} min and {} sec".format(hours, minutes, seconds)
                )
            elif minutes >= 1:
                print(text + "{} min and {} sec".format(minutes, seconds))
            else:
                print(text + "{} sec".format(seconds))

            return result

        return wrapper

    return decorator
