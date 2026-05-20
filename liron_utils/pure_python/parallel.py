import functools
import multiprocessing as mp
import sys
import threading
import typing
import warnings
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

from .progress_bar import tqdm_

_T = typing.TypeVar("_T")

NUM_CPUS = mp.cpu_count()
NUM_PROCESSES_TO_USE = NUM_CPUS
NUM_THREADS_TO_USE = 20


def parallel_map(
    func: Callable[..., _T],
    iterable: Iterable[typing.Any],
    *,
    callback: Callable[..., typing.Any] | None = None,
    error_callback: Callable[..., typing.Any] | None = None,
    num_processes: int = NUM_PROCESSES_TO_USE,
    tqdm_kw: dict[str, typing.Any] | None = None,
    **kwargs: typing.Any,
) -> list[_T]:
    """Run ``func`` over ``iterable`` in parallel using a process pool.

    ``func`` must be a global (picklable) function. On macOS the ``fork`` start
    method is used; on Windows the default (``spawn``) is used, so anything not
    guarded by ``if __name__ == "__main__"`` may be executed in each worker.

    References:
        - https://stackoverflow.com/questions/64095876
        - https://stackoverflow.com/questions/72935231
        - https://superfastpython.com/multiprocessing-pool-issue-tasks

    Args:
        func: Function to evaluate in parallel; its first positional argument is
            the per-iteration value.
        iterable: Source of first-argument values for ``func``.
        callback: Per-task success callback.
        error_callback: Per-task error callback.
        num_processes: Worker process count; capped at ``min(NUM_CPUS, len(iterable))``.
        tqdm_kw: Forwarded to ``tqdm_``.
        **kwargs: Forwarded to ``func`` as keyword arguments.

    Returns:
        ``func`` outputs in the iteration order of ``iterable``.

    Example:
        >>> import time
        >>> def func(it, x, y):
        ...     time.sleep(1)
        ...     return (x + y) ** it
        >>>
        >>> if __name__ == "__main__":
        ...     out = parallel_map(func=func, iterable=range(100), num_processes=8, x=1, y=2)
    """
    if sys.platform == "darwin":  # in UNIX 'fork' can be used (faster but more dangerous)
        pool_cls = mp.get_context("fork").Pool
    else:  # In Windows only 'spawn' is available
        pool_cls = mp.Pool
        mp.Process()

    if num_processes > NUM_CPUS:
        warnings.warn(
            f"Requested number of processes {num_processes} is larger than number of CPUs {NUM_CPUS}.\n"
            f"For better performance, consider reducing 'num_processes'.",
            category=UserWarning,
        )
    iterable = list(iterable)
    num_processes = min(num_processes, NUM_CPUS, len(iterable))

    with pool_cls(processes=num_processes) as pool:
        func_partial = functools.partial(func, **kwargs)  # pass kwargs to func

        out_async = [
            pool.apply_async(
                func=func_partial,
                args=(i,),
                callback=callback,
                error_callback=error_callback,
            )
            for i in iterable
        ]

        out: list[_T] = []

        if tqdm_kw is None:
            tqdm_kw = {}

        for out_async_i in tqdm_(out_async, **tqdm_kw):
            try:
                out += [out_async_i.get()]

            except KeyboardInterrupt as e:
                raise e

            except Exception as e:  # pylint: disable=broad-exception-caught
                warnings.warn(f"Exception at index {out_async_i}: {e}")

    return out


def parallel_threading(
    func: Callable[..., _T],
    iterable: Iterable[typing.Any],
    lock: bool = False,
    num_threads: int = NUM_THREADS_TO_USE,
    tqdm_kw: dict[str, typing.Any] | None = None,
    **kwargs: typing.Any,
) -> list[_T | None]:
    """Run ``func`` over ``iterable`` in parallel using a thread pool.

    Args:
        func: Function evaluated per item; its first positional argument is the
            per-iteration value.
        iterable: Source of first-argument values for ``func``.
        lock: If True, serialize calls to ``func`` via a shared lock.
        num_threads: Worker thread count.
        tqdm_kw: Forwarded to ``tqdm_``.
        **kwargs: Forwarded to ``func`` as keyword arguments.

    Returns:
        ``func`` outputs in the iteration order of ``iterable``;
        any item whose call raised an exception is ``None``.
    """
    thread_lock: threading.Lock | None = threading.Lock() if lock else None

    def wrapped_func(index: int, item: typing.Any) -> tuple[int, _T | None]:
        try:
            if thread_lock is not None:
                with thread_lock:
                    result = func(item, **kwargs)
            else:
                result = func(item, **kwargs)
            return index, result
        except Exception as e:  # pylint: disable=broad-exception-caught
            warnings.warn(f"Exception at index {index}: {e}")
            return index, None

    iterable = list(iterable)
    out: list[_T | None] = [None] * len(iterable)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(wrapped_func, i, item) for i, item in enumerate(iterable)]

        if tqdm_kw is None:
            tqdm_kw = {}
        tqdm_kw = {"total": len(futures)} | tqdm_kw

        for future in tqdm_(as_completed(futures), **tqdm_kw):
            index, result = future.result()
            out[index] = result

    return out
