# pylint: disable=consider-using-f-string

__all__ = [
    "BaseProgressBar",
    "TextProgressBar",
    "EnhancedTextProgressBar",
    "TqdmProgressBar",
    "HTMLProgressBar",
    "tqdm_",
]

import datetime
import sys
import time
import typing
from collections.abc import Callable, Generator, Iterable

from tqdm.auto import tqdm

_T = typing.TypeVar("_T")


class BaseProgressBar:
    """Abstract progress bar with shared timing helpers.

    Example:
        >>> import numpy as np
        >>> n_vec = np.linspace(0, 10, 100)
        >>> pbar = TextProgressBar(len(n_vec))
        >>> for n in n_vec:
        ...     pbar.update()
        >>> pbar.finished()
    """

    def __init__(self, iterations: int | float = 0, chunk_size: int | float = 10) -> None:
        """Initialize the bar.

        Args:
            iterations: Total iteration count (or duration).
            chunk_size: Progress threshold for updates (in percent).
        """
        self.n_total = float(iterations)
        self.n = 0
        self.p_chunk_size = chunk_size
        self.p_chunk = chunk_size
        self.t_start = time.time()
        self.t_done = self.t_start - 1

    def update(self) -> None:
        """Advance the progress bar by one step. Overridden by subclasses."""

    def total_time(self) -> float:
        """Return the total elapsed time, in seconds."""
        return self.t_done - self.t_start

    def time_elapsed(self) -> str:
        """Return elapsed time formatted as ``"%6.2fs"``."""
        return "%6.2fs" % (time.time() - self.t_start)

    def time_remaining_est(self, p: float) -> str:
        """Estimate the remaining time at progress percentage ``p``.

        Args:
            p: Current progress in percent (``0 < p <= 100``).

        Returns:
            ``"DD:HH:MM:SS"`` string of remaining time.
        """
        if 100 >= p > 0.0:
            t_r_est = (time.time() - self.t_start) * (100.0 - p) / p
        else:
            t_r_est = 0

        dd = datetime.datetime(1, 1, 1) + datetime.timedelta(seconds=t_r_est)
        time_string = "%02d:%02d:%02d:%02d" % (
            dd.day - 1,
            dd.hour,
            dd.minute,
            dd.second,
        )

        return time_string

    def finished(self) -> None:
        """Mark the bar as finished and record the completion time."""
        self.t_done = time.time()


class TextProgressBar(BaseProgressBar):
    """A simple text-based progress bar printed to stdout."""

    def update(self) -> None:
        self.n += 1
        n = self.n
        p = (n / self.n_total) * 100.0
        if p >= self.p_chunk:
            print(
                "%4.1f%%." % p
                + " Run time: %s." % self.time_elapsed()
                + " Est. time left: %s" % self.time_remaining_est(p),
            )
            sys.stdout.flush()
            self.p_chunk += self.p_chunk_size

    def finished(self) -> None:
        self.t_done = time.time()
        print("Total run time: %s" % self.time_elapsed())


class EnhancedTextProgressBar(BaseProgressBar):
    """A text-based bar that draws a fixed-width ``[*** ]`` progress widget."""

    def __init__(self, iterations: int | float = 0, chunk_size: int | float = 10) -> None:
        super().__init__(iterations, chunk_size)
        self.fill_char = "*"
        self.width = 25

    def update(self) -> None:
        self.n += 1
        n = self.n
        percent_done = int(round(n / self.n_total * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        prog_bar = "[" + self.fill_char * num_hashes + " " * (all_full - num_hashes) + "]"
        pct_place = (len(prog_bar) // 2) - len(str(percent_done))
        pct_string = "%d%%" % percent_done
        prog_bar = prog_bar[0:pct_place] + (pct_string + prog_bar[pct_place + len(pct_string) :])
        prog_bar += " Elapsed {} / Remaining {}".format(
            self.time_elapsed().strip(),
            self.time_remaining_est(percent_done),
        )
        print("\r", prog_bar, end="")
        sys.stdout.flush()

    def finished(self) -> None:
        self.t_done = time.time()
        print("\r", "Total run time: %s" % self.time_elapsed())


class TqdmProgressBar(BaseProgressBar):
    """A progress bar backed by the ``tqdm`` library."""

    def __init__(self, iterations: int | float = 0, chunk_size: int | float = 10, **kwargs: typing.Any) -> None:
        """Initialize the bar.

        Args:
            iterations: Total iteration count.
            chunk_size: Inherited from base; unused by ``tqdm``.
            **kwargs: Forwarded to ``tqdm.auto.tqdm``.
        """
        super().__init__(iterations, chunk_size)
        self.pbar = tqdm(total=iterations, **kwargs)
        self.t_start = time.time()
        self.t_done = self.t_start - 1

    def update(self) -> None:
        self.pbar.update()

    def finished(self) -> None:
        self.pbar.close()
        self.t_done = time.time()


class HTMLProgressBar(BaseProgressBar):  # pylint: disable=too-many-instance-attributes
    """HTML progress bar for IPython notebooks.

    Based on the IPython ProgressBar demo notebook at
    https://github.com/ipython/ipython/tree/master/examples/notebooks.

    Example:
        >>> import numpy as np
        >>> n_vec = np.linspace(0, 10, 100)
        >>> pbar = HTMLProgressBar(len(n_vec))
        >>> for n in n_vec:
        ...     pbar.update()
    """

    def __init__(self, iterations: int | float = 0, chunk_size: float = 1.0) -> None:
        super().__init__(iterations, chunk_size)

        import uuid  # pylint: disable=import-outside-toplevel

        from IPython.display import (  # pylint: disable=import-outside-toplevel
            HTML,
            Javascript,
            display,
        )

        self.display = display
        self._javascript_cls = Javascript
        self.divid = str(uuid.uuid4())
        self.textid = str(uuid.uuid4())
        self.pb = HTML(  # type: ignore[no-untyped-call]
            '<div style="border: 2px solid grey; width: 600px">\n  '
            f'<div id="{self.divid}" '
            'style="background-color: rgba(121,195,106,0.75); '
            'width:0%">&nbsp;</div>\n'
            "</div>\n"
            f'<p id="{self.textid}"></p>\n',
        )
        self.display(self.pb)  # type: ignore[no-untyped-call]

    def update(self) -> None:
        self.n += 1
        n = self.n
        p = (n / self.n_total) * 100.0
        if p >= self.p_chunk:
            lbl = "Elapsed time: %s. " % self.time_elapsed() + "Est. remaining time: %s." % self.time_remaining_est(p)
            js_code = (
                "$('div#%s').width('%i%%');"
                % (
                    self.divid,
                    p,
                )
                + f"$('p#{self.textid}').text('{lbl}');"
            )
            self.display(self._javascript_cls(js_code))  # type: ignore[no-untyped-call]
            self.p_chunk += self.p_chunk_size

    def finished(self) -> None:
        self.t_done = time.time()
        lbl = "Elapsed time: %s" % self.time_elapsed()
        # BUG: int(self.divid, 100.0) is invalid — self.divid is a UUID string, not a number
        js_code = (
            "$('div#%s').width('%i%%');" % int(self.divid, 100.0)  # type: ignore[call-overload]
            + f"$('p#{self.textid}').text('{lbl}');"
        )
        self.display(self._javascript_cls(js_code))  # type: ignore[no-untyped-call]


def tqdm_(
    iterable: Iterable[_T],
    desc: str | Callable[..., str] | None = None,
    total: int | float | None = None,
    *,
    disable: bool = False,
    unit: str = "it",
    postfix: dict[str, typing.Any] | Callable[..., dict[str, typing.Any] | None] | None = None,
    **kwargs: typing.Any,
) -> Generator[_T, None, None]:
    """Wrap ``iterable`` with a tqdm bar whose description/postfix can be functions of index.

    Args:
        iterable: Iterable to decorate.
        desc: Prefix string for the bar, or callable ``(i) -> str``.
        total: Expected iteration count; auto-detected from ``len(iterable)``.
        disable: If True, disable the bar entirely.
        unit: Unit label displayed in the bar.
        postfix: Dict (or callable ``(i) -> dict | None``) appended to the bar.
        **kwargs: Forwarded to ``tqdm.auto.tqdm``.

    Yields:
        Items from ``iterable`` after updating the bar.
    """
    if not callable(desc):
        desc_str = desc

        def desc_fn(_i: int) -> str | None:
            return desc_str

        resolved_desc: Callable[..., str | None] = desc_fn
    else:
        resolved_desc = desc

    if not callable(postfix):

        def postfix_fn(_i: int) -> dict[str, typing.Any] | None:
            return None

        resolved_postfix: Callable[..., dict[str, typing.Any] | None] = postfix_fn
    else:
        resolved_postfix = postfix

    if hasattr(iterable, "__len__"):
        total = len(iterable)  # type: ignore[arg-type]

    pbar = tqdm(total=total, disable=disable, unit=unit, **kwargs)

    for i, val in enumerate(iterable):
        pbar.set_description(resolved_desc(i))
        pbar.set_postfix(resolved_postfix(i))
        yield val
        pbar.update(1)
