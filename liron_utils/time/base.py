from datetime import datetime

time_str_formats = {
    "full": "%Y-%m-%d %H:%M:%S,%f",  # 2023-07-11 00:33:28,881542
    "full_dash": "%Y-%m-%d %H-%M-%S-%f",  # 2023-07-11 00-33-28
    "yyyy-mm-dd": "%Y-%m-%d",  # 2023-07-11
    "dd/mm/yyyy": "%d/%m/%Y",  # 11/07/2023
    "mm/dd/yyyy": "%m/%d/%Y",  # 07/11/2023
    "d, m_name, y": "%b %d, %Y",  # Jul 11, 2023
    "d_name, m_name, y": "%a, %B %d, %Y",  # Tue, July 11, 2023
}


def get_time() -> datetime:
    """Return the current local time as a ``datetime`` object."""
    return datetime.now()


def get_time_str(fmt: str = time_str_formats["full_dash"]) -> str:
    """Return the current local time formatted as a string.

    Args:
        fmt: ``strftime`` format string; defaults to ``time_str_formats["full_dash"]``.

    Returns:
        Formatted current time.
    """
    return datetime.now().strftime(fmt)
