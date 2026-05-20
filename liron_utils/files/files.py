import os
import platform
import shutil
import subprocess
import typing
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

move_file = os.rename
remove_file = os.remove

_T = typing.TypeVar("_T")


def natural_sort(
    seq: Sequence[_T] | Iterable[_T],
    key: Callable[[_T], typing.Any] | None = None,
    *,
    reverse: bool = False,
    alg: typing.Any = None,
) -> list[_T]:
    """Sort an iterable naturally (e.g. ``["num2", "num3", "num5"]``).

    Args:
        seq: The input to sort.
        key: Single-argument key applied to each element (not recursive).
        reverse: Return the list in reversed sorted order.
        alg: ``natsort`` algorithm flag; defaults to ``natsort.ns.DEFAULT``.

    Returns:
        The sorted input as a list.

    Example:
        >>> natural_sort(["num3", "num5", "num2"])
        ['num2', 'num3', 'num5']
    """
    import natsort  # pylint: disable=import-outside-toplevel

    if alg is None:
        alg = natsort.ns.DEFAULT

    return natsort.natsorted(seq=seq, key=key, reverse=reverse, alg=alg)


def mkdirs(dirs: str | Path, *args: typing.Any, **kwargs: typing.Any) -> None:
    """Create directories (recursive, idempotent).

    Args:
        dirs: Target directory path.
        *args: Forwarded to ``os.makedirs``.
        **kwargs: Forwarded to ``os.makedirs`` (``exist_ok=True`` by default).
    """
    kwargs = {"exist_ok": True} | kwargs
    os.makedirs(dirs, *args, **kwargs)


def rmdir(dirname: str | Path) -> None:
    """Remove a directory, silently ignoring missing paths.

    Args:
        dirname: Directory to remove.
    """
    try:
        os.rmdir(dirname)
    except FileNotFoundError:
        pass


def _normalize_copy_paths(
    src: str | Path | Iterable[str | Path],
    dst: str | Path | Iterable[str | Path],
) -> tuple[list[str], list[str]]:
    """Normalize src/dst into matching same-length string lists.

    Args:
        src: Single source path or iterable of source paths.
        dst: Single destination path, or matching iterable.

    Returns:
        ``(src_list, dst_list)`` where both have equal length. If ``src`` is an
        iterable but ``dst`` is a single directory, ``dst_list`` is expanded by
        joining each ``src`` basename onto it.

    Raises:
        ValueError: If multiple sources are passed to a single non-directory dst,
            or when both iterables are passed but have mismatched lengths.
    """
    src_list: list[str] = [str(src)] if isinstance(src, (str, Path)) else [str(s) for s in src]
    dst_list: list[str] = [str(dst)] if isinstance(dst, (str, Path)) else [str(s) for s in dst]

    if len(src_list) == 1 and len(dst_list) > 1:
        raise ValueError("Cannot copy file/directory to multiple destinations.")
    if len(src_list) > 1 and len(dst_list) == 1:
        dst_list = [os.path.join(dst_list[0], os.path.basename(s)) for s in src_list]
    elif len(src_list) > 1 and len(src_list) != len(dst_list):
        raise ValueError("When copying multiple files, `dst` must be an iterable of the same length as `src`.")

    return src_list, dst_list


def _create_symlink(src: str, dst: str, *, target_is_directory: bool = False) -> None:
    """Create a symbolic link, replacing an existing non-directory dst.

    Args:
        src: Source file or directory.
        dst: Destination path for the symlink.
        target_is_directory: Forwarded to ``os.symlink``.

    Raises:
        ValueError: If ``src == dst`` or ``dst`` exists as a non-empty directory.
    """
    if src == dst:
        raise ValueError("Source and destination cannot be the same.")
    if os.path.isdir(dst):
        raise ValueError(f"Directory '{dst}' is not empty.")
    if os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst, target_is_directory=target_is_directory)


def _resolve_symlinks(src: str, dst: str, symlink: bool | None) -> tuple[str, str, bool]:
    """Follow symlinks in src/dst and decide whether to symlink the result.

    Args:
        src: Source path, possibly a symlink (followed once).
        dst: Destination path, possibly a symlink (followed once).
        symlink: Tri-state preference: ``True`` to force, ``False`` to forbid,
            ``None`` to inherit (symlink iff ``src`` is a symlink).

    Returns:
        ``(resolved_src, resolved_dst, do_symlink)``.
    """
    if os.path.islink(src):
        src = os.readlink(src).replace("\\\\?\\", "")
        if symlink is None:
            symlink = True
    if os.path.islink(dst):
        dst = os.readlink(dst).replace("\\\\?\\", "")
    return src, dst, bool(symlink)


def _copy_one(src: str, dst: str, *, symlink: bool) -> None:
    """Copy a single resolved src to dst, or symlink it.

    Args:
        src: Resolved source path (file or directory).
        dst: Resolved destination path.
        symlink: If True, create a symlink instead of copying.

    Raises:
        ValueError: If ``src`` is neither a file nor a directory.
    """
    if os.path.isdir(src):
        mkdirs(os.path.dirname(dst))
        if symlink:
            _create_symlink(src, dst, target_is_directory=True)
        else:
            shutil.copytree(src, dst, dirs_exist_ok=True)
    elif os.path.isfile(src):
        if not os.path.splitext(dst)[1]:
            dst = os.path.join(dst, os.path.basename(src))
        mkdirs(os.path.dirname(dst))
        if symlink:
            _create_symlink(src, dst, target_is_directory=False)
        else:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Source {src} is neither a file nor a directory.")


def copy(
    src: str | Path | Iterable[str | Path],
    dst: str | Path | Iterable[str | Path],
    *,
    overwrite: bool = True,
    symlink: bool | None = None,
) -> None:
    """Copy file(s) or directories, or create symbolic links.

    Args:
        src: Source file(s) or directory(ies).
        dst: Either a single target path/directory (when ``src`` is a single path)
            or an iterable of the same length as ``src``.
        overwrite: Overwrite destination if it exists.
        symlink: If True, create symbolic links instead of copying.
            If None, follow ``src``'s type — symlink iff ``src`` is itself a symlink.
    """
    src_list, dst_list = _normalize_copy_paths(src, dst)
    for s, d in zip(src_list, dst_list):
        if not overwrite and os.path.exists(d):
            continue
        s_resolved, d_resolved, do_link = _resolve_symlinks(s, d, symlink)
        _copy_one(s_resolved, d_resolved, symlink=do_link)


def open_file(file: str | Path) -> None:
    """Open a file with the OS default application.

    Args:
        file: Path to the file to open.
    """
    if platform.system() == "Windows":
        os.startfile(file)  # type: ignore[attr-defined]  # pylint: disable=no-member
    elif platform.system() == "Darwin":
        subprocess.run(["open", str(file)], check=True)
    else:
        subprocess.run(["xdg-open", str(file)], check=True)


def wslpath(filename: str | Path) -> Path:
    """Convert a Windows-style path to a WSL/Linux path when needed.

    Args:
        filename: Input path, possibly containing backslashes or quotes.

    Returns:
        Sanitized POSIX-style ``Path``; identical input is returned unchanged.
    """
    filename = str(filename).strip().replace('"', "").replace("'", "")
    if "\\" in str(filename):
        filename = subprocess.run(
            ["wslpath", "-u", filename],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()  # Convert to linux path

    return Path(filename)
