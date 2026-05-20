import os
import typing

import numpy as np
import scipy.signal

from .base import array

_Array1D = np.ndarray[tuple[int], np.dtype[typing.Any]]
_Array3D = np.ndarray[tuple[int, int, int], np.dtype[typing.Any]]
_Array = np.ndarray[typing.Any, np.dtype[typing.Any]]


def resample(
    y: _Array,
    fs_old: int,
    fs_new: int,
    axis: int = -1,
    **kwargs: typing.Any,
) -> _Array:
    """Resample ``y`` along ``axis`` from ``fs_old`` to ``fs_new``.

    Args:
        y: Input signal (any dim).
        fs_old: Original sample rate.
        fs_new: Target sample rate.
        axis: Axis along which to resample.
        **kwargs: Forwarded to ``scipy.signal.resample``.

    Returns:
        Resampled array; identical to ``y`` if ``fs_old == fs_new``.
    """
    if fs_old == fs_new:
        return y
    n = round(y.shape[axis] * fs_new / fs_old)
    return typing.cast(_Array, scipy.signal.resample(y, n, axis=axis, **kwargs))


def _read_audio_metadata(
    files_arr: _Array1D,
    sf_module: typing.Any,
) -> tuple[int, int, int]:
    """Return ``(channels, sample_rate, frames)`` after asserting consistency across files.

    Args:
        files_arr: 1D array of audio file paths.
        sf_module: The imported ``soundfile`` module.

    Returns:
        ``(channels, sample_rate, frames)`` shared by every file.

    Raises:
        AssertionError: If any field differs between files.
    """
    channels, samplerate, frames = [], [], []
    for file in files_arr:
        info = sf_module.info(file)
        channels.append(info.channels)
        samplerate.append(info.samplerate)
        frames.append(info.frames)
    channels_arr = np.unique(channels)
    samplerate_arr = np.unique(samplerate)
    frames_arr = np.unique(frames)
    assert channels_arr.size == 1, "Inconsistent number of channels."
    assert samplerate_arr.size == 1, "Inconsistent sample rates."
    assert frames_arr.size == 1, "Inconsistent signal lengths."
    return int(channels_arr[0]), int(samplerate_arr[0]), int(frames_arr[0])


def audio_read(
    files: str | list[str],
    fs_new: int | None = None,
    always_3d: bool = False,
    **kwargs: typing.Any,
) -> tuple[_Array, int]:
    """Read one or more audio files into a single ndarray.

    Args:
        files: File names, specified either as a list or a single string.
        fs_new: Desired sample rate (resampled if different from the original).
        always_3d: If True, always return a 3D ``(n_files, n_frames, n_channels)``
            array; otherwise squeeze trailing/leading singleton dimensions.
        **kwargs: Forwarded to ``soundfile.read``.

    Returns:
        ``(audio, fs_orig)`` where ``audio`` is the (possibly squeezed) 3D array
        and ``fs_orig`` is the original sample rate.
    """
    import soundfile as sf  # pylint: disable=import-outside-toplevel

    files_arr = typing.cast(_Array1D, np.atleast_1d(array(files)))
    assert files_arr.ndim == 1
    channels, fs_orig, frames = _read_audio_metadata(files_arr, sf)

    kwargs.setdefault("start", 0)
    kwargs.setdefault("stop", frames)
    n_frames = kwargs["stop"] - kwargs["start"]

    audio = np.zeros((len(files_arr), n_frames, channels))
    for i, file in enumerate(files_arr):
        audio[i, :, :], _ = sf.read(file, always_2d=True, **kwargs)

    if fs_new is not None:
        audio = resample(audio, fs_orig, fs_new, axis=1)

    if not always_3d:
        audio = audio.squeeze()
    return audio, fs_orig


def audio_write(
    files: str | list[str],
    audio: _Array,
    fs: int,
    fs_save: int | None = None,
    mode: str = "w",
    **kwargs: typing.Any,
) -> None:
    """Write one or more audio signals to disk.

    Args:
        files: File names, specified either as a list or a single string.
        audio: Signals to write.
            - 3D: ``(n_files, n_frames, n_channels)``.
            - 2D ``(n_frames, 2)``: single stereo signal.
            - 2D ``(n_files, n_frames)``: that many mono signals.
            - 1D: single mono signal.
        fs: Original sample rate in Hz.
        fs_save: Desired sample rate to save at (resampled if different from ``fs``).
        mode: Save mode, ``"w+"`` (write) or ``"a"`` (append).
        **kwargs: Forwarded to ``soundfile.write``.

    Raises:
        ValueError: If ``mode`` is not one of ``"w+"`` or ``"a"``.
    """
    import soundfile as sf  # pylint: disable=import-outside-toplevel

    files_arr = typing.cast(_Array1D, np.atleast_1d(array(files)))
    audio = array(audio)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :, np.newaxis]
    elif audio.ndim == 2:
        if audio.shape[1] == 2:
            audio = audio[np.newaxis, :, :]
        else:
            audio = audio[:, :, np.newaxis]
    assert files_arr.ndim == 1 and audio.ndim == 3
    assert files_arr.size == audio.shape[0]

    if fs_save is None:
        fs_save = fs
    else:
        audio = resample(audio, fs, fs_save, axis=1)

    for i, file in enumerate(files_arr):
        if mode.lower() == "a":
            if not os.path.isfile(file):
                with sf.SoundFile(file, "w", samplerate=fs_save, **kwargs):
                    pass

            with sf.SoundFile(file, "r+") as h:
                h._update_frames(h.seek(0, sf.SEEK_END))  # pylint: disable=protected-access
                h.buffer_write(audio, dtype="float64")

        elif mode.lower() == "w+":
            sf.write(file, audio[i, :, :].squeeze(), fs_save, **kwargs)

        else:
            raise ValueError("Invalid declaration of variable `mode`.")


def time_array(n: int, fs: int) -> _Array1D:
    """Create a 1D time array from ``0`` to ``(n-1)/fs`` with ``n`` samples.

    Args:
        n: Number of samples.
        fs: Sample rate in Hz.

    Returns:
        1D array of length ``n``.
    """
    t = np.linspace(0, (n - 1) / fs, n)
    return typing.cast(_Array1D, t)


def time2samples(fs: int, *args: typing.Any) -> typing.Any:
    """Convert one or more time values [s] into sample counts at sample rate ``fs``.

    Args:
        fs: Sample rate in Hz.
        *args: Time values; ``None`` entries are passed through unchanged.

    Returns:
        Single sample count when one time value is given, otherwise a list.
    """
    args_list = list(args)
    for i, arg in enumerate(args_list):
        if arg is not None:
            args_list[i] = round(fs * arg)
    if len(args_list) == 1:
        return args_list[0]
    return args_list
