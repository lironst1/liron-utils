# graphics.mpl + graphics.plotly Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `liron_utils/graphics/` into `graphics/common/` + `graphics/mpl/` (existing matplotlib code) and add a new plotly-idiomatic backend `graphics/plotly/` per `docs/superpowers/specs/2026-07-20-graphics-plotly-design.md`.

**Architecture:** `graphics/__init__` becomes side-effect-free and exposes only library-agnostic code from `graphics/common/`. The matplotlib module moves to `graphics/mpl/` (its rc-params import side effect moves with it). The plotly backend is a `Figure` class subclassing `plotly.graph_objects.Figure` built on `make_subplots`, plus a template port of the rc themes. Shared scipy/numpy math is extracted to `graphics/common/` so both backends use one implementation.

**Tech Stack:** Python ≥3.10, numpy, scipy, matplotlib (existing), plotly + kaleido (new), pytest, mypy strict, pylint.

## Global Constraints

- Follow repo typing conventions: `import typing` (qualified `typing.cast`/`typing.Any`), shape-parameterized aliases (`_Vec[_N] = np.ndarray[tuple[_N], np.dtype[typing.Any]]`), `typing.cast` at every numpy/scipy boundary, `T | None = None`.
- mypy strict must stay at 0 errors; pylint must stay 10.00/10. pylint has `useless-suppression` enabled — remove any `# pylint: disable` that turns out unneeded.
- Tests are collectable pytest files (`tests/graphics/test_*.py`) — a deliberate deviation from the repo's `_`-prefixed convention (approved in spec). Tests are also checked by mypy strict and pylint: annotate every test `-> None`, cast numpy literals.
- Docstrings: Google style, as in the existing `graphics` module. Keep comments minimal per repo style.
- No behavior change in the matplotlib backend — moves and extraction only.
- Fail fast: no defensive try/except; invalid input raises (ValueError/AssertionError).
- Commit after each task. Do NOT embed tuning constants in commit messages.
- Local venv already has plotly 6.7.0; kaleido is NOT installed locally (Task 3 installs it; the static-export test must skip gracefully when export is unavailable).
- If any `pip install`/tox run fails with `401 Error, Credentials not correct` (CodeArtifact-routed pip), run `exodigo-dev-link-pip-to-codeartifact` (alias `code-artifact`) once, then retry.
- Run tests with `python -m pytest <file> -v` from the repo root: `/Users/Liron.Stettiner/code_projects/liron-utils`.

---

### Task 1: Restructure — create `graphics/common/`, move matplotlib code to `graphics/mpl/`

Pure mechanical move: no logic changes, only relocation + import fixes + `__init__` rewiring.

**Files:**
- Create: `liron_utils/graphics/common/__init__.py`
- Create: `liron_utils/graphics/common/files.py` (function moved out of `axes.py`)
- Move (git mv): `liron_utils/graphics/utils/COLORS.py` → `liron_utils/graphics/common/COLORS.py`
- Move (git mv): `liron_utils/graphics/base.py` → `liron_utils/graphics/common/color_conversion.py`
- Move (git mv): `liron_utils/graphics/utils/etc.py` → `liron_utils/graphics/common/screen.py`
- Move (git mv): `liron_utils/graphics/axes.py` → `liron_utils/graphics/mpl/axes.py`
- Move (git mv): `liron_utils/graphics/plotting.py` → `liron_utils/graphics/mpl/plotting.py`
- Move (git mv): `liron_utils/graphics/utils/rc_params.py` → `liron_utils/graphics/mpl/utils/rc_params.py`
- Move (git mv): `liron_utils/graphics/utils/default_kwargs.py` → `liron_utils/graphics/mpl/utils/default_kwargs.py`
- Move (git mv): `liron_utils/graphics/utils/__init__.py` → `liron_utils/graphics/mpl/utils/__init__.py` (then edit)
- Rewrite: `liron_utils/graphics/__init__.py`
- Create: `liron_utils/graphics/mpl/__init__.py`
- Modify: `liron_utils/signal_processing/fitting.py:13`
- Test: `tests/graphics/test_imports.py`

**Interfaces:**
- Produces: `liron_utils.graphics.common` package exposing `COLORS` (module of hex constants), `hex2rgb`, `rgb2hex`, `get_pixel_color`, `get_savefig_file_name(file_name=None, time_dir=False, mkdir=False) -> str`; `liron_utils.graphics.mpl` exposing the entire previous `graphics` API (`Axes`, `set_props`, `update_rc_params`, ...). Later tasks import `from ..common import COLORS`, `from ..common.files import get_savefig_file_name`.

Note (spec deviation, intentional): the spec sketched one `common/colors.py` holding constants + conversions. Splitting into `COLORS.py` (constants, moved verbatim) + `color_conversion.py` (hex2rgb/rgb2hex) preserves the `COLORS.__all__` iteration semantics that `rc_params._resolve_color_list` relies on (`getattr(COLORS, name) for name in COLORS.__all__` must yield only color strings, never functions).

- [ ] **Step 1: Write the failing import test**

Create `tests/graphics/test_imports.py`:

```python
import pathlib

import liron_utils.graphics


def test_graphics_root_is_matplotlib_free() -> None:
    """graphics/__init__ and graphics/common must not reference matplotlib.

    A runtime sys.modules check is impossible here: importing liron_utils.graphics
    executes liron_utils/__init__, which imports signal_processing -> fitting ->
    graphics.mpl -> matplotlib. So assert at the source level instead.
    """
    root = pathlib.Path(liron_utils.graphics.__file__).parent
    sources = [root / "__init__.py", *(root / "common").glob("*.py")]
    assert len(sources) >= 6
    for source_file in sources:
        assert "matplotlib" not in source_file.read_text(), source_file


def test_graphics_root_exports() -> None:
    from liron_utils import graphics

    assert isinstance(graphics.COLORS.DARK_BLUE, str)
    assert graphics.hex2rgb("#FF0000") == (1.0, 0.0, 0.0)
    assert callable(graphics.get_savefig_file_name)
    assert callable(graphics.get_pixel_color)


def test_graphics_mpl_exports() -> None:
    from liron_utils.graphics import mpl as gr

    assert hasattr(gr, "Axes")
    assert hasattr(gr, "set_props")
    assert hasattr(gr, "update_rc_params")
    assert hasattr(gr, "COLORS")
    axes = gr.Axes(shape=(2, 1))
    assert axes.axs.shape == (2, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/graphics/test_imports.py -v`
Expected: FAIL — `liron_utils.graphics` currently imports matplotlib (first test), and `graphics.mpl` doesn't exist (third test).

- [ ] **Step 3: git mv the files**

```bash
cd /Users/Liron.Stettiner/code_projects/liron-utils
mkdir -p liron_utils/graphics/common liron_utils/graphics/mpl/utils
git mv liron_utils/graphics/utils/COLORS.py liron_utils/graphics/common/COLORS.py
git mv liron_utils/graphics/base.py liron_utils/graphics/common/color_conversion.py
git mv liron_utils/graphics/utils/etc.py liron_utils/graphics/common/screen.py
git mv liron_utils/graphics/axes.py liron_utils/graphics/mpl/axes.py
git mv liron_utils/graphics/plotting.py liron_utils/graphics/mpl/plotting.py
git mv liron_utils/graphics/utils/rc_params.py liron_utils/graphics/mpl/utils/rc_params.py
git mv liron_utils/graphics/utils/default_kwargs.py liron_utils/graphics/mpl/utils/default_kwargs.py
git mv liron_utils/graphics/utils/__init__.py liron_utils/graphics/mpl/utils/__init__.py
```

- [ ] **Step 4: Create `liron_utils/graphics/common/files.py`**

Move `get_savefig_file_name` out of `mpl/axes.py` (delete it there — it's the last function in the file, lines ~1103–1136) into this new file, body unchanged:

```python
import os

from ...files import MAIN_FILE_DIR, mkdirs
from ...time import TIME_STR, get_time_str


def get_savefig_file_name(
    file_name: str | None = None,
    time_dir: bool = False,
    mkdir: bool = False,
) -> str:
    """Build the full path for saving a figure.

    Args:
        file_name: If it has no directory component, the path resolves under
            ``<MAIN_FILE_DIR>/figs``. If None, an auto-generated ``"fig <ts>"`` name
            is used.
        time_dir: If True, append a timestamped subdirectory.
        mkdir: Create the directory if it doesn't exist.

    Returns:
        Resolved absolute (or near-absolute) file path.
    """
    if file_name is None or os.path.dirname(file_name) == "":
        dir_name = os.path.join(MAIN_FILE_DIR, "figs")
    else:
        dir_name = os.path.dirname(file_name)

    if time_dir:
        dir_name = os.path.join(dir_name, TIME_STR)

    if mkdir and not os.path.exists(dir_name):
        mkdirs(dir_name)

    if file_name is None:
        file_name = os.path.join(dir_name, f"fig {get_time_str()}")
    elif os.path.dirname(file_name) == "":
        file_name = os.path.join(dir_name, file_name)

    return file_name
```

- [ ] **Step 5: Create `liron_utils/graphics/common/__init__.py`**

```python
# flake8: noqa: F401

from . import COLORS
from .color_conversion import hex2rgb, rgb2hex
from .files import get_savefig_file_name
from .screen import get_pixel_color
```

- [ ] **Step 6: Rewrite `liron_utils/graphics/__init__.py`**

Replace the entire file with:

```python
# flake8: noqa: F401

from .common import COLORS, get_pixel_color, get_savefig_file_name, hex2rgb, rgb2hex
```

- [ ] **Step 7: Create `liron_utils/graphics/mpl/__init__.py`**

Carries over the old `graphics/__init__.py` content (including its TODO comment), with the import side effect now living here:

```python
# flake8: noqa: F401, F403

from ...pure_python import is_notebook
from ..common import COLORS, get_pixel_color, get_savefig_file_name, hex2rgb, rgb2hex
from .axes import *
from .plotting import *
from .utils import *

__all__ = [s for s in dir() if not s.startswith("_")]

update_rc_params()  # Change default MatPlotLib parameters (e.g, figure size, label size, grid, colors, etc.)

if is_notebook():
    update_rc_params("liron-utils-notebook")

# TODO:
#   - matplotlib.animation.FuncAnimation
#   - Transfer my default kwargs to merge with mpl.rcParams
```

- [ ] **Step 8: Fix imports in the moved files**

`liron_utils/graphics/mpl/axes.py` — replace the import block

```python
from ..files import MAIN_FILE_DIR, mkdirs, open_file
from ..pure_python.dicts import dl_to_ld
from ..time import TIME_STR, get_time_str
from .utils.default_kwargs import merge_kwargs
```

with

```python
from ...files import open_file
from ...pure_python.dicts import dl_to_ld
from ..common.files import get_savefig_file_name
from .utils.default_kwargs import merge_kwargs
```

(and delete the `get_savefig_file_name` function from the end of the file — moved in Step 4; `save_fig` and `set_props` keep calling it via the new import; `import os` stays, it's still used).

`liron_utils/graphics/mpl/plotting.py`:

```python
from ..signal_processing.base import interp1   →  from ...signal_processing.base import interp1
from ..uncertainties_math import to_numpy      →  from ...uncertainties_math import to_numpy
```

(`from .axes import _Axes` unchanged.)

`liron_utils/graphics/mpl/utils/rc_params.py`:

```python
from ..base import hex2rgb    →  from ...common.color_conversion import hex2rgb
from . import COLORS          →  from ...common import COLORS
```

`liron_utils/graphics/mpl/utils/default_kwargs.py`:

```python
from ...pure_python import MetaDict   →  from ....pure_python import MetaDict
from . import COLORS                  →  from ...common import COLORS
```

`liron_utils/graphics/mpl/utils/__init__.py` — replace entire content (drops the moved `etc`/`COLORS` and the vestigial `matplotlib.style` import):

```python
# flake8: noqa: F401, F403

from .default_kwargs import *
from .rc_params import *
```

- [ ] **Step 9: Update the consumer**

`liron_utils/signal_processing/fitting.py:13`:

```python
from .. import graphics as gr   →  from ..graphics import mpl as gr
```

(`gr.Axes` call sites at lines ~151 and ~318 keep working unchanged.)

- [ ] **Step 10: Run the tests**

Run: `python -m pytest tests/graphics/test_imports.py -v`
Expected: 3 passed.

Also verify the consumer still imports:

Run: `python -c "import liron_utils; from liron_utils.signal_processing import fitting; print('ok')"`
Expected: `ok`

- [ ] **Step 11: Run lint + typing on the touched code**

Run: `python -m mypy liron_utils/ tests/ && python -m pylint liron_utils/ tests/`
Expected: mypy 0 errors; pylint 10.00/10. Fix any stale-path fallout (e.g. relative-import depth) before proceeding.

- [ ] **Step 12: Commit**

```bash
git add -A liron_utils/ tests/graphics/test_imports.py
git commit -m "refactor: split graphics into common/ and mpl/ subpackages

graphics/__init__ is now side-effect free and matplotlib-free; the rc-params
import side effect moved to graphics.mpl. Library-agnostic code (COLORS,
hex/rgb conversion, save-path resolution, pixel picker) moved to
graphics.common. No behavior change.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Extract shared math to `common/fitting.py` and `common/spectra.py`

**Files:**
- Create: `liron_utils/graphics/common/fitting.py`
- Create: `liron_utils/graphics/common/spectra.py`
- Modify: `liron_utils/graphics/mpl/plotting.py` (delegate to the new modules)
- Test: `tests/graphics/test_common_math.py`

**Interfaces:**
- Consumes: `to_numpy` from `liron_utils.uncertainties_math`.
- Produces (used by mpl now, plotly in Tasks 8–10):
  - `curve_fit_prep_data(x, y, xerr, yerr, p_opt) -> tuple[x, y, xerr, yerr, p_opt]`
  - `curve_fit_confidence_band(fit_fcn, x_interp, p_opt, p_cov, n_std) -> tuple[fit_low, fit_high]`
  - `spectrum_display_data(spectrum, *, which="power", db=False, eps=1e-20) -> tuple[ydata, ylabel]`
  - `fft_data(x, fs=1.0, n=None, *, one_sided=True, normalize=False, input_time=True) -> tuple[spectrum, freqs]`
  - `periodogram_data(x, fs=1.0, n=None, *, window="boxcar", one_sided=True, normalize=False) -> tuple[psd, freqs]`
  - `frequency_response_data(b, a=1, fs=1.0, num_freq_points=512, *, one_sided=True) -> tuple[h, freqs]` (plotly-only consumer)
  - `impulse_response_data(b, a=1, dt=1, t=None, *, n=None) -> tuple[h, t_out, is_discrete]`
  - `spectrogram_data(y, fs=1.0, *, nfft=4096, window="blackmanharris", overlap_fraction=0.85, db=True, eps=1e-20) -> tuple[spec, freqs, times]` (plotly-only consumer)
  - `si_prefix_scale(max_value) -> tuple[scale_exponent, prefix_letter]`

- [ ] **Step 1: Write the failing tests**

Create `tests/graphics/test_common_math.py`:

```python
import typing

import numpy as np
import pytest

from liron_utils.graphics.common.fitting import curve_fit_confidence_band, curve_fit_prep_data
from liron_utils.graphics.common.spectra import (
    fft_data,
    frequency_response_data,
    impulse_response_data,
    periodogram_data,
    si_prefix_scale,
    spectrogram_data,
    spectrum_display_data,
)

_Array1D = np.ndarray[tuple[int], np.dtype[typing.Any]]


def test_spectrum_display_data_power_db() -> None:
    spectrum = typing.cast(_Array1D, np.array([1.0 + 0j, 10.0 + 0j]))
    ydata, ylabel = spectrum_display_data(spectrum, which="power", db=True)
    assert ylabel == "Power [dB]"
    np.testing.assert_allclose(ydata, [0.0, 20.0], atol=1e-6)


def test_spectrum_display_data_amp_phase() -> None:
    spectrum = typing.cast(_Array1D, np.array([3.0 + 4.0j]))
    ydata, ylabel = spectrum_display_data(spectrum, which="amp")
    assert ylabel == "Amplitude"
    np.testing.assert_allclose(ydata, [5.0])
    _, ylabel = spectrum_display_data(spectrum, which="phase")
    assert ylabel == "Phase [deg]"


def test_spectrum_display_data_invalid_which() -> None:
    with pytest.raises(ValueError):
        spectrum_display_data(typing.cast(_Array1D, np.ones(4)), which="bogus")


def test_fft_data_real_sine_peak() -> None:
    fs = 100.0
    t = np.arange(0, 1, 1 / fs)
    x = typing.cast(_Array1D, np.sin(2 * np.pi * 10 * t))
    spectrum, freqs = fft_data(x, fs=fs)
    assert spectrum.shape == freqs.shape == (len(t) // 2,)
    assert freqs[np.argmax(np.abs(spectrum))] == pytest.approx(10.0)


def test_fft_data_complex_is_two_sided() -> None:
    x = typing.cast(_Array1D, np.exp(2j * np.pi * 0.1 * np.arange(64)))
    spectrum, freqs = fft_data(x, fs=1.0, one_sided=True)  # auto-disabled for complex input
    assert spectrum.shape == freqs.shape == (64,)
    assert freqs[0] == pytest.approx(-0.5)


def test_periodogram_data_peak() -> None:
    fs = 100.0
    t = np.arange(0, 2, 1 / fs)
    x = typing.cast(_Array1D, np.sin(2 * np.pi * 10 * t))
    psd, freqs = periodogram_data(x, fs=fs)
    assert freqs[np.argmax(psd)] == pytest.approx(10.0)


def test_frequency_response_data_fir_dc_gain() -> None:
    b = typing.cast(_Array1D, np.ones(4) / 4)
    h, freqs = frequency_response_data(b, fs=1.0, num_freq_points=128)
    assert h.shape == freqs.shape == (128,)
    assert np.abs(h[0]) == pytest.approx(1.0)
    assert freqs[0] == pytest.approx(0.0)


def test_frequency_response_data_two_sided_centered() -> None:
    b = typing.cast(_Array1D, np.ones(4) / 4)
    h, freqs = frequency_response_data(b, fs=1.0, num_freq_points=128, one_sided=False)
    assert h.shape == freqs.shape == (128,)
    assert freqs[0] == pytest.approx(-0.5)
    assert np.all(np.diff(freqs) > 0)


def test_impulse_response_data_fir_discrete() -> None:
    b = typing.cast(_Array1D, np.array([1.0, 2.0, 3.0]))
    h, t_out, is_discrete = impulse_response_data(b, n=3)
    assert is_discrete
    np.testing.assert_allclose(h, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(t_out, [0.0, 1.0, 2.0])


def test_impulse_response_data_requires_t_or_n() -> None:
    with pytest.raises(ValueError):
        impulse_response_data(typing.cast(_Array1D, np.array([1.0])))


def test_spectrogram_data_shapes() -> None:
    fs = 8000.0
    y = typing.cast(_Array1D, np.random.default_rng(0).standard_normal(2**14))
    spec, freqs, times = spectrogram_data(y, fs=fs, nfft=256, overlap_fraction=0.5)
    assert spec.shape == (freqs.shape[0], times.shape[0])
    assert freqs[-1] == pytest.approx(fs / 2)


def test_si_prefix_scale() -> None:
    assert si_prefix_scale(500.0) == (0, "")
    assert si_prefix_scale(24_000.0) == (3, "K")
    assert si_prefix_scale(2.4e6) == (6, "M")
    assert si_prefix_scale(1.0e10) == (9, "G")


def test_curve_fit_prep_data_sorts_by_x() -> None:
    x = typing.cast(_Array1D, np.array([3.0, 1.0, 2.0]))
    y = typing.cast(_Array1D, np.array([30.0, 10.0, 20.0]))
    yerr = typing.cast(_Array1D, np.array([0.3, 0.1, 0.2]))
    x_out, y_out, xerr_out, yerr_out, p_opt = curve_fit_prep_data(x, y, None, yerr, None)
    np.testing.assert_allclose(x_out, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(y_out, [10.0, 20.0, 30.0])
    np.testing.assert_allclose(yerr_out, [0.1, 0.2, 0.3])
    assert xerr_out is None
    assert p_opt is None


def test_curve_fit_confidence_band_linear() -> None:
    def fit_fcn(x: _Array1D, a: float, b: float) -> _Array1D:
        return typing.cast(_Array1D, a * x + b)

    x = typing.cast(_Array1D, np.linspace(0, 1, 11))
    p_opt = typing.cast(_Array1D, np.array([2.0, 1.0]))
    p_cov = typing.cast(np.ndarray[tuple[int, int], np.dtype[typing.Any]], np.diag([0.01, 0.04]))
    fit_low, fit_high = curve_fit_confidence_band(fit_fcn, x, p_opt, p_cov, n_std=2)
    mid = fit_fcn(x, 2.0, 1.0)
    assert np.all(fit_low <= mid)
    assert np.all(mid <= fit_high)
    assert fit_high[0] - fit_low[0] == pytest.approx(2 * 2 * 0.2)  # at x=0 only the intercept perturbs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/graphics/test_common_math.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'liron_utils.graphics.common.fitting'`.

- [ ] **Step 3: Create `liron_utils/graphics/common/fitting.py`**

Bodies moved verbatim from the `Axes` staticmethods in `mpl/plotting.py` (drop the leading underscore — these are now shared public helpers):

```python
import typing
from collections.abc import Callable

import numpy as np

from ...uncertainties_math import to_numpy

_N = typing.TypeVar("_N", bound=int)
_K = typing.TypeVar("_K", bound=int)

_Vec = np.ndarray[tuple[_N], np.dtype[typing.Any]]
_Mat = np.ndarray[tuple[_N, _K], np.dtype[typing.Any]]


def curve_fit_prep_data(
    x: _Vec[_N],
    y: _Vec[_N],
    xerr: _Vec[_N] | None,
    yerr: _Vec[_N] | None,
    p_opt: _Vec[_K] | None,
) -> tuple[_Vec[_N], _Vec[_N], _Vec[_N] | None, _Vec[_N] | None, _Vec[_K] | None]:
    """Convert ``(x, y, xerr, yerr, p_opt)`` to numpy (via to_numpy) and sort by x.

    Args:
        x: 1D x-axis data of length N; uncertainties arrays are unpacked into ``(x, xerr)``.
        y: 1D y-axis data of length N; uncertainties arrays are unpacked into ``(y, yerr)``.
        xerr: 1D errors in x, length N. Ignored if x is an uncertainties array.
        yerr: 1D errors in y, length N. Ignored if y is an uncertainties array.
        p_opt: 1D fit parameters of length K; uncertainties unpacked to nominal values.

    Returns:
        ``(x, y, xerr, yerr, p_opt)`` as plain numpy arrays, sorted by x.
    """
    x, xerr = typing.cast(tuple[_Vec[_N], _Vec[_N] | None], to_numpy(x, xerr))
    y, yerr = typing.cast(tuple[_Vec[_N], _Vec[_N] | None], to_numpy(y, yerr))
    p_opt = typing.cast(_Vec[_K] | None, to_numpy(p_opt)[0])
    idx = np.argsort(x)
    x, y = typing.cast(_Vec[_N], x[idx]), typing.cast(_Vec[_N], y[idx])
    if xerr is not None:
        xerr = typing.cast(_Vec[_N], xerr[idx])
    if yerr is not None:
        yerr = typing.cast(_Vec[_N], yerr[idx])
    return x, y, xerr, yerr, p_opt


def curve_fit_confidence_band(
    fit_fcn: Callable[..., _Vec[_N]],
    x_interp: _Vec[_N],
    p_opt: _Vec[_K],
    p_cov: _Mat[_K, _K],
    n_std: float,
) -> tuple[_Vec[_N], _Vec[_N]]:
    """Compute the lower/upper fit envelope by perturbing each parameter ±n_std·σ.

    For each parameter ``p_opt[i]``, evaluate ``fit_fcn`` at ``p_opt[i] ± n_std·σ_i``
    (with σ_i from the diagonal of ``p_cov``) and take the element-wise min/max
    across all parameter perturbations to produce the band edges.

    Args:
        fit_fcn: Model function ``f(x, *params)`` that returns a 1D vector matching x.
        x_interp: 1D x-axis values of length N at which to evaluate the band.
        p_opt: 1D best-fit parameter values of length K.
        p_cov: K×K covariance matrix; its diagonal gives the per-parameter variances.
        n_std: Number of standard deviations spanning the band.

    Returns:
        ``(fit_low, fit_high)`` as 1D arrays of length N.
    """
    p_err = np.sqrt(np.diag(p_cov))
    fit_low = np.full(x_interp.size, np.inf)
    fit_high = np.full(x_interp.size, -np.inf)
    for i, (mid, err) in enumerate(zip(p_opt, p_err)):
        p_opt_i = p_opt.copy()
        p_opt_i[i] = mid - n_std * err
        low = fit_fcn(x_interp, *p_opt_i)
        p_opt_i[i] = mid + n_std * err
        high = fit_fcn(x_interp, *p_opt_i)
        fit_low = np.minimum(fit_low, np.minimum(low, high))
        fit_high = np.maximum(fit_high, np.maximum(low, high))
    return typing.cast(_Vec[_N], fit_low), typing.cast(_Vec[_N], fit_high)
```

- [ ] **Step 4: Create `liron_utils/graphics/common/spectra.py`**

The `spectrum_display_data`/`fft_data`/`periodogram_data`/`impulse_response_data` bodies are the non-rendering parts of the corresponding mpl inner functions, moved verbatim. `frequency_response_data`, `spectrogram_data`, `si_prefix_scale` are new (plotly consumers).

```python
# pylint: disable=invalid-name
import typing

import numpy as np
import scipy.signal

_Array1D = np.ndarray[tuple[int], np.dtype[typing.Any]]
_Array2D = np.ndarray[tuple[int, int], np.dtype[typing.Any]]

_SI_PREFIXES = {0: "", 3: "K", 6: "M", 9: "G", 12: "T"}


def spectrum_display_data(
    spectrum: _Array1D,
    *,
    which: str = "power",
    db: bool = False,
    eps: float = 1e-20,
) -> tuple[_Array1D, str]:
    """Convert a complex spectrum to displayable amplitude/power/phase data.

    Args:
        spectrum: 1D complex spectrum.
        which: One of ``"amp"``, ``"power"``, ``"phase"``.
        db: If True (and ``which`` is amp/power), convert to dB.
        eps: Floor added before ``log10`` to avoid log(0).

    Returns:
        ``(ydata, ylabel)`` — the display values and the matching y-axis label.

    Raises:
        ValueError: If ``which`` is not one of ``"amp"``, ``"power"``, ``"phase"``.
    """
    which = which.lower()
    ydata: _Array1D
    if which == "amp":
        ydata = typing.cast(_Array1D, np.abs(spectrum))
        ylabel = "Amplitude"
    elif which == "power":
        ydata = typing.cast(_Array1D, np.abs(spectrum) ** 2)
        ylabel = "Power"
    elif which == "phase":
        ydata = typing.cast(_Array1D, np.degrees(np.unwrap(np.angle(spectrum))))
        ylabel = "Phase [deg]"
    else:
        raise ValueError(f"which must be one of 'amp', 'power', or 'phase'. Got: {which}")

    if db and which in ("amp", "power"):
        ydata = typing.cast(_Array1D, 10 * np.log10(ydata + eps))
        ylabel += " [dB]"

    return ydata, ylabel


def fft_data(
    x: _Array1D,
    fs: float = 1.0,
    n: int | None = None,
    *,
    one_sided: bool = True,
    normalize: bool = False,
    input_time: bool = True,
) -> tuple[_Array1D, _Array1D]:
    """Compute an FFT spectrum and its frequency axis, one- or two-sided.

    Args:
        x: 1D input signal. Complex inputs always yield two-sided output.
        fs: Sampling frequency in Hz.
        n: FFT length. If None, uses ``len(x)``.
        one_sided: If True, return only the positive frequencies (auto-disabled for complex x).
        normalize: If True, scale the spectrum so its peak is 1.
        input_time: If True, treat ``x`` as time-domain and apply FFT.
            If False, treat ``x`` as already in the frequency domain.

    Returns:
        ``(spectrum, freqs)`` as 1D arrays (two-sided output is fft-shifted).
    """
    x_arr = typing.cast(_Array1D, np.asarray(x))
    if n is None:
        n = x_arr.shape[0]

    spectrum: _Array1D
    if input_time:
        if np.iscomplexobj(x_arr):
            one_sided = False
        spectrum = typing.cast(_Array1D, np.fft.fft(x_arr, n=n, axis=0))
    else:
        spectrum = x_arr.copy()

    if normalize:
        spectrum /= spectrum.max(axis=0)

    freqs = typing.cast(_Array1D, np.fft.fftfreq(n=n, d=1 / fs))

    if one_sided:
        spectrum = typing.cast(_Array1D, spectrum[: n // 2])
        freqs = typing.cast(_Array1D, freqs[: n // 2])
    else:
        spectrum = typing.cast(_Array1D, np.fft.fftshift(spectrum, axes=0))
        freqs = typing.cast(_Array1D, np.fft.fftshift(freqs))

    return spectrum, freqs


def periodogram_data(
    x: _Array1D,
    fs: float = 1.0,
    n: int | None = None,
    *,
    window: str = "boxcar",
    one_sided: bool = True,
    normalize: bool = False,
) -> tuple[_Array1D, _Array1D]:
    """Compute a PSD estimate via ``scipy.signal.periodogram``.

    Args:
        x: 1D input signal. Complex inputs always yield two-sided output.
        fs: Sampling frequency in Hz.
        n: FFT length; ``None`` uses ``len(x)``.
        window: Window function name accepted by ``scipy.signal.periodogram``.
        one_sided: If True, return only the positive frequencies (auto-disabled for complex x).
        normalize: If True, scale the PSD so its peak is 1.

    Returns:
        ``(psd, freqs)`` as 1D arrays (two-sided output is fft-shifted).
    """
    x_arr = typing.cast(_Array1D, np.asarray(x))
    if n is None:
        n = x_arr.shape[0]
    if np.iscomplexobj(x_arr):
        one_sided = False

    freqs, psd = scipy.signal.periodogram(
        x_arr,
        fs=fs,
        window=typing.cast(typing.Any, window),
        nfft=n,
        detrend=False,
        return_onesided=one_sided,
        scaling="density",
        axis=0,
    )

    if normalize:
        psd = psd / psd.max(axis=0)

    if not one_sided:
        psd = typing.cast(_Array1D, np.fft.fftshift(psd, axes=0))
        freqs = typing.cast(_Array1D, np.fft.fftshift(freqs))

    return typing.cast(_Array1D, psd), typing.cast(_Array1D, freqs)


def frequency_response_data(
    b: _Array1D | float,
    a: _Array1D | float = 1,
    fs: float | None = 1.0,
    num_freq_points: int = 512,
    *,
    one_sided: bool = True,
) -> tuple[_Array1D, _Array1D]:
    """Compute the frequency response of an LTI system given its ``(b, a)`` coefficients.

    For ``fs is None`` the system is treated as continuous-time (``scipy.signal.freqs``);
    otherwise ``scipy.signal.freqz`` is used.

    Args:
        b: 1D numerator coefficients (or a scalar).
        a: 1D denominator coefficients (set ``a == 1`` for FIR systems).
        fs: Sampling frequency in Hz. ``None`` selects continuous-time.
        num_freq_points: Number of frequency points to evaluate.
        one_sided: If True, evaluate ``[0, fs/2)``; otherwise ``[-fs/2, fs/2)``, centered.

    Returns:
        ``(h, freqs)`` — complex response and matching frequency axis (Hz, or rad/s/2π
        for continuous-time).
    """
    b_arr = typing.cast(_Array1D, np.atleast_1d(b))
    a_arr = typing.cast(_Array1D, np.atleast_1d(a))

    if fs is None:  # Continuous-time system
        w, h = scipy.signal.freqs(
            typing.cast(typing.Any, b_arr),
            typing.cast(typing.Any, a_arr),
            worN=num_freq_points,
        )
        return typing.cast(_Array1D, h), typing.cast(_Array1D, np.asarray(w) / (2 * np.pi))

    freqs, h = scipy.signal.freqz(b=b_arr, a=a_arr, fs=fs, worN=num_freq_points, whole=not one_sided)
    freqs = typing.cast(_Array1D, np.asarray(freqs))
    h = typing.cast(_Array1D, np.asarray(h))
    if not one_sided:
        # freqz(whole=True) evaluates [0, fs); recenter to [-fs/2, fs/2).
        upper = freqs >= fs / 2
        freqs = typing.cast(_Array1D, np.concatenate([freqs[upper] - fs, freqs[~upper]]))
        h = typing.cast(_Array1D, np.concatenate([h[upper], h[~upper]]))
    return h, freqs


def impulse_response_data(
    b: _Array1D | float,
    a: _Array1D | float = 1,
    dt: float | None = 1,
    t: _Array1D | None = None,
    *,
    n: int | None = None,
) -> tuple[_Array1D, _Array1D, bool]:
    """Compute the impulse response of an LTI system given its ``(b, a)`` coefficients.

    Args:
        b: 1D numerator coefficients (or a scalar).
        a: 1D denominator coefficients (set ``a == 1`` for FIR systems).
        dt: Sample period. If None, treats the system as continuous-time and requires ``t``.
        t: 1D time grid for the response. If None for discrete-time, ``n`` is used to build one.
        n: Sample count for the discrete-time response; required when ``t`` is None.

    Returns:
        ``(h, t_out, is_discrete)``.

    Raises:
        ValueError: For continuous-time when ``t`` is not given; for discrete-time when
            neither ``t`` nor ``n`` is given.
    """
    b_arr = typing.cast(_Array1D, np.atleast_1d(b))
    a_arr = typing.cast(_Array1D, np.atleast_1d(a))
    if len(b_arr) > len(a_arr):
        a_arr = typing.cast(_Array1D, np.pad(a_arr, (0, len(b_arr) - len(a_arr)), "constant", constant_values=0))
    elif len(a_arr) > len(b_arr):
        b_arr = typing.cast(_Array1D, np.pad(b_arr, (0, len(a_arr) - len(b_arr)), "constant", constant_values=0))

    if dt is None:  # Continuous-time system
        system: typing.Any = scipy.signal.lti(typing.cast(typing.Any, b_arr), typing.cast(typing.Any, a_arr))
        if t is None:
            raise ValueError("t should be given for continuous-time system.")
        t_out, h = scipy.signal.impulse(system, T=t)
        return typing.cast(_Array1D, h), typing.cast(_Array1D, t_out), False

    system = scipy.signal.dlti(typing.cast(typing.Any, b_arr), typing.cast(typing.Any, a_arr), dt=dt)
    if t is None:
        if n is None:
            raise ValueError("Either t or n should be given.")
        t = typing.cast(_Array1D, np.arange(0, n * dt, dt))
    t_out, h_seq = scipy.signal.dimpulse(system, n=len(t))
    h = typing.cast(_Array1D, np.squeeze(h_seq))
    return h, typing.cast(_Array1D, np.asarray(t_out)), True


def spectrogram_data(
    y: _Array1D,
    fs: float = 1.0,
    *,
    nfft: int = 4096,
    window: str = "blackmanharris",
    overlap_fraction: float = 0.85,
    db: bool = True,
    eps: float = 1e-20,
) -> tuple[_Array2D, _Array1D, _Array1D]:
    """Compute a spectrogram via ``scipy.signal.spectrogram``.

    Args:
        y: 1D time-domain signal.
        fs: Sample rate in Hz.
        nfft: Segment/FFT length.
        window: Window function name accepted by ``scipy.signal.get_window``.
        overlap_fraction: Segment overlap as a fraction of ``nfft``.
        db: If True, convert power to dB.
        eps: Floor added before ``log10`` to avoid log(0).

    Returns:
        ``(spec, freqs, times)`` — spectrogram matrix of shape (freqs, times) and its axes.
    """
    y_arr = typing.cast(_Array1D, np.asarray(y))
    freqs, times, spec = scipy.signal.spectrogram(
        y_arr,
        fs=fs,
        window=typing.cast(typing.Any, scipy.signal.get_window(window, nfft)),
        nperseg=nfft,
        noverlap=int(overlap_fraction * nfft),
        detrend=False,
        scaling="density",
        mode="psd",
    )
    spec = typing.cast(_Array2D, spec)
    if db:
        spec = typing.cast(_Array2D, 10 * np.log10(spec + eps))
    return spec, typing.cast(_Array1D, freqs), typing.cast(_Array1D, times)


def si_prefix_scale(max_value: float) -> tuple[int, str]:
    """Pick an SI prefix (engineering scale) for a positive axis maximum.

    Args:
        max_value: Largest value on the axis.

    Returns:
        ``(scale_exponent, prefix_letter)`` — e.g. ``(3, "K")``; divide values by
        ``10**scale_exponent`` and prepend the letter to the unit.
    """
    if max_value <= 1:
        return 0, ""
    scale = 3 * (int(np.log10(max_value)) // 3)
    scale = min(max(scale, 0), 12)
    return scale, _SI_PREFIXES[scale]
```

- [ ] **Step 5: Run the new tests**

Run: `python -m pytest tests/graphics/test_common_math.py -v`
Expected: all pass.

- [ ] **Step 6: Refactor `mpl/plotting.py` to delegate**

Add imports after the existing ones:

```python
from ..common.fitting import curve_fit_confidence_band, curve_fit_prep_data
from ..common.spectra import fft_data, impulse_response_data, periodogram_data, spectrum_display_data
```

Then:

1. Delete the `_curve_fit_prep_data` and `_curve_fit_confidence_band` staticmethods; replace their call sites in `_plot_data_and_curve_fit`:
   - `self._curve_fit_prep_data(x, y, xerr, yerr, p_opt)` → `curve_fit_prep_data(x, y, xerr, yerr, p_opt)`
   - `self._curve_fit_confidence_band(fit_fcn, x_interp, p_opt, p_cov, n_std)` → `curve_fit_confidence_band(fit_fcn, x_interp, p_opt, p_cov, n_std)`

2. Replace the body of `_plot_spectrum` (keep signature and docstring) with:

```python
        ydata, ylabel = spectrum_display_data(spectrum, which=which, db=db, eps=eps)

        line: list[typing.Any] = ax.plot(freqs, ydata, **plot_kw)

        if ax.get_xlabel() == "":
            ax.set_xlabel("Frequency [normalized]" if fs == 1.0 else "Frequency [Hz]")
        if ax.get_ylabel() == "":
            ax.set_ylabel(ylabel)

        return line
```

3. In the inner `_plot_fft`, replace everything between `x_arr = ...` and `if plot_spectrum_kw is None:` with:

```python
            spectrum, freqs = fft_data(
                x,
                fs=fs,
                n=n,
                one_sided=one_sided,
                normalize=normalize,
                input_time=input_time,
            )
```

4. In the inner `_plot_periodogram`, replace the body up to `if plot_spectrum_kw is None:` with:

```python
            psd, freqs = periodogram_data(x, fs=fs, n=n, window=window, one_sided=one_sided, normalize=normalize)
```

   (the `np.sqrt(psd)` passed to `self._plot_spectrum` and the return tuple stay as they are).

5. In the inner `_plot_impulse_response`, replace the body with:

```python
            h, t_out, is_discrete = impulse_response_data(b, a, dt=dt, t=t, n=n)
            line = ax.stem(t_out, h, **plot_kw) if is_discrete else ax.plot(t_out, h, **plot_kw)
            return (h, t_out), line
```

Do NOT touch `plot_frequency_response` — its mpl flow composes `plot_fft` and has different freqs semantics than the new `frequency_response_data` (which is a plotly-only consumer).

- [ ] **Step 7: Verify no mpl behavior change + full checks**

Run: `python -m pytest tests/graphics/ -v && python -m mypy liron_utils/ tests/ && python -m pylint liron_utils/ tests/`
Expected: tests pass, mypy 0 errors, pylint 10.00/10. If pylint flags now-unused imports in `mpl/plotting.py` (e.g. `scipy.signal` if nothing else uses it — `plot_frequency_response` still does), remove only genuinely unused ones.

- [ ] **Step 8: Commit**

```bash
git add liron_utils/graphics/ tests/graphics/test_common_math.py
git commit -m "refactor: extract shared plotting math to graphics.common

Curve-fit prep/confidence-band and FFT/periodogram/impulse-response
computation move out of the mpl plotting layer into graphics.common
(fitting.py, spectra.py) so the upcoming plotly backend shares one
implementation. Adds frequency-response, spectrogram, and SI-prefix
helpers for the plotly backend. No mpl behavior change.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Add plotly + kaleido dependencies and mypy configuration

**Files:**
- Modify: `pyproject.toml` (dependencies list ~line 20; mypy overrides block ~line 178)

**Interfaces:**
- Produces: `plotly` and `kaleido` importable in all tox envs; mypy treats `plotly.*` as untyped; the module `liron_utils.graphics.plotly.*` is allowed to subclass the untyped `go.Figure` (plotly ships no `py.typed`, so `go.Figure` is `Any` and strict mode's `disallow_subclassing_any` would reject `class Figure(go.Figure)`).

- [ ] **Step 1: Edit `pyproject.toml`**

In `[project] dependencies`, add (keeping rough alphabetical order):

```toml
    "kaleido",
```
after `"ipython",` and

```toml
    "plotly",
```
after `"pydantic",`.

In the existing `[[tool.mypy.overrides]]` module list, add `"plotly.*",` (before `"pytest_mock.*",`).

Append a new override block after the existing one:

```toml
[[tool.mypy.overrides]]
module = [
    "liron_utils.graphics.plotly.*",
]
disallow_subclassing_any = false
```

- [ ] **Step 2: Install kaleido into the dev venv**

Run: `python -m pip install kaleido`
Expected: installs successfully (if pip 401s against CodeArtifact, run `code-artifact` first and retry).

- [ ] **Step 3: Sanity-check the config parses**

Run: `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('ok')" && python -c "import plotly, kaleido; print(plotly.__version__)"`
Expected: `ok` and a plotly version ≥6.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add plotly and kaleido dependencies

mypy: ignore missing plotly stubs and allow subclassing go.Figure (Any)
in the upcoming graphics.plotly package.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Plotly templates + `graphics/plotly/__init__.py`

**Files:**
- Create: `liron_utils/graphics/plotly/templates.py`
- Create: `liron_utils/graphics/plotly/__init__.py`
- Test: `tests/graphics/test_plotly_templates.py`

**Interfaces:**
- Consumes: `COLORS` from `graphics.common`; `is_notebook` from `liron_utils.pure_python`.
- Produces: templates registered in `plotly.io.templates` under `"liron-utils-default"`, `"liron-utils-article"`, `"liron-utils-notebook"`; `register_templates() -> None`; `text_color_template(color: str) -> go.layout.Template`; module constant `COLORWAY: list[str]`. `pio.templates.default` set on import of `liron_utils.graphics.plotly`. `article`/`notebook` are **overlays** — combine as `"liron-utils-default+liron-utils-article"` (mirrors mpl's update-on-top semantics).

- [ ] **Step 1: Write the failing tests**

Create `tests/graphics/test_plotly_templates.py`:

```python
import plotly.io as pio

from liron_utils.graphics.common import COLORS

# Importing anything from the plotly subpackage runs its __init__, which registers the templates.
from liron_utils.graphics.plotly.templates import COLORWAY, text_color_template


def test_templates_registered() -> None:
    for name in ("liron-utils-default", "liron-utils-article", "liron-utils-notebook"):
        assert name in pio.templates


def test_default_template_active() -> None:
    assert str(pio.templates.default).startswith("liron-utils-default")


def test_default_template_colorway() -> None:
    template = pio.templates["liron-utils-default"]
    assert list(template.layout.colorway) == COLORWAY
    assert COLORWAY[0] == COLORS.DARK_BLUE


def test_default_template_trace_defaults() -> None:
    template = pio.templates["liron-utils-default"]
    assert len(template.data.surface) == 1
    assert len(template.data.heatmap) == 1


def test_text_color_template() -> None:
    template = text_color_template("#FF0000")
    assert template.layout.font.color == "#FF0000"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/graphics/test_plotly_templates.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'liron_utils.graphics.plotly'`.

- [ ] **Step 3: Create `liron_utils/graphics/plotly/templates.py`**

Translation notes baked in below: mpl 8×8 in @ 100 dpi → 800×800 px; `figure.facecolor: none` → transparent paper/plot background (works on light and dark pages); spines left+bottom only → `showline` without `mirror`; `axes.formatter.use_mathtext`/scientific limits → `exponentformat="power"`; the article style (seaborn-talk-derived) has all four spines → `mirror=True`, inward ticks, no grid, 10.4×7.15 in → 1040×715 px; notebook 6.4×4.8 in → 640×480 px with smaller fonts. mpl font sizes translate as: base 13, axis-title 15, figure-title 19 (default); article: base 13, axis-title 14, title 16; notebook: base 11, axis-title 11, title 13.

```python
import typing

import plotly.graph_objects as go
import plotly.io as pio

from ..common import COLORS

COLORWAY: list[str] = [
    COLORS.DARK_BLUE,
    COLORS.ORANGE_B,
    COLORS.PURPLE_D,
    COLORS.GREEN,
    COLORS.GREY_BROWN,
    COLORS.GOLD,
    COLORS.MAROON_D,
    COLORS.BLUE_C,
    COLORS.BLACK,
    COLORS.PINK,
]

_TRANSPARENT = "rgba(0, 0, 0, 0)"


def _axis_defaults(**overrides: typing.Any) -> dict[str, typing.Any]:
    return {
        "showgrid": True,
        "gridcolor": COLORS.LIGHT_GREY,
        "showline": True,
        "linecolor": COLORS.BLACK,
        "mirror": False,
        "zeroline": False,
        "ticks": "outside",
        "exponentformat": "power",
        "title": {"font": {"size": 15}},
    } | overrides


def _template_default() -> go.layout.Template:
    return go.layout.Template(
        layout={
            "colorway": COLORWAY,
            "font": {"size": 13},
            "title": {"font": {"size": 19}},
            "width": 800,
            "height": 800,
            "paper_bgcolor": _TRANSPARENT,
            "plot_bgcolor": _TRANSPARENT,
            "xaxis": _axis_defaults(),
            "yaxis": _axis_defaults(),
        },
        data={
            "surface": [go.Surface(colorscale="Viridis")],
            "heatmap": [go.Heatmap(colorscale="Inferno")],
        },
    )


def _template_article() -> go.layout.Template:
    axis = {
        "showgrid": False,
        "mirror": True,
        "ticks": "inside",
        "tickfont": {"size": 13},
        "title": {"font": {"size": 14}},
    }
    return go.layout.Template(
        layout={
            "font": {"size": 13},
            "title": {"font": {"size": 16}},
            "width": 1040,
            "height": 715,
            "xaxis": axis,
            "yaxis": axis,
        },
    )


def _template_notebook() -> go.layout.Template:
    axis = {"tickfont": {"size": 10}, "title": {"font": {"size": 11}}}
    return go.layout.Template(
        layout={
            "font": {"size": 11},
            "title": {"font": {"size": 13}},
            "width": 640,
            "height": 480,
            "xaxis": axis,
            "yaxis": axis,
        },
    )


def text_color_template(color: str) -> go.layout.Template:
    """Build an overlay template that recolors all text (titles, labels, ticks, legend).

    Args:
        color: Any CSS color string.

    Returns:
        Template to combine with a base, e.g.
        ``pio.templates.default = "liron-utils-default+my-color"`` after registering it,
        or by passing ``template=...`` to ``update_layout``.
    """
    return go.layout.Template(layout={"font": {"color": color}})


def register_templates() -> None:
    """Register the liron-utils templates in ``plotly.io.templates``."""
    pio.templates["liron-utils-default"] = _template_default()
    pio.templates["liron-utils-article"] = _template_article()
    pio.templates["liron-utils-notebook"] = _template_notebook()
```

- [ ] **Step 4: Create `liron_utils/graphics/plotly/__init__.py`**

All imports stay at the top (pylint `wrong-import-position`); the registration statements run after them. Task 5 will insert `from .figure import Figure` into this import block.

```python
# flake8: noqa: F401

import plotly.io as pio

from ...pure_python import is_notebook
from .templates import COLORWAY, register_templates, text_color_template

register_templates()

pio.templates.default = "liron-utils-default"
if is_notebook():
    pio.templates.default = "liron-utils-default+liron-utils-notebook"
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/graphics/test_plotly_templates.py -v`
Expected: all pass.

- [ ] **Step 6: Lint + typing**

Run: `python -m mypy liron_utils/ tests/ && python -m pylint liron_utils/ tests/`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add liron_utils/graphics/plotly/ tests/graphics/test_plotly_templates.py
git commit -m "feat: plotly templates port of the graphics rc themes

Registers liron-utils default/article/notebook templates and a
parameterized text-color overlay; importing graphics.plotly sets the
default template (notebook-aware), mirroring the mpl import behavior.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: `Figure` class — constructor, subplot grid, span layout

**Files:**
- Create: `liron_utils/graphics/plotly/figure.py`
- Modify: `liron_utils/graphics/plotly/__init__.py` (export `Figure`)
- Test: `tests/graphics/test_plotly_figure.py`

**Interfaces:**
- Produces: `Figure(shape=(1, 1), *, span_layout=None, subplot_type="xy", shared_xaxes=False, shared_yaxes=False, specs=None, **make_subplots_kw)` subclassing `go.Figure`. `span_layout` uses the mpl module's semantics: list of `[row_range, col_range]`, each an int `i` (→ `(i, i+1)`) or 0-based half-open `(start, end)`. All later-task methods hang off this class and use plotly's native 1-based `row=`/`col=`.

- [ ] **Step 1: Write the failing tests**

Create `tests/graphics/test_plotly_figure.py`:

```python
import plotly.graph_objects as go
import pytest

from liron_utils.graphics.plotly import Figure


def _num_xaxes(fig: go.Figure) -> int:
    return sum(1 for key in fig.to_dict()["layout"] if key.startswith("xaxis"))


def test_default_single_subplot() -> None:
    fig = Figure()
    assert isinstance(fig, go.Figure)
    fig.add_scatter(x=[1, 2], y=[3, 4], row=1, col=1)
    assert len(fig.data) == 1


def test_grid_shape() -> None:
    fig = Figure(shape=(2, 3))
    assert _num_xaxes(fig) == 6
    fig.add_scatter(x=[1], y=[1], row=2, col=3)
    assert len(fig.data) == 1


def test_span_layout() -> None:
    # 2x3 grid: one 2x2 span at top-left, singles at (0,2) and (1,2) -> 3 subplots.
    fig = Figure(shape=(2, 3), span_layout=[[(0, 2), (0, 2)], [0, (2, 3)]])
    assert _num_xaxes(fig) == 3


def test_span_layout_int_promotes_to_range() -> None:
    fig = Figure(shape=(2, 2), span_layout=[[0, (0, 2)]])
    assert _num_xaxes(fig) == 3  # top span + two bottom cells


def test_shared_xaxes() -> None:
    fig = Figure(shape=(2, 1), shared_xaxes=True)
    assert fig.layout.xaxis2.matches == "x" or fig.layout.xaxis.matches == "x2"


def test_scene_subplot_accepts_surface() -> None:
    fig = Figure(subplot_type="scene")
    fig.add_trace(go.Surface(z=[[1, 2], [3, 4]]), row=1, col=1)
    assert len(fig.data) == 1


def test_surface_on_xy_subplot_raises() -> None:
    fig = Figure()
    with pytest.raises(ValueError):
        fig.add_trace(go.Surface(z=[[1, 2], [3, 4]]), row=1, col=1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/graphics/test_plotly_figure.py -v`
Expected: FAIL — `ImportError: cannot import name 'Figure'`.

- [ ] **Step 3: Create `liron_utils/graphics/plotly/figure.py`**

```python
# pylint: disable=invalid-name
import typing
from collections.abc import Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

_SpanRange = int | tuple[int, int]
_SpecRow = list[dict[str, typing.Any] | None]


def _as_range(value: _SpanRange) -> tuple[int, int]:
    """Promote a single int row/col index to a 0-based half-open ``(start, end)`` range."""
    if isinstance(value, int):
        return value, value + 1
    return value


def _span_layout_to_specs(
    shape: tuple[int, int],
    span_layout: Sequence[Sequence[_SpanRange]],
    subplot_type: str,
) -> list[_SpecRow]:
    """Convert an mpl-style span layout to a ``make_subplots`` specs grid.

    Args:
        shape: ``(nrows, ncols)``.
        span_layout: List of ``[row_range, col_range]`` span specs; each range is an
            int ``i`` (interpreted as ``(i, i + 1)``) or a 0-based half-open
            ``(start, end)`` tuple.
        subplot_type: Subplot type applied to every cell (``"xy"``, ``"scene"``, ...).

    Returns:
        Specs grid for ``make_subplots``: span-covered cells are None, span anchors
        carry ``rowspan``/``colspan``, remaining cells are plain single subplots.
    """
    rows, cols = shape
    specs: list[_SpecRow] = [[{"type": subplot_type} for _ in range(cols)] for _ in range(rows)]
    for row_range, col_range in span_layout:
        r0, r1 = _as_range(row_range)
        c0, c1 = _as_range(col_range)
        for r in range(r0, r1):
            for c in range(c0, c1):
                specs[r][c] = None
        specs[r0][c0] = {"type": subplot_type, "rowspan": r1 - r0, "colspan": c1 - c0}
    return specs


class Figure(go.Figure):
    """A ``plotly.graph_objects.Figure`` with a subplot grid and domain plot helpers.

    All native plotly methods work directly (``add_scatter(row=, col=)``,
    ``update_layout``, ``show``, ...). Rows/cols in method calls are plotly's
    native 1-based indices; ``span_layout`` uses 0-based half-open ranges
    (mirroring ``graphics.mpl``'s ``grid_layout``).

    Example:
        >>> from liron_utils.graphics.plotly import Figure
        >>>
        >>> fig = Figure(shape=(2, 1), shared_xaxes=True)
        >>> fig.add_scatter(x=[0, 1], y=[1, 0], row=2, col=1)
        >>> fig.update_layout(title="Example")
        >>> fig.show()
    """

    def __init__(
        self,
        shape: tuple[int, int] = (1, 1),
        *,
        span_layout: Sequence[Sequence[_SpanRange]] | None = None,
        subplot_type: str = "xy",
        shared_xaxes: bool = False,
        shared_yaxes: bool = False,
        specs: list[_SpecRow] | None = None,
        **make_subplots_kw: typing.Any,
    ) -> None:
        """Create a figure holding an ``nrows × ncols`` subplot grid.

        Args:
            shape: ``(nrows, ncols)`` for the subplot grid.
            span_layout: Optional span specs (see :func:`_span_layout_to_specs`).
                Example: ``[[(0, 2), (0, 2)], [0, (2, 3)]]``.
            subplot_type: Type for all cells (``"xy"``, ``"scene"``, ``"polar"``, ...).
                Ignored for cells covered by an explicit ``specs``.
            shared_xaxes: Share x-axes across subplots (forwarded to ``make_subplots``).
            shared_yaxes: Share y-axes across subplots (forwarded to ``make_subplots``).
            specs: Explicit ``make_subplots`` specs grid; overrides ``span_layout``.
            **make_subplots_kw: Forwarded to ``plotly.subplots.make_subplots``
                (``subplot_titles``, ``row_heights``, ``column_widths``,
                ``horizontal_spacing``, ...).
        """
        rows, cols = shape
        if specs is None:
            if span_layout is not None:
                specs = _span_layout_to_specs(shape, span_layout, subplot_type)
            else:
                specs = [[{"type": subplot_type} for _ in range(cols)] for _ in range(rows)]
        base = make_subplots(
            rows=rows,
            cols=cols,
            specs=specs,
            shared_xaxes=shared_xaxes,
            shared_yaxes=shared_yaxes,
            **make_subplots_kw,
        )
        super().__init__(base)
        # make_subplots stores the grid on the instance it returns; without copying it,
        # row=/col= addressing on this subclass instance fails.
        self._grid_ref = base._grid_ref  # pylint: disable=protected-access
        self._grid_str = base._grid_str  # pylint: disable=protected-access
```

- [ ] **Step 4: Export from `liron_utils/graphics/plotly/__init__.py`**

Insert into the existing import block (before the `from .templates import ...` line, keeping imports at the top of the module — `figure.py` does not require registration to have run at its import time):

```python
from .figure import Figure
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/graphics/test_plotly_figure.py tests/graphics/test_plotly_templates.py -v`
Expected: all pass.

- [ ] **Step 6: Lint + typing**

Run: `python -m mypy liron_utils/ tests/ && python -m pylint liron_utils/ tests/`
Expected: clean (the `disallow_subclassing_any` override from Task 3 covers the subclass).

- [ ] **Step 7: Commit**

```bash
git add liron_utils/graphics/plotly/ tests/graphics/test_plotly_figure.py
git commit -m "feat: plotly Figure subclass with subplot grid and span layout

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: `Figure.save`

**Files:**
- Modify: `liron_utils/graphics/plotly/figure.py`
- Test: `tests/graphics/test_plotly_figure.py` (append)

**Interfaces:**
- Consumes: `get_savefig_file_name` from `graphics.common.files` (Task 1).
- Produces: `Figure.save(file_name: str | None = None, **write_kw) -> str` — returns the resolved path.

- [ ] **Step 1: Write the failing tests**

Append to `tests/graphics/test_plotly_figure.py`:

```python
def test_save_html(tmp_path: pathlib.Path) -> None:
    fig = Figure()
    fig.add_scatter(x=[1, 2], y=[3, 4])
    out = fig.save(str(tmp_path / "fig.html"))
    assert out == str(tmp_path / "fig.html")
    assert (tmp_path / "fig.html").stat().st_size > 0


def test_save_defaults_to_html(tmp_path: pathlib.Path) -> None:
    fig = Figure()
    out = fig.save(str(tmp_path / "figure"))
    assert out.endswith(".html")
    assert pathlib.Path(out).exists()


def test_save_png(tmp_path: pathlib.Path) -> None:
    pytest.importorskip("kaleido")
    fig = Figure()
    fig.add_scatter(x=[1, 2], y=[3, 4])
    try:
        out = fig.save(str(tmp_path / "fig.png"))
    except Exception as error:  # noqa: BLE001  # kaleido needs a Chrome binary; skip when absent
        pytest.skip(f"static image export unavailable: {error}")
    assert pathlib.Path(out).stat().st_size > 0
```

and add `import pathlib` to the file's imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/graphics/test_plotly_figure.py -v -k save`
Expected: FAIL — `AttributeError: ... has no attribute 'save'`.

- [ ] **Step 3: Implement `save` on `Figure`**

Add `import os` to `figure.py` imports and `from ..common.files import get_savefig_file_name`, then add the method:

```python
    def save(self, file_name: str | None = None, **write_kw: typing.Any) -> str:
        """Save the figure to disk, dispatching on the file extension.

        ``.html`` (or no extension) saves an interactive page via ``write_html``;
        any other extension (``.png``, ``.pdf``, ``.svg``, ...) is rendered via
        ``write_image`` (requires kaleido).

        Args:
            file_name: Output path. None or a bare filename resolves under
                ``<MAIN_FILE_DIR>/figs`` with an auto-generated name.
            **write_kw: Forwarded to ``write_html`` / ``write_image``.

        Returns:
            The full path the file was saved to.
        """
        file_name = get_savefig_file_name(file_name, mkdir=True)
        ext = os.path.splitext(file_name)[-1].lower()
        if ext == "":
            file_name += ".html"
            ext = ".html"
        if ext == ".html":
            self.write_html(file_name, **write_kw)
        else:
            self.write_image(file_name, **write_kw)
        return file_name
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/graphics/test_plotly_figure.py -v`
Expected: pass (png test may skip locally if Chrome isn't available to kaleido).

- [ ] **Step 5: Commit**

```bash
git add liron_utils/graphics/plotly/figure.py tests/graphics/test_plotly_figure.py
git commit -m "feat: Figure.save with html/static-image extension dispatch

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Basic plot helpers — `plot`, `plot_vlines`/`plot_hlines`, `draw_xy_lines`, `plot_errorbar`, `plot_filled_error`

**Files:**
- Modify: `liron_utils/graphics/plotly/figure.py`
- Test: `tests/graphics/test_plotly_plotting.py`

**Interfaces:**
- Consumes: `to_numpy` from `liron_utils.uncertainties_math`; `COLORS` from `graphics.common`.
- Produces (each returns `self` for chaining):
  - `plot(x, y=None, z=None, *, row=1, col=1, **scatter_kw)`
  - `plot_vlines(x=0, *, label=None, row=None, col=None, **vline_kw)`
  - `plot_hlines(y=0, *, label=None, row=None, col=None, **hline_kw)`
  - `draw_xy_lines(*, row=None, col=None, **line_kw)`
  - `plot_errorbar(x, y=None, xerr=None, yerr=None, *, row=1, col=1, **scatter_kw)`
  - `plot_filled_error(x, y=None, *, yerr=None, n_std=2, y_low=None, y_high=None, row=1, col=1, **scatter_kw)`

- [ ] **Step 1: Write the failing tests**

Create `tests/graphics/test_plotly_plotting.py`:

```python
import typing

import numpy as np
import pytest

from liron_utils.graphics.plotly import Figure

_Array1D = np.ndarray[tuple[int], np.dtype[typing.Any]]


def _arr(values: list[float]) -> _Array1D:
    return typing.cast(_Array1D, np.asarray(values, dtype=np.float64))


def test_plot_line() -> None:
    fig = Figure()
    fig.plot(_arr([0, 1, 2]), _arr([0, 1, 4]), name="curve")
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].mode == "lines"


def test_plot_y_only_uses_index_x() -> None:
    fig = Figure()
    fig.plot(_arr([5, 6, 7]))
    np.testing.assert_allclose(fig.data[0].x, [0, 1, 2])
    np.testing.assert_allclose(fig.data[0].y, [5, 6, 7])


def test_plot_3d() -> None:
    fig = Figure(subplot_type="scene")
    fig.plot(_arr([0, 1]), _arr([0, 1]), _arr([0, 1]))
    assert fig.data[0].type == "scatter3d"


def test_plot_vlines_label_on_last_only() -> None:
    fig = Figure()
    fig.plot_vlines(_arr([1.0, 2.0, 3.0]), label="cuts")
    shapes = fig.layout.shapes
    assert len(shapes) == 3
    assert [bool(s.showlegend) for s in shapes] == [False, False, True]
    assert shapes[-1].name == "cuts"


def test_plot_hlines_scalar() -> None:
    fig = Figure()
    fig.plot_hlines(2.5)
    assert len(fig.layout.shapes) == 1


def test_draw_xy_lines() -> None:
    fig = Figure()
    fig.draw_xy_lines()
    assert len(fig.layout.shapes) == 2


def test_plot_errorbar() -> None:
    fig = Figure()
    fig.plot_errorbar(_arr([1, 2, 3]), _arr([2, 4, 6]), yerr=_arr([0.1, 0.2, 0.3]))
    trace = fig.data[0]
    assert trace.mode == "markers"
    np.testing.assert_allclose(trace.error_y.array, [0.1, 0.2, 0.3])
    assert trace.error_x.array is None or len(trace.error_x.array) == 0


def test_plot_errorbar_y_only_rejects_errors() -> None:
    fig = Figure()
    with pytest.raises(AssertionError):
        fig.plot_errorbar(_arr([1, 2]), yerr=_arr([0.1, 0.2]))


def test_plot_filled_error_band() -> None:
    fig = Figure()
    fig.plot_filled_error(_arr([0, 1, 2]), _arr([1, 2, 3]), yerr=_arr([0.5, 0.5, 0.5]), n_std=2)
    assert len(fig.data) == 2
    np.testing.assert_allclose(fig.data[0].y, [0, 1, 2])  # low = y - 2*0.5
    np.testing.assert_allclose(fig.data[1].y, [2, 3, 4])  # high = y + 2*0.5
    assert fig.data[1].fill == "tonexty"


def test_plot_filled_error_explicit_bounds() -> None:
    fig = Figure()
    fig.plot_filled_error(_arr([0, 1]), y_low=_arr([0, 0]), y_high=_arr([1, 1]))
    assert len(fig.data) == 2


def test_plot_filled_error_rejects_mixed_args() -> None:
    fig = Figure()
    with pytest.raises(AssertionError):
        fig.plot_filled_error(_arr([0, 1]), _arr([1, 1]), y_low=_arr([0, 0]), y_high=_arr([2, 2]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/graphics/test_plotly_plotting.py -v`
Expected: FAIL — `AttributeError: 'Figure' object has no attribute 'plot'`.

- [ ] **Step 3: Implement the helpers on `Figure`**

Add imports to `figure.py`:

```python
import numpy as np

from ...uncertainties_math import to_numpy
from ..common import COLORS
```

and the type aliases near the top (after `_SpecRow`):

```python
_Array1D = np.ndarray[tuple[int], np.dtype[typing.Any]]
```

Methods:

```python
    def plot(
        self,
        x: _Array1D,
        y: _Array1D | None = None,
        z: _Array1D | None = None,
        *,
        row: int = 1,
        col: int = 1,
        **scatter_kw: typing.Any,
    ) -> "Figure":
        """Add a 2D line trace (or 3D when ``z`` is given).

        Args:
            x: 1D x data. If ``y`` is None, ``x`` is treated as y and the x-axis
                becomes ``range(len(x))``.
            y: 1D y data.
            z: 1D z data; when given, a ``Scatter3d`` is added (requires a
                ``"scene"`` subplot).
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **scatter_kw: Forwarded to ``go.Scatter`` / ``go.Scatter3d``.

        Returns:
            self (chainable).
        """
        if y is None:
            x, y = typing.cast(_Array1D, np.arange(len(x))), x
        scatter_kw = {"mode": "lines"} | scatter_kw
        trace: typing.Any
        if z is not None:
            trace = go.Scatter3d(x=x, y=y, z=z, **scatter_kw)
        else:
            trace = go.Scatter(x=x, y=y, **scatter_kw)
        self.add_trace(trace, row=row, col=col)
        return self

    def plot_vlines(
        self,
        x: _Array1D | float = 0,
        *,
        label: str | None = None,
        row: int | str | None = None,
        col: int | str | None = None,
        **vline_kw: typing.Any,
    ) -> "Figure":
        """Add one or more vertical lines (as shapes) spanning the y-axis.

        Args:
            x: Line positions — a scalar or a 1D array.
            label: Legend label, attached to the last line only.
            row: Target subplot row; None applies to all subplots.
            col: Target subplot column; None applies to all subplots.
            **vline_kw: Forwarded to ``add_vline`` (e.g. ``line_color``, ``line_dash``).

        Returns:
            self (chainable).
        """
        x_arr = typing.cast(_Array1D, np.atleast_1d(x))
        for i, xx in enumerate(x_arr):
            last = i == x_arr.shape[0] - 1
            self.add_vline(
                x=float(xx),
                row=row,
                col=col,
                name=label if last else None,
                showlegend=label is not None and last,
                **vline_kw,
            )
        return self

    def plot_hlines(
        self,
        y: _Array1D | float = 0,
        *,
        label: str | None = None,
        row: int | str | None = None,
        col: int | str | None = None,
        **hline_kw: typing.Any,
    ) -> "Figure":
        """Add one or more horizontal lines (as shapes) spanning the x-axis.

        Args:
            y: Line positions — a scalar or a 1D array.
            label: Legend label, attached to the last line only.
            row: Target subplot row; None applies to all subplots.
            col: Target subplot column; None applies to all subplots.
            **hline_kw: Forwarded to ``add_hline``.

        Returns:
            self (chainable).
        """
        y_arr = typing.cast(_Array1D, np.atleast_1d(y))
        for i, yy in enumerate(y_arr):
            last = i == y_arr.shape[0] - 1
            self.add_hline(
                y=float(yy),
                row=row,
                col=col,
                name=label if last else None,
                showlegend=label is not None and last,
                **hline_kw,
            )
        return self

    def draw_xy_lines(
        self,
        *,
        row: int | str | None = None,
        col: int | str | None = None,
        **line_kw: typing.Any,
    ) -> "Figure":
        """Draw bold ``x=0`` and ``y=0`` lines to highlight the origin.

        Args:
            row: Target subplot row; None applies to all subplots.
            col: Target subplot column; None applies to all subplots.
            **line_kw: Forwarded to ``add_hline`` / ``add_vline``.

        Returns:
            self (chainable).
        """
        line_kw = {"line_color": COLORS.DARK_GREY, "line_width": 2} | line_kw
        self.add_hline(y=0, row=row, col=col, **line_kw)
        self.add_vline(x=0, row=row, col=col, **line_kw)
        return self

    def plot_errorbar(
        self,
        x: _Array1D,
        y: _Array1D | None = None,
        xerr: _Array1D | None = None,
        yerr: _Array1D | None = None,
        *,
        row: int = 1,
        col: int = 1,
        **scatter_kw: typing.Any,
    ) -> "Figure":
        """Add a marker trace with error bars.

        If ``y`` is None, ``x`` is treated as the y data and the x-axis becomes
        ``range(len(x))``. ``x``/``y`` may be uncertainties arrays, in which case
        the errors are taken from the embedded uncertainties.

        Args:
            x: 1D x data (or y data when ``y`` is None).
            y: 1D y data.
            xerr: 1D errors in x. Ignored if ``x`` is an uncertainties array.
            yerr: 1D errors in y. Ignored if ``y`` is an uncertainties array.
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **scatter_kw: Forwarded to ``go.Scatter``.

        Returns:
            self (chainable).
        """
        if y is None:
            assert xerr is None and yerr is None, "If y is not given, xerr and yerr should not be given."
            x, y = typing.cast(_Array1D, np.arange(len(x))), x

        x, xerr = typing.cast(tuple[_Array1D, _Array1D | None], to_numpy(x, xerr))
        y, yerr = typing.cast(tuple[_Array1D, _Array1D | None], to_numpy(y, yerr))

        scatter_kw = {"mode": "markers", "marker": {"size": 8}} | scatter_kw
        error_bar_style = {"type": "data", "color": COLORS.RED_E, "thickness": 1.4}
        trace = go.Scatter(
            x=x,
            y=y,
            error_x=error_bar_style | {"array": xerr} if xerr is not None else None,
            error_y=error_bar_style | {"array": yerr} if yerr is not None else None,
            **scatter_kw,
        )
        self.add_trace(trace, row=row, col=col)
        return self

    def plot_filled_error(
        self,
        x: _Array1D,
        y: _Array1D | None = None,
        *,
        yerr: _Array1D | None = None,
        n_std: float = 2,
        y_low: _Array1D | None = None,
        y_high: _Array1D | None = None,
        row: int = 1,
        col: int = 1,
        **scatter_kw: typing.Any,
    ) -> "Figure":
        """Add a y±n_std·yerr confidence band as a filled area (two traces).

        Either ``(y, yerr)`` or ``(y_low, y_high)`` must be given.

        Args:
            x: 1D x data.
            y: 1D center values; required when ``y_low``/``y_high`` are None.
            yerr: 1D errors in y; the band is ``y ± n_std·yerr``.
            n_std: Number of standard deviations spanned by the band.
            y_low: 1D lower bound; required when ``y`` is None.
            y_high: 1D upper bound; required when ``y`` is None.
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **scatter_kw: Forwarded to the upper (filled) ``go.Scatter``.

        Returns:
            self (chainable).

        Raises:
            AssertionError: If neither ``(y, yerr)`` nor ``(y_low, y_high)`` is provided.
        """
        x = typing.cast(_Array1D, to_numpy(x)[0])
        if y is None:
            assert y_low is not None and y_high is not None, "(y, yerr) or (y_low, y_high) should be given."
        else:
            assert y_low is None and y_high is None, "(y, yerr) or (y_low, y_high) should be given."
            y, yerr = typing.cast(tuple[_Array1D, _Array1D | None], to_numpy(y, yerr))
            assert yerr is not None, "yerr should be given."
            y_low = typing.cast(_Array1D, y - n_std * yerr)
            y_high = typing.cast(_Array1D, y + n_std * yerr)

        fillcolor = scatter_kw.pop("fillcolor", "rgba(187, 187, 187, 0.4)")  # LIGHT_GRAY at 0.4 alpha
        band_style = {"mode": "lines", "line": {"width": 0}, "hoverinfo": "skip"}
        self.add_trace(go.Scatter(x=x, y=y_low, showlegend=False, **band_style), row=row, col=col)
        self.add_trace(
            go.Scatter(
                x=x,
                y=y_high,
                fill="tonexty",
                fillcolor=fillcolor,
                showlegend=scatter_kw.pop("showlegend", False),
                **band_style,
                **scatter_kw,
            ),
            row=row,
            col=col,
        )
        return self
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/graphics/test_plotly_plotting.py -v`
Expected: all pass. (If `add_vline(..., name=..., showlegend=...)` rejects the kwargs, plotly is too old — the deps require plotly ≥5.14 semantics; we verified 6.7.0 locally.)

- [ ] **Step 5: Lint + typing, then commit**

Run: `python -m mypy liron_utils/ tests/ && python -m pylint liron_utils/ tests/`
Expected: clean.

```bash
git add liron_utils/graphics/plotly/figure.py tests/graphics/test_plotly_plotting.py
git commit -m "feat: plotly basic plot helpers (lines, errorbar, filled band, origin lines)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Fit helpers — `plot_data_and_curve_fit`, `plot_data_and_lin_reg`

**Files:**
- Modify: `liron_utils/graphics/plotly/figure.py`
- Test: `tests/graphics/test_plotly_plotting.py` (append)

**Interfaces:**
- Consumes: `curve_fit_prep_data`, `curve_fit_confidence_band` from `graphics.common.fitting` (Task 2); `interp1` from `liron_utils.signal_processing.base`; `plot_errorbar`, `plot_filled_error`, `plot` (Task 7).
- Produces:
  - `plot_data_and_curve_fit(x, y, fit_fcn, *, xerr=None, yerr=None, p_opt=None, p_cov=None, n_std=2, interp_factor=20, curve_fit_scatter_kw=None, row=1, col=1, **errorbar_kw) -> Figure`
  - `plot_data_and_lin_reg(x, y, reg=None, *, xerr=None, yerr=None, reg_scatter_kw=None, row=1, col=1, **errorbar_kw) -> Figure`

- [ ] **Step 1: Write the failing tests**

Append to `tests/graphics/test_plotly_plotting.py`:

```python
def test_plot_data_and_curve_fit() -> None:
    def fit_fcn(x: _Array1D, a: float, b: float) -> _Array1D:
        return typing.cast(_Array1D, a * x + b)

    x = _arr([0, 1, 2, 3])
    y = _arr([1, 3, 5, 7])
    p_opt = _arr([2.0, 1.0])
    p_cov = typing.cast(
        np.ndarray[tuple[int, int], np.dtype[typing.Any]],
        np.diag([0.01, 0.01]),
    )
    fig = Figure()
    fig.plot_data_and_curve_fit(x, y, fit_fcn, yerr=_arr([0.1] * 4), p_opt=p_opt, p_cov=p_cov)
    # errorbar + fit line + 2 band traces
    assert len(fig.data) == 4
    assert fig.data[0].name == "Data"
    assert fig.data[1].name == "Curve fit"


def test_plot_data_and_curve_fit_without_cov() -> None:
    def fit_fcn(x: _Array1D, a: float) -> _Array1D:
        return typing.cast(_Array1D, a * x)

    fig = Figure()
    fig.plot_data_and_curve_fit(_arr([0, 1, 2]), _arr([0, 2, 4]), fit_fcn, p_opt=_arr([2.0]))
    assert len(fig.data) == 2  # no band without p_cov


def test_plot_data_and_lin_reg() -> None:
    rng = np.random.default_rng(0)
    x = typing.cast(_Array1D, np.arange(50, dtype=np.float64))
    y = typing.cast(_Array1D, 2 * x + rng.standard_normal(50))
    reg = scipy.stats.linregress(x, y)
    fig = Figure()
    fig.plot_data_and_lin_reg(x, y, reg)
    assert len(fig.data) == 2
    assert "slope" in fig.data[1].name
```

and add `import scipy.stats` to the file's imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/graphics/test_plotly_plotting.py -v -k "curve_fit or lin_reg"`
Expected: FAIL — attribute errors.

- [ ] **Step 3: Implement on `Figure`**

Add imports to `figure.py`:

```python
from collections.abc import Callable, Sequence

from ...signal_processing.base import interp1
from ..common.fitting import curve_fit_confidence_band, curve_fit_prep_data
```

(`Sequence` is already imported; merge into the existing line.) Methods:

```python
    def plot_data_and_curve_fit(  # pylint: disable=too-many-arguments
        self,
        x: _Array1D,
        y: _Array1D,
        fit_fcn: Callable[..., _Array1D],
        *,
        xerr: _Array1D | None = None,
        yerr: _Array1D | None = None,
        p_opt: _Array1D | None = None,
        p_cov: np.ndarray[tuple[int, int], np.dtype[typing.Any]] | None = None,
        n_std: float = 2,
        interp_factor: int = 20,
        curve_fit_scatter_kw: dict[str, typing.Any] | None = None,
        row: int = 1,
        col: int = 1,
        **errorbar_kw: typing.Any,
    ) -> "Figure":
        """Scatter the data with errorbars and overlay a smooth curve fit with a confidence band.

        The mid curve uses ``fit_fcn(x, *p_opt)`` on a denser x-axis (``interp_factor`` ×
        original sample count). When both ``p_opt`` and ``p_cov`` are given, a filled
        ±n_std·σ confidence band is added.

        Args:
            x: 1D x data (uncertainties arrays supported).
            y: 1D y data (uncertainties arrays supported).
            fit_fcn: Model function ``f(x, *params)``.
            xerr: 1D errors in x.
            yerr: 1D errors in y.
            p_opt: 1D best-fit parameters (e.g. from ``scipy.optimize.curve_fit``).
            p_cov: Covariance of the fit parameters.
            n_std: Standard deviations spanned by the confidence band.
            interp_factor: x-axis upsampling factor for the fit curve.
            curve_fit_scatter_kw: Forwarded to the fit-line ``plot`` call.
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **errorbar_kw: Forwarded to ``plot_errorbar`` for the data scatter.

        Returns:
            self (chainable).
        """
        errorbar_kw = {"name": "Data"} | errorbar_kw
        curve_fit_scatter_kw = {"name": "Curve fit"} | (curve_fit_scatter_kw or {})

        x, y, xerr, yerr, p_opt = curve_fit_prep_data(x, y, xerr, yerr, p_opt)
        self.plot_errorbar(x, y, xerr=xerr, yerr=yerr, row=row, col=col, **errorbar_kw)

        if p_opt is not None:
            x_interp = typing.cast(_Array1D, interp1(x, interp_factor * len(x)))
            self.plot(x_interp, fit_fcn(x_interp, *p_opt), row=row, col=col, **curve_fit_scatter_kw)
            if p_cov is not None:
                fit_low, fit_high = curve_fit_confidence_band(fit_fcn, x_interp, p_opt, p_cov, n_std)
                self.plot_filled_error(x_interp, y_low=fit_low, y_high=fit_high, row=row, col=col)
        return self

    def plot_data_and_lin_reg(
        self,
        x: _Array1D,
        y: _Array1D,
        reg: typing.Any = None,
        *,
        xerr: _Array1D | None = None,
        yerr: _Array1D | None = None,
        reg_scatter_kw: dict[str, typing.Any] | None = None,
        row: int = 1,
        col: int = 1,
        **errorbar_kw: typing.Any,
    ) -> "Figure":
        """Scatter the data with errorbars and overlay a linear regression line.

        Args:
            x: 1D x data (uncertainties arrays supported).
            y: 1D y data (uncertainties arrays supported).
            reg: Output of ``scipy.stats.linregress`` (line + slope/stderr/R² label).
            xerr: 1D errors in x.
            yerr: 1D errors in y.
            reg_scatter_kw: Forwarded to the regression-line ``plot`` call.
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **errorbar_kw: Forwarded to ``plot_errorbar`` for the data scatter.

        Returns:
            self (chainable).
        """
        errorbar_kw = {"name": "Data"} | errorbar_kw

        x, xerr = typing.cast(tuple[_Array1D, _Array1D | None], to_numpy(x, xerr))
        y, yerr = typing.cast(tuple[_Array1D, _Array1D | None], to_numpy(y, yerr))
        self.plot_errorbar(x, y, xerr=xerr, yerr=yerr, row=row, col=col, **errorbar_kw)

        if reg is not None:
            reg_scatter_kw = {
                "name": f"{errorbar_kw['name']} linreg: slope={reg.slope:.3f}±{reg.stderr:.3f}, "
                f"R²={reg.rvalue ** 2:.3f}",
            } | (reg_scatter_kw or {})
            y_reg = typing.cast(_Array1D, reg.slope * x + reg.intercept)
            self.plot(x, y_reg, row=row, col=col, **reg_scatter_kw)
        return self
```

- [ ] **Step 4: Run tests, lint, typing**

Run: `python -m pytest tests/graphics/test_plotly_plotting.py -v && python -m mypy liron_utils/ tests/ && python -m pylint liron_utils/ tests/`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add liron_utils/graphics/plotly/figure.py tests/graphics/test_plotly_plotting.py
git commit -m "feat: plotly curve-fit and linear-regression plot helpers

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Spectral helpers — `plot_fft`, `plot_periodogram`, `plot_frequency_response`, `plot_impulse_response`

**Files:**
- Modify: `liron_utils/graphics/plotly/figure.py`
- Test: `tests/graphics/test_plotly_plotting.py` (append)

**Interfaces:**
- Consumes: `fft_data`, `periodogram_data`, `frequency_response_data`, `impulse_response_data`, `spectrum_display_data` from `graphics.common.spectra` (Task 2); `COLORWAY` from `.templates`.
- Produces (these return computed data, not `self` — matching the mpl module's contract):
  - `plot_fft(x, fs=1.0, n=None, *, one_sided=True, normalize=False, input_time=True, which="power", db=False, eps=1e-20, row=1, col=1, **scatter_kw) -> tuple[spectrum, freqs]`
  - `plot_periodogram(x, fs=1.0, n=None, *, window="boxcar", one_sided=True, normalize=False, which="power", db=False, eps=1e-20, row=1, col=1, **scatter_kw) -> tuple[psd, freqs]`
  - `plot_frequency_response(b, a=1, fs=1.0, num_freq_points=512, *, one_sided=True, which="amp", db=False, eps=1e-20, normalize=False, row=1, col=1, **scatter_kw) -> tuple[h, freqs]`
  - `plot_impulse_response(b, a=1, dt=1, t=None, *, n=None, row=1, col=1, **scatter_kw) -> tuple[h, t_out]`

- [ ] **Step 1: Write the failing tests**

Append to `tests/graphics/test_plotly_plotting.py`:

```python
def test_plot_fft() -> None:
    fs = 100.0
    t = np.arange(0, 1, 1 / fs)
    x = typing.cast(_Array1D, np.sin(2 * np.pi * 10 * t))
    fig = Figure()
    spectrum, freqs = fig.plot_fft(x, fs=fs, db=True)
    assert len(fig.data) == 1
    assert freqs[np.argmax(np.abs(spectrum))] == pytest.approx(10.0)
    assert fig.layout.xaxis.title.text == "Frequency [Hz]"
    assert fig.layout.yaxis.title.text == "Power [dB]"


def test_plot_fft_normalized_axis_label() -> None:
    fig = Figure()
    fig.plot_fft(_arr([0, 1, 0, -1] * 8), fs=1.0)
    assert fig.layout.xaxis.title.text == "Frequency [normalized]"


def test_plot_periodogram() -> None:
    fs = 100.0
    t = np.arange(0, 2, 1 / fs)
    x = typing.cast(_Array1D, np.sin(2 * np.pi * 10 * t))
    fig = Figure()
    psd, freqs = fig.plot_periodogram(x, fs=fs)
    assert len(fig.data) == 1
    assert freqs[np.argmax(psd)] == pytest.approx(10.0)


def test_plot_frequency_response() -> None:
    b = typing.cast(_Array1D, np.ones(8) / 8)
    fig = Figure()
    h, freqs = fig.plot_frequency_response(b, fs=1.0, db=True)
    assert len(fig.data) == 1
    assert h.shape == freqs.shape
    assert np.abs(h[0]) == pytest.approx(1.0)


def test_plot_impulse_response_discrete_stem() -> None:
    fig = Figure()
    h, t_out = fig.plot_impulse_response(_arr([1, 2, 3]), n=3)
    assert len(fig.data) == 2  # stem segments + markers
    assert fig.data[0].mode == "lines"
    assert fig.data[1].mode == "markers"
    np.testing.assert_allclose(h, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(t_out, [0.0, 1.0, 2.0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/graphics/test_plotly_plotting.py -v -k "fft or periodogram or frequency_response or impulse"`
Expected: FAIL — attribute errors.

- [ ] **Step 3: Implement on `Figure`**

Add imports to `figure.py`:

```python
from ..common.spectra import (
    fft_data,
    frequency_response_data,
    impulse_response_data,
    periodogram_data,
    spectrum_display_data,
)
from .templates import COLORWAY
```

Add a private axis-title helper and the four methods:

```python
    def _set_spectrum_axis_titles(self, fs: float | None, ylabel: str, *, row: int, col: int) -> None:
        """Set frequency/quantity axis titles on one subplot."""
        xlabel = "Frequency [normalized]" if fs == 1.0 else "Frequency [Hz]"
        self.update_xaxes(title_text=xlabel, row=row, col=col)
        self.update_yaxes(title_text=ylabel, row=row, col=col)

    def plot_fft(  # pylint: disable=too-many-arguments
        self,
        x: _Array1D,
        fs: float = 1.0,
        n: int | None = None,
        *,
        one_sided: bool = True,
        normalize: bool = False,
        input_time: bool = True,
        which: str = "power",
        db: bool = False,
        eps: float = 1e-20,
        row: int = 1,
        col: int = 1,
        **scatter_kw: typing.Any,
    ) -> tuple[_Array1D, _Array1D]:
        """Plot the magnitude/power/phase spectrum of an FFT.

        Args:
            x: 1D input signal. Complex inputs always yield two-sided plots.
            fs: Sampling frequency in Hz. ``fs == 1.0`` produces a normalized axis label.
            n: FFT length. If None, uses ``len(x)``.
            one_sided: If True, plot only positive frequencies (auto-disabled for complex x).
            normalize: If True, scale the spectrum so its peak is 1.
            input_time: If True, treat ``x`` as time-domain and apply FFT.
            which: One of ``"amp"``, ``"power"``, ``"phase"``.
            db: If True (and ``which`` is amp/power), use a dB y-axis.
            eps: Floor added before ``log10`` to avoid log(0).
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **scatter_kw: Forwarded to the line trace.

        Returns:
            ``(spectrum, freqs)``.
        """
        spectrum, freqs = fft_data(x, fs=fs, n=n, one_sided=one_sided, normalize=normalize, input_time=input_time)
        ydata, ylabel = spectrum_display_data(spectrum, which=which, db=db, eps=eps)
        self.plot(freqs, ydata, row=row, col=col, **scatter_kw)
        self._set_spectrum_axis_titles(fs, ylabel, row=row, col=col)
        return spectrum, freqs

    def plot_periodogram(  # pylint: disable=too-many-arguments
        self,
        x: _Array1D,
        fs: float = 1.0,
        n: int | None = None,
        *,
        window: str = "boxcar",
        one_sided: bool = True,
        normalize: bool = False,
        which: str = "power",
        db: bool = False,
        eps: float = 1e-20,
        row: int = 1,
        col: int = 1,
        **scatter_kw: typing.Any,
    ) -> tuple[_Array1D, _Array1D]:
        """Plot the PSD estimate of a signal via ``scipy.signal.periodogram``.

        Args:
            x: 1D input signal. Complex inputs always yield two-sided plots.
            fs: Sampling frequency in Hz.
            n: FFT length; ``None`` uses ``len(x)``.
            window: Window name accepted by ``scipy.signal.periodogram``.
            one_sided: If True, plot only positive frequencies (auto-disabled for complex x).
            normalize: If True, scale the PSD so its peak is 1.
            which: One of ``"amp"``, ``"power"``, ``"phase"``.
            db: If True (and ``which`` is amp/power), use a dB y-axis.
            eps: Floor added before ``log10`` to avoid log(0).
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **scatter_kw: Forwarded to the line trace.

        Returns:
            ``(psd, freqs)``.
        """
        psd, freqs = periodogram_data(x, fs=fs, n=n, window=window, one_sided=one_sided, normalize=normalize)
        ydata, ylabel = spectrum_display_data(
            typing.cast(_Array1D, np.sqrt(psd)),
            which=which,
            db=db,
            eps=eps,
        )
        self.plot(freqs, ydata, row=row, col=col, **scatter_kw)
        self._set_spectrum_axis_titles(fs, ylabel, row=row, col=col)
        return psd, freqs

    def plot_frequency_response(  # pylint: disable=too-many-arguments
        self,
        b: _Array1D | float,
        a: _Array1D | float = 1,
        fs: float | None = 1.0,
        num_freq_points: int = 512,
        *,
        one_sided: bool = True,
        which: str = "amp",
        db: bool = False,
        eps: float = 1e-20,
        normalize: bool = False,
        row: int = 1,
        col: int = 1,
        **scatter_kw: typing.Any,
    ) -> tuple[_Array1D, _Array1D]:
        """Plot the frequency response of an LTI system given its ``(b, a)`` coefficients.

        Args:
            b: 1D numerator coefficients (or a scalar).
            a: 1D denominator coefficients (``a == 1`` for FIR systems).
            fs: Sampling frequency in Hz. ``None`` selects continuous-time.
            num_freq_points: Number of frequency points to evaluate.
            one_sided: If True, plot only positive frequencies.
            which: One of ``"amp"``, ``"power"``, ``"phase"``.
            db: If True (and ``which`` is amp/power), use a dB y-axis.
            eps: Floor added before ``log10`` to avoid log(0).
            normalize: If True, scale the response so its peak magnitude is 1.
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **scatter_kw: Forwarded to the line trace.

        Returns:
            ``(h, freqs)``.
        """
        h, freqs = frequency_response_data(b, a, fs=fs, num_freq_points=num_freq_points, one_sided=one_sided)
        if normalize:
            h = typing.cast(_Array1D, h / np.abs(h).max())
        ydata, ylabel = spectrum_display_data(h, which=which, db=db, eps=eps)
        self.plot(freqs, ydata, row=row, col=col, **scatter_kw)
        self._set_spectrum_axis_titles(fs, ylabel, row=row, col=col)
        return h, freqs

    def plot_impulse_response(
        self,
        b: _Array1D | float,
        a: _Array1D | float = 1,
        dt: float | None = 1,
        t: _Array1D | None = None,
        *,
        n: int | None = None,
        row: int = 1,
        col: int = 1,
        **scatter_kw: typing.Any,
    ) -> tuple[_Array1D, _Array1D]:
        """Plot the impulse response of an LTI system given its ``(b, a)`` coefficients.

        Discrete-time responses are drawn as a stem plot (markers plus vertical
        segments — plotly has no native stem); continuous-time as a line.

        Args:
            b: 1D numerator coefficients (or a scalar).
            a: 1D denominator coefficients (``a == 1`` for FIR systems).
            dt: Sample period. If None, treats the system as continuous-time and requires ``t``.
            t: 1D time grid. If None for discrete-time, ``n`` is used to build one.
            n: Sample count for the discrete-time response; required when ``t`` is None.
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **scatter_kw: Forwarded to the marker/line trace.

        Returns:
            ``(h, t_out)``.
        """
        h, t_out, is_discrete = impulse_response_data(b, a, dt=dt, t=t, n=n)
        if is_discrete:
            color = scatter_kw.pop("color", COLORWAY[0])
            segments_x = typing.cast(_Array1D, np.column_stack([t_out, t_out, np.full_like(t_out, np.nan)]).ravel())
            segments_y = typing.cast(_Array1D, np.column_stack([np.zeros_like(h), h, np.full_like(h, np.nan)]).ravel())
            self.add_trace(
                go.Scatter(x=segments_x, y=segments_y, mode="lines", line={"color": color}, showlegend=False),
                row=row,
                col=col,
            )
            self.add_trace(
                go.Scatter(x=t_out, y=h, mode="markers", marker={"color": color}, **scatter_kw),
                row=row,
                col=col,
            )
        else:
            self.plot(t_out, h, row=row, col=col, **scatter_kw)
        return h, t_out
```

- [ ] **Step 4: Run tests, lint, typing**

Run: `python -m pytest tests/graphics/test_plotly_plotting.py -v && python -m mypy liron_utils/ tests/ && python -m pylint liron_utils/ tests/`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add liron_utils/graphics/plotly/figure.py tests/graphics/test_plotly_plotting.py
git commit -m "feat: plotly spectral plot helpers (fft, periodogram, frequency/impulse response)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: `plot_specgram`

**Files:**
- Modify: `liron_utils/graphics/plotly/figure.py`
- Test: `tests/graphics/test_plotly_plotting.py` (append)

**Interfaces:**
- Consumes: `spectrogram_data`, `si_prefix_scale` from `graphics.common.spectra` (Task 2).
- Produces: `plot_specgram(y, fs=1.0, *, nfft=4096, window="blackmanharris", overlap_fraction=0.85, row=1, col=1, **heatmap_kw) -> tuple[spec, freqs, times]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/graphics/test_plotly_plotting.py`:

```python
def test_plot_specgram() -> None:
    fs = 48_000.0
    t = np.arange(0, 0.5, 1 / fs)
    y = typing.cast(_Array1D, np.sin(2 * np.pi * 1000 * t))
    fig = Figure()
    spec, freqs, times = fig.plot_specgram(y, fs=fs, nfft=1024)
    assert fig.data[0].type == "heatmap"
    assert spec.shape == (freqs.shape[0], times.shape[0])
    assert fig.layout.yaxis.title.text == "Frequency [KHz]"
    assert fig.layout.xaxis.title.text == "Time [sec]"
    np.testing.assert_allclose(fig.data[0].y, freqs / 1e3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/graphics/test_plotly_plotting.py -v -k specgram`
Expected: FAIL — attribute error.

- [ ] **Step 3: Implement on `Figure`**

Add `si_prefix_scale` and `spectrogram_data` to the `..common.spectra` import, then:

```python
    def plot_specgram(
        self,
        y: _Array1D,
        fs: float = 1.0,
        *,
        nfft: int = 4096,
        window: str = "blackmanharris",
        overlap_fraction: float = 0.85,
        row: int = 1,
        col: int = 1,
        **heatmap_kw: typing.Any,
    ) -> tuple[np.ndarray[tuple[int, int], np.dtype[typing.Any]], _Array1D, _Array1D]:
        """Plot a spectrogram of a 1D time-domain signal as a heatmap.

        Args:
            y: 1D time-domain signal.
            fs: Sample rate in Hz; ``fs != 1`` triggers SI-prefix scaling on the frequency axis.
            nfft: Segment/FFT length.
            window: Window name accepted by ``scipy.signal.get_window``.
            overlap_fraction: Segment overlap as a fraction of ``nfft``.
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **heatmap_kw: Forwarded to ``go.Heatmap``.

        Returns:
            ``(spec, freqs, times)`` — dB spectrogram matrix and its axes (freqs unscaled).
        """
        spec, freqs, times = spectrogram_data(y, fs=fs, nfft=nfft, window=window, overlap_fraction=overlap_fraction)

        scale, prefix = si_prefix_scale(float(freqs[-1])) if fs != 1 else (0, "")
        heatmap_kw = {"colorbar": {"title": {"text": "Power [dB]"}}} | heatmap_kw
        self.add_trace(go.Heatmap(x=times, y=freqs / 10**scale, z=spec, **heatmap_kw), row=row, col=col)
        self.update_xaxes(title_text="Time [sec]", row=row, col=col)
        self.update_yaxes(title_text=f"Frequency [{prefix}Hz]", showgrid=False, row=row, col=col)
        return spec, freqs, times
```

- [ ] **Step 4: Run tests, lint, typing**

Run: `python -m pytest tests/graphics/test_plotly_plotting.py -v && python -m mypy liron_utils/ tests/ && python -m pylint liron_utils/ tests/`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add liron_utils/graphics/plotly/figure.py tests/graphics/test_plotly_plotting.py
git commit -m "feat: plotly spectrogram heatmap helper

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 11: 3D + specialty — `plot_surf`, `plot_contour`

**Files:**
- Modify: `liron_utils/graphics/plotly/figure.py`
- Test: `tests/graphics/test_plotly_plotting.py` (append)

**Interfaces:**
- Produces:
  - `plot_surf(x, y, z, *, row=1, col=1, **surface_kw) -> Figure` — 1D x/y meshgridded; callable `z` supported; requires a `"scene"` subplot (plotly's own add_trace error covers the wrong-subplot case — a deliberate simplification of the spec's "assert scene", the native message is clear).
  - `plot_contour(x, y, z, contours=None, *, row=1, col=1, **contour_kw) -> Figure` — labeled contours; `contours` is `int` (number of levels) or `(start, end, size)`. (Deviation from mpl's explicit-levels array: plotly only supports uniformly spaced explicit levels.)

- [ ] **Step 1: Write the failing tests**

Append to `tests/graphics/test_plotly_plotting.py`:

```python
def test_plot_surf_1d_inputs_meshgridded() -> None:
    x = _arr([0, 1, 2])
    y = _arr([0, 1])
    z = typing.cast(np.ndarray[tuple[int, int], np.dtype[typing.Any]], np.ones((2, 3)))
    fig = Figure(subplot_type="scene")
    fig.plot_surf(x, y, z)
    assert fig.data[0].type == "surface"
    assert np.asarray(fig.data[0].z).shape == (2, 3)


def test_plot_surf_callable_z() -> None:
    fig = Figure(subplot_type="scene")
    fig.plot_surf(_arr([0, 1, 2]), _arr([0, 1, 2]), lambda x_grid, y_grid: x_grid + y_grid)
    assert np.asarray(fig.data[0].z).shape == (3, 3)


def test_plot_contour_int_levels() -> None:
    fig = Figure()
    fig.plot_contour(_arr([0, 1, 2]), _arr([0, 1]), typing.cast(
        np.ndarray[tuple[int, int], np.dtype[typing.Any]], np.arange(6, dtype=np.float64).reshape(2, 3)
    ), contours=5)
    assert fig.data[0].type == "contour"
    assert fig.data[0].ncontours == 5
    assert fig.data[0].contours.showlabels is True


def test_plot_contour_explicit_levels() -> None:
    fig = Figure()
    fig.plot_contour(_arr([0, 1, 2]), _arr([0, 1]), typing.cast(
        np.ndarray[tuple[int, int], np.dtype[typing.Any]], np.arange(6, dtype=np.float64).reshape(2, 3)
    ), contours=(0.0, 5.0, 1.0))
    assert fig.data[0].contours.start == 0.0
    assert fig.data[0].contours.size == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/graphics/test_plotly_plotting.py -v -k "surf or contour"`
Expected: FAIL — attribute errors.

- [ ] **Step 3: Implement on `Figure`**

```python
    def plot_surf(
        self,
        x: _Array1D | np.ndarray[tuple[int, int], np.dtype[typing.Any]],
        y: _Array1D | np.ndarray[tuple[int, int], np.dtype[typing.Any]],
        z: np.ndarray[tuple[int, int], np.dtype[typing.Any]]
        | Callable[[typing.Any, typing.Any], np.ndarray[tuple[int, int], np.dtype[typing.Any]]],
        *,
        row: int = 1,
        col: int = 1,
        **surface_kw: typing.Any,
    ) -> "Figure":
        """Plot a 3D surface ``z = f(x, y)`` (requires a ``"scene"`` subplot).

        Args:
            x: x-coordinates — 1D of length M (meshgrid applied) or 2D meshgrid.
            y: y-coordinates — 1D of length N (meshgrid applied) or 2D meshgrid.
            z: 2D z-values, or a callable ``f(x_grid, y_grid) -> z_grid``.
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **surface_kw: Forwarded to ``go.Surface``.

        Returns:
            self (chainable).
        """
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        if x_arr.ndim == 1:
            x_grid, y_grid = np.meshgrid(x_arr, y_arr)
        else:
            x_grid, y_grid = x_arr, y_arr
        z_grid = np.asarray(z(x_grid, y_grid)) if callable(z) else np.asarray(z)
        if z_grid.shape == tuple(reversed(x_grid.shape)):
            z_grid = z_grid.T

        self.add_trace(go.Surface(x=x_grid, y=y_grid, z=z_grid, **surface_kw), row=row, col=col)
        return self

    def plot_contour(
        self,
        x: _Array1D | np.ndarray[tuple[int, int], np.dtype[typing.Any]],
        y: _Array1D | np.ndarray[tuple[int, int], np.dtype[typing.Any]],
        z: np.ndarray[tuple[int, int], np.dtype[typing.Any]],
        contours: int | tuple[float, float, float] | None = None,
        *,
        row: int = 1,
        col: int = 1,
        **contour_kw: typing.Any,
    ) -> "Figure":
        """Plot labeled contour lines for the scalar field ``z = f(x, y)``.

        Args:
            x: x-coordinates — 1D of length M, or a 2D meshgrid (its first row is used).
            y: y-coordinates — 1D of length N, or a 2D meshgrid (its first column is used).
            z: 2D z-values of shape (N, M).
            contours: Number of levels (int), or explicit uniform levels as
                ``(start, end, size)``. None lets plotly pick.
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **contour_kw: Forwarded to ``go.Contour``.

        Returns:
            self (chainable).
        """
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        x_1d = x_arr[0, :] if x_arr.ndim == 2 else x_arr
        y_1d = y_arr[:, 0] if y_arr.ndim == 2 else y_arr

        contours_spec: dict[str, typing.Any] = {"showlabels": True}
        if isinstance(contours, int):
            # ncontours is honored while autocontour (default True) is on and contours.size is unset.
            contour_kw = {"ncontours": contours} | contour_kw
        elif contours is not None:
            start, end, size = contours
            contours_spec |= {"start": start, "end": end, "size": size}

        self.add_trace(
            go.Contour(x=x_1d, y=y_1d, z=z, contours=contours_spec, **contour_kw),
            row=row,
            col=col,
        )
        return self
```

- [ ] **Step 4: Run tests, lint, typing**

Run: `python -m pytest tests/graphics/test_plotly_plotting.py -v && python -m mypy liron_utils/ tests/ && python -m pylint liron_utils/ tests/`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add liron_utils/graphics/plotly/figure.py tests/graphics/test_plotly_plotting.py
git commit -m "feat: plotly 3D surface and labeled contour helpers

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 12: `plot_animation` — frames + play button + slider

**Files:**
- Modify: `liron_utils/graphics/plotly/figure.py`
- Test: `tests/graphics/test_plotly_plotting.py` (append)

**Interfaces:**
- Produces: `plot_animation(data, *, trace_type="auto", titles=None, frame_duration=100, row=1, col=1, **trace_kw) -> Figure`. `data` shape `[n_frames, h, w]` → Heatmap frames; `[n_frames, 2, n_pts]` → line Scatter frames; `trace_type` (`"heatmap"`/`"lines"`) disambiguates explicitly (required when `h == 2`). No gif export (mpl-only, per spec).

- [ ] **Step 1: Write the failing tests**

Append to `tests/graphics/test_plotly_plotting.py`:

```python
def test_plot_animation_images() -> None:
    data = typing.cast(
        np.ndarray[tuple[int, int, int], np.dtype[typing.Any]],
        np.random.default_rng(0).random((5, 8, 8)),
    )
    fig = Figure()
    fig.plot_animation(data)
    assert fig.data[0].type == "heatmap"
    assert len(fig.frames) == 5
    assert fig.layout.updatemenus[0].buttons[0].label == "Play"
    assert len(fig.layout.sliders[0].steps) == 5


def test_plot_animation_lines_with_titles() -> None:
    n_pts = 20
    data = typing.cast(
        np.ndarray[tuple[int, int, int], np.dtype[typing.Any]],
        np.stack([np.stack([np.arange(n_pts), np.sin(np.arange(n_pts) + i)]) for i in range(3)]),
    )
    fig = Figure()
    fig.plot_animation(data, trace_type="lines", titles=lambda i: f"frame {i}")
    assert fig.data[0].type == "scatter"
    assert len(fig.frames) == 3
    assert fig.frames[1].layout.title.text == "frame 1"


def test_plot_animation_ambiguous_shape_raises() -> None:
    data = typing.cast(
        np.ndarray[tuple[int, int, int], np.dtype[typing.Any]],
        np.zeros((4, 2, 10)),
    )
    fig = Figure()
    with pytest.raises(ValueError):
        fig.plot_animation(data)  # h == 2 is ambiguous; must pass trace_type
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/graphics/test_plotly_plotting.py -v -k animation`
Expected: FAIL — attribute error.

- [ ] **Step 3: Implement on `Figure`**

```python
    def plot_animation(
        self,
        data: np.ndarray[tuple[int, int, int], np.dtype[typing.Any]],
        *,
        trace_type: str = "auto",
        titles: Sequence[str] | Callable[[int], str] | None = None,
        frame_duration: int = 100,
        row: int = 1,
        col: int = 1,
        **trace_kw: typing.Any,
    ) -> "Figure":
        """Build an interactive animation (frames + play/pause button + slider).

        Args:
            data: Per-frame data — ``[n_frames, h, w]`` for images (Heatmap frames),
                ``[n_frames, 2, n_pts]`` for ``(x, y)`` line data (Scatter frames).
            trace_type: ``"heatmap"``, ``"lines"``, or ``"auto"``. ``"auto"`` infers
                from the shape and raises for the ambiguous ``h == 2`` case.
            titles: Per-frame figure titles — a sequence indexed by frame, or a
                callable ``(i) -> str``.
            frame_duration: Per-frame duration in milliseconds during playback.
            row: Target subplot row (1-based).
            col: Target subplot column (1-based).
            **trace_kw: Forwarded to the per-frame trace constructor.

        Returns:
            self (chainable).

        Raises:
            ValueError: If ``data`` is not 3D, or the shape is ambiguous with
                ``trace_type="auto"``.
        """
        data_arr = np.asarray(data)
        if data_arr.ndim != 3:
            raise ValueError(f"data must be 3D ([n_frames, h, w] or [n_frames, 2, n_pts]); got ndim={data_arr.ndim}.")
        n_frames = data_arr.shape[0]

        if trace_type == "auto":
            if data_arr.shape[1] == 2:
                raise ValueError("data with second dimension 2 is ambiguous; pass trace_type='heatmap' or 'lines'.")
            trace_type = "heatmap"
        if trace_type not in ("heatmap", "lines"):
            raise ValueError(f"trace_type must be one of 'auto', 'heatmap', 'lines'. Got: {trace_type}")

        def make_trace(frame: typing.Any) -> typing.Any:
            if trace_type == "lines":
                return go.Scatter(x=frame[0], y=frame[1], mode="lines", **trace_kw)
            return go.Heatmap(z=frame, **trace_kw)

        titles_list: list[str] | None
        if callable(titles):
            titles_list = [titles(i) for i in range(n_frames)]
        else:
            titles_list = list(titles) if titles is not None else None

        self.add_trace(make_trace(data_arr[0]), row=row, col=col)
        trace_index = len(self.data) - 1

        self.frames = [
            go.Frame(
                data=[make_trace(data_arr[i])],
                traces=[trace_index],
                name=str(i),
                layout={"title": {"text": titles_list[i]}} if titles_list is not None else None,
            )
            for i in range(n_frames)
        ]

        self.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [None, {"frame": {"duration": frame_duration, "redraw": True}, "fromcurrent": True}],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                        },
                    ],
                },
            ],
            sliders=[
                {
                    "steps": [
                        {
                            "label": str(i),
                            "method": "animate",
                            "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                        }
                        for i in range(n_frames)
                    ],
                },
            ],
        )
        return self
```

- [ ] **Step 4: Run tests, lint, typing**

Run: `python -m pytest tests/graphics/test_plotly_plotting.py -v && python -m mypy liron_utils/ tests/ && python -m pylint liron_utils/ tests/`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add liron_utils/graphics/plotly/figure.py tests/graphics/test_plotly_plotting.py
git commit -m "feat: plotly frame-based animation with play button and slider

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 13: Full validation — tox across all environments

**Files:**
- Possibly modify: whatever the checks flag (config or code fixes only, no new features).

- [ ] **Step 1: Visual smoke check of one figure**

Generate one representative HTML output and eyeball it (this is a load-bearing GUI output):

```bash
SMOKE_DIR=$(mktemp -d)
python - "$SMOKE_DIR" <<'EOF'
import sys

import numpy as np

from liron_utils.graphics.plotly import Figure

smoke_dir = sys.argv[1]
fs = 1000.0
t = np.arange(0, 1, 1 / fs)
x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.random.default_rng(0).standard_normal(t.size)
fig = Figure(shape=(2, 1))
fig.plot(t, x, row=1, col=1, name="signal")
fig.plot_fft(x, fs=fs, db=True, row=2, col=1)
fig.update_layout(title="Smoke check")
print(fig.save(f"{smoke_dir}/plotly_smoke.html"))
EOF
agent-browser open "file://$SMOKE_DIR/plotly_smoke.html"
agent-browser wait 2000
agent-browser screenshot "$SMOKE_DIR/plotly_smoke.png" --full
```

Inspect the screenshot: two stacked subplots, colored per the default colorway, grid on, spectrum axis titles present.

- [ ] **Step 2: Run the full tox suite**

If pip resolves through CodeArtifact (401 error), run `code-artifact` first.

Run: `tox`
Expected: `style`, `tests`, `lint`, `typing` all green (`tests` has `ignore_outcome = true`, but the new test files must actually pass — check the pytest summary line shows 0 failures).

- [ ] **Step 3: Fix anything flagged, re-run until green**

Typical candidates: pre-commit reformatting (black/isort line splits), pylint `useless-suppression` on a disable that turned out unnecessary, mypy complaints in test files. Apply minimal fixes; re-run `tox` until green.

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "chore: tox fixes for the graphics restructure and plotly backend

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

(Skip the commit if Step 3 required no changes.)

---

## Coverage map (spec → tasks)

| Spec requirement | Task |
|---|---|
| Side-effect-free `graphics/__init__`, `common/`, `mpl/` move, consumers | 1 |
| `common/fitting.py` + `common/spectra.py` extraction, mpl delegation | 2 |
| plotly + kaleido required deps; mypy strict handling | 3 |
| Template port (default/article/notebook/text-color), notebook-aware default | 4 |
| `Figure(go.Figure)` with `make_subplots`, `_grid_ref` copy, span layout, shared axes, scene subplots | 5 |
| `save` extension dispatch (html / static via kaleido) | 6 |
| plot, vlines/hlines (label-on-last), xy origin lines, errorbar (uncertainties-aware), filled band | 7 |
| curve-fit + confidence band, linear regression | 8 |
| fft / periodogram / frequency response / impulse response (stem) | 9 |
| specgram heatmap with SI-prefix frequency axis | 10 |
| surf (meshgrid, callable z), labeled contour | 11 |
| animation frames + play/slider; gif export skipped | 12 |
| tox green; collectable tests; visual verification | 13 |
| Skipped by design: gif export, `plot_line_collection`, `set_props`, `DefaultKwargs` | — |
