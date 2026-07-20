# Design: Restructure `graphics` into `graphics.mpl` + `graphics.plotly`

**Date:** 2026-07-20
**Status:** Approved

## Goal

Add a plotly-based graphics backend alongside the existing matplotlib one. The plotly
module is a fresh, plotly-idiomatic API (not a port of the `Axes`/`set_props`/`_vectorize`
design). The matplotlib module stays fully functional; backwards compatibility of import
paths is explicitly **not** required.

## Decisions (from brainstorming)

| Topic | Decision |
|---|---|
| Strategy | Restructure: `graphics/mpl/` + `graphics/plotly/`, side-effect-free `graphics/__init__` |
| API shape | Plotly-idiomatic redesign; `Figure` subclasses `plotly.graph_objects.Figure` |
| Output modes | Interactive HTML, notebooks, static images (kaleido) |
| Dependencies | `plotly` + `kaleido` as required deps |
| Styling | Native plotly: templates carry defaults; no `set_props` / `DefaultKwargs` port |
| Theming | Full port of the four custom styles as plotly templates |
| Skipped in plotly (mpl-only) | `.gif` export, `plot_line_collection` (per-segment colored lines) |
| Kept in plotly | Interactive animation (frames + play/slider), all signal-processing plots, 3D + specialty plots |
| Tests | Collectable pytest tests for the plotly module (deviates from `_`-prefix convention deliberately) |

## Package layout

```
liron_utils/graphics/
├── __init__.py         # side-effect free; re-exports common only
├── common/
│   ├── __init__.py
│   ├── colors.py       # COLORS constants + hex2rgb/rgb2hex (from base.py + utils/COLORS.py)
│   ├── screen.py       # get_pixel_color (from utils/etc.py — library-agnostic)
│   ├── files.py        # get_savefig_file_name (from axes.py — used by both backends)
│   ├── fitting.py      # curve-fit prep + confidence-band math (from mpl Axes staticmethods)
│   └── spectra.py      # signal-processing math extracted from mpl plotting.py (see below)
├── mpl/
│   ├── __init__.py     # runs update_rc_params() on import (side effect moves here from graphics/__init__)
│   ├── axes.py         # moved unchanged (minus get_savefig_file_name)
│   ├── plotting.py     # moved; scipy math delegated to common/spectra.py
│   └── utils/          # rc_params.py, default_kwargs.py (mpl-specific remainder)
└── plotly/
    ├── __init__.py     # registers templates; sets pio.templates.default (notebook-aware via is_notebook())
    ├── figure.py       # class Figure(go.Figure)
    └── templates.py    # plotly Template port of the rc themes
```

Notes:
- `import plotly` inside the `graphics/plotly/` subpackage resolves absolutely to the
  installed plotly package — no shadowing with Python 3 absolute imports.
- Consumers updated: `signal_processing/fitting.py` (`from ..graphics import mpl as gr`)
  and `liron_utils/__init__.py`.

### `common/spectra.py` + `common/fitting.py` — extracted computation

Pure-numpy/scipy functions shared by both backends (currently inline in mpl
`plotting.py`). `spectra.py` holds the signal-processing prep; `fitting.py` holds
the curve-fit helpers:

- Spectrum-to-display conversion: amp / power / phase, optional dB with `eps` floor,
  plus the derived axis labels (the `_plot_spectrum` math without the `ax.plot`).
- FFT prep: FFT/fftshift/one-sided slicing/normalization + frequency axis
  (the non-rendering body of `plot_fft`).
- Periodogram prep (scipy.signal.periodogram + shifting/normalization).
- Frequency-response prep (scipy.signal.freqs / freqz).
- Impulse-response prep (scipy.signal.impulse / dimpulse, coefficient padding).
- Curve-fit helpers (`fitting.py`): `_curve_fit_prep_data` (to_numpy + sort) and
  `_curve_fit_confidence_band` (±n_std·σ envelope) — moved from `Axes` staticmethods.
- Spectrogram prep for the plotly backend (scipy.signal.spectrogram → dB matrix,
  freq/time axes, SI-prefix scaling); mpl keeps using `ax.specgram`.

mpl `plotting.py` is refactored to call these; behavior must not change.

## `Figure(go.Figure)` — figure.py

Constructed via `plotly.subplots.make_subplots`; the subplot grid reference
(`_grid_ref`, `_grid_str`) must be copied onto `self`, otherwise `row=`/`col=`
addressing breaks (known go.Figure-subclassing caveat).

```python
Figure(
    shape=(2, 3),
    span_layout=[[(0, 2), (0, 2)], [0, (2, 3)]],  # same span semantics as mpl grid_layout,
                                                   # converted to make_subplots specs rowspan/colspan/None
    shared_xaxes=True, shared_yaxes=False,
    subplot_type="xy",          # or "scene" (3D), "polar", ...; per-cell specs= passthrough wins
    subplot_titles=...,
    **make_subplots_kw,
)
```

All native plotly methods work directly (`add_scatter(row=, col=)`, `update_layout`,
`add_hline`, `show`, ...). Added methods (all accept `row=1, col=1`, forwarding
`'all'` where plotly supports it):

| Method | Implementation sketch |
|---|---|
| `save(file_name=None, **kw)` | Extension dispatch: `.html` → `write_html`; `.png/.pdf/.svg/.jpg/.webp` → `write_image` (kaleido); no extension → `.html`. Default path/name via `common.files.get_savefig_file_name` (same `MAIN_FILE_DIR/figs` + timestamp behavior). |
| `plot(x, y=None, ...)` | Thin `add_scatter` with the y-only convenience (`x` becomes `range(len(x))`). |
| `plot_errorbar` | `add_scatter` with `error_x`/`error_y`; uncertainties arrays unpacked via `to_numpy`. |
| `plot_filled_error` | Two scatters with `fill='tonexty'`; `(y, yerr, n_std)` or `(y_low, y_high)`. |
| `plot_data_and_curve_fit` | errorbar + dense fit line + confidence band, math from `common/fitting.py`. |
| `plot_data_and_lin_reg` | errorbar + regression line labeled with slope±stderr and R². |
| `plot_vlines` / `plot_hlines` | Arrays over native `add_vline`/`add_hline`; legend label attached once. |
| `draw_xy_lines` | Bold `x=0`/`y=0` lines via `add_hline`/`add_vline`. |
| `plot_fft` / `plot_periodogram` / `plot_frequency_response` | Prep from `common/spectra.py`, rendered as scatter lines; default axis titles (Frequency [Hz]/[normalized], Amplitude/Power/Phase, dB). Returns the computed `(spectrum, freqs)` like the mpl version. |
| `plot_impulse_response` | Continuous: line. Discrete: stem = marker trace + one None-separated vertical-segment trace (plotly has no native stem). |
| `plot_specgram` | `common/spectra.py` spectrogram prep → `go.Heatmap`, dB colorbar, SI-prefix frequency axis, default titles. |
| `plot_surf` | `go.Surface`; asserts the target cell is a `scene` subplot; 1D inputs meshgridded, callable `z` supported. |
| `plot_contour` | `go.Contour` with labeled contours (`contours.showlabels`), int levels or explicit array. |
| `plot_animation` | plotly frames + play button and slider; supports the data-driven variant (per-frame image stacks → Heatmap frames, per-frame line data → Scatter frames) and per-frame titles. No `.gif` export. |

Not ported (mpl-only): `plot_line_collection`, `.gif` export, `set_props`,
`DefaultKwargs`/`merge_kwargs`, `_vectorize` broadcasting.

## Templates — templates.py

Port of the four custom styles as `go.layout.Template`, registered in
`plotly.io.templates` under the same names:

- **`liron-utils-default`** — colorway from the current `axes.prop_cycle` colors;
  grid on (light grey); spines equivalent (`showline=True`, left/bottom only —
  `mirror=False`); title/label font sizes; 800×800 px (8×8 in @ 100 dpi);
  per-trace-type defaults replacing `DefaultKwargs` (errorbar marker/color/width,
  surface colorscale Viridis, heatmap colorscale Inferno).
- **`liron-utils-article`** — the seaborn-talk-derived sizes, grid off, all four
  axis lines (`mirror=True`), inward ticks, 1040×715 px.
- **`liron-utils-notebook`** — smaller fonts, 640×480 px.
- **`liron_utils_text_color(color) -> Template`** — parameterized function (title,
  axis-label, and tick fonts), composable via plotly's template stacking
  (`"liron-utils-default+<custom>"` or assignment).

`graphics/plotly/__init__.py` registers all templates and sets
`pio.templates.default = "liron-utils-default"`, switching to the notebook variant
when `is_notebook()` — mirroring the mpl import behavior.

## Dependencies, typing, tests, config

- **pyproject.toml**: add `plotly` and `kaleido` to required dependencies.
- **mypy strict**: plotly ships partial typing; add a `[[tool.mypy.overrides]]`
  `ignore_missing_imports`/relaxation for `plotly.*` only if strict mode fails on it.
- **Typing style**: follow repo conventions — qualified `typing.` imports,
  shape-parameterized numpy aliases (`_Vec[_N]`, `_Mat[_N, _K]`), `typing.cast` at
  numpy/scipy boundaries.
- **Tests**: `tests/graphics/test_plotly.py` — collectable pytest, headless
  (no rendering): constructor grid/spans/shared axes, each plot method's trace
  count/type/data, template registration and defaults, `save` extension dispatch
  (html written to tmp_path; static export smoke-test gated on kaleido availability).
  Import-smoke test for the moved `graphics.mpl`.
- **tox** (`style`, `tests`, `lint`, `typing`) must stay green; pylint config may need
  the module-level naming allowances the mpl module already uses.

## Error handling

Fail fast (repo convention): invalid `span_layout`, non-scene cell for `plot_surf`,
unknown `which` in spectrum plots, and missing `(y, yerr)`/`(y_low, y_high)` raise
immediately (ValueError/AssertionError as appropriate). No defensive fallbacks.
