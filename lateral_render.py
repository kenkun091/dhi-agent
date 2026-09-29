"""Rubric-style panel for lateral amplitude contrast (spec §4.4, §5.2.1 P0).

Mirrors rubric/lateral_amplitude_contrast_levels.png: two smooth curves, blue = WET leg,
green = HC leg, with the HC peak expected to the LEFT exactly as the rubric draws it. The
second row of the rubric shows the same curves rotated; `orientation="vertical"` reproduces
that and doubles as the self-consistency rendering for the VLM. Every visual choice here is
versioned: change anything, bump RENDER_VERSION, re-run the regression set.

The x axis is NOT raw amplitude (RENDER_VERSION "2"):
- Polarity-normalised: each leg is plotted as `-s_dir * a_fluid`. Brightness is
  `b = s_dir * a_fluid` (larger b = more hydrocarbon-like), so `-b` puts hydrocarbons LEFT
  for every survey. With the default s_dir = -1 this is the raw amplitude; on a
  reverse-polarity survey (s_dir = +1) plotting raw amplitude would draw the HC peak RIGHT
  and the VLM would read "wrong direction" (level 1) while the report's sAUC says the
  opposite.
- Standardised: `u = (a_norm - median(wet)) / robust_scale(pooled)` (MAD-based, the same
  scale separation_stats uses for d_mode/d_med). Raw tick labels would expose each survey's
  amplitude scale (a survey-identity shortcut) and put survey-specific numbers into the
  number-free L0 image; standardised ticks are the same for every prospect up to shape.

By default the panel carries NO sAUC/overlap/peak-shift numbers (`annotate=False`) — those
reach the VLM rater through the L1 measurement report text, never the image, so an L0
(image-only) prompt is not silently contaminated with L1 numbers. `annotate=True` draws the
stats line for a human-facing overlay. The legend's `n=` counts are unaffected either way.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lateral_contrast import _leg_density, _robust_scale, _shared_bandwidth

RENDER_VERSION = "2"
WET_COLOUR, HC_COLOUR = "#1f5fbf", "#2e7d32"          # blue wet, green HC (rubric)


def _curves(a_hc, a_wet):
    """Same smoothing as separation_stats: one absolute bandwidth h for both legs, so the
    drawn curves are the ones OVL and d_mode were computed from (a constant leg draws as a
    point mass of width h instead of crashing gaussian_kde)."""
    pooled = np.concatenate([a_hc, a_wet])
    if pooled.std(ddof=1) == 0:                      # every value identical: one spike, no KDE possible
        h = 1.0
    else:
        h = _shared_bandwidth(pooled)
    grid = np.linspace(pooled.min() - 4 * h, pooled.max() + 4 * h, 400)
    return grid, _leg_density(a_hc, h, grid), _leg_density(a_wet, h, grid)


# Two lines so the label fits the 5-inch vertical axis. In the rotated (vertical) panel the
# amplitude axis is y, so "left" becomes "low" -- same data, same direction.
AXIS_LABEL = ("standardised amplitude (wet-leg median = 0, robust-SD units;\n"
              "polarity-normalised so hydrocarbons sit {side})")


def _standardise(a_hc, a_wet, s_dir):
    """Polarity-normalise (-s_dir*a: HC left for every survey) then centre on the wet-leg
    median and divide by the pooled robust SD, so the axis carries no survey amplitude
    scale. MAD collapses when >half the values tie -> sample SD; all-identical -> 1.0."""
    n_hc = -s_dir * a_hc; n_wet = -s_dir * a_wet
    pooled = np.concatenate([n_hc, n_wet])
    scale = _robust_scale(pooled)
    if scale == 0:
        scale = pooled.std(ddof=1) if pooled.size > 1 else 0.0
    if not scale > 0:
        scale = 1.0
    centre = np.median(n_wet)
    return (n_hc - centre) / scale, (n_wet - centre) / scale


def render_panel(a_hc, a_wet, stats, out_path, orientation="horizontal", title=None, annotate=False,
                 s_dir=-1):
    a_hc = np.asarray(a_hc, float); a_wet = np.asarray(a_wet, float)
    a_hc, a_wet = _standardise(a_hc, a_wet, s_dir)
    grid, f_hc, f_wet = _curves(a_hc, a_wet)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    if orientation == "horizontal":
        ax.plot(grid, f_wet, color=WET_COLOUR, lw=3, label=f"wet leg (n={stats['n_wet']})")
        ax.plot(grid, f_hc, color=HC_COLOUR, lw=3, label=f"HC leg (n={stats['n_hc']})")
        ax.set_xlabel(AXIS_LABEL.format(side="left")); ax.set_ylabel("density")
    else:
        ax.plot(f_wet, grid, color=WET_COLOUR, lw=3, label=f"wet leg (n={stats['n_wet']})")
        ax.plot(f_hc, grid, color=HC_COLOUR, lw=3, label=f"HC leg (n={stats['n_hc']})")
        ax.set_ylabel(AXIS_LABEL.format(side="low")); ax.set_xlabel("density")
    if orientation == "horizontal":
        ax.set_yticks([])
    else:
        ax.set_xticks([])
    ax.legend(loc="upper right", frameon=False)
    ax.set_title(title or "Lateral amplitude contrast: HC leg vs wet leg", loc="left")
    if annotate:
        txt = (f"sAUC = {stats['sAUC']:.2f}   overlap = {stats['OVL']:.2f}   "
               f"peak shift = {stats['d_mode']:+.2f} robust SD")
        ax.text(0.01, -0.14, txt, transform=ax.transAxes, fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
