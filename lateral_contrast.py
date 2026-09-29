"""Lateral amplitude contrast — the spec §4.4 measurement.

The rubric compares the amplitude distribution of the interpreted hydrocarbon leg
(up-dip of a proposed contact) with the down-dip wet leg, using ONLY points inside the
container. Everything here works on raw points; no gridding.

Sign matters: level 1 (Q) is separation in the WRONG direction, so the core statistic is
the signed Mann-Whitney probability sAUC = P(b_hc > b_wet) with b = s_dir * a_fluid,
which runs 0 (wrong way) .. 0.5 (identical) .. 1 (fully separated, expected way).
"""
import numpy as np
from scipy.stats import gaussian_kde, mannwhitneyu, norm

from dhi_io import points_in_polygon

LEVEL_OVERRIDE_NO_WET = 2          # rubric: "no down dip wet leg identified" -> W


def split_legs(z, z_contact, n_min=30, min_frac=0.05):
    z = np.asarray(z, dtype=np.float64)
    hc = z >= z_contact              # shallower (elevation, positive up) = hydrocarbon leg
    wet = ~hc
    flags = []
    n = len(z)
    for name, mask in (("no-hc-leg", hc), ("no-wet-leg", wet)):
        if mask.sum() < n_min or (n > 0 and mask.sum() < min_frac * n):
            flags.append(name)
    return hc, wet, flags


def _robust_scale(a):
    mad = np.median(np.abs(a - np.median(a)))
    return 1.4826 * mad


def _shared_bandwidth(pooled):
    """One absolute bandwidth for BOTH legs (Scott's rule on the pooled sample) -- the
    analogue of the rubric's 'overlay the histograms' with one bin width for both."""
    return float(gaussian_kde(pooled).factor * pooled.std(ddof=1))


def _leg_density(leg, h, grid):
    """KDE of one leg with ABSOLUTE bandwidth h. gaussian_kde's bandwidth is
    factor * std(leg), so factor = h / std(leg). A constant leg (std 0) is a point mass
    smoothed by the same h -- never a curve of zero overlap."""
    leg = np.asarray(leg, dtype=np.float64)
    s = leg.std(ddof=1) if len(leg) > 1 else 0.0
    if s == 0:
        return norm.pdf(grid, loc=leg[0], scale=h)
    return gaussian_kde(leg, bw_method=h / s)(grid)


def separation_stats(b_hc, b_wet):
    b_hc = np.asarray(b_hc, dtype=np.float64); b_wet = np.asarray(b_wet, dtype=np.float64)
    n1, n2 = len(b_hc), len(b_wet)
    # Mann-Whitney U -- ties are handled by midranks; the tie correction only affects the
    # (unused) p-value, not the U statistic -> AUC = U / (n1*n2); all-ties gives exactly 0.5
    u = mannwhitneyu(b_hc, b_wet, alternative="two-sided", method="asymptotic").statistic
    sauc = float(u / (n1 * n2))
    pooled = np.concatenate([b_hc, b_wet])
    pooled_std = pooled.std(ddof=1)
    if pooled_std == 0:               # every value identical: no spread anywhere
        return dict(sAUC=sauc, OVL=1.0, d_mode=0.0, d_med=0.0, n_hc=n1, n_wet=n2)
    scale = _robust_scale(pooled)
    if scale == 0:                    # >half the pooled values tied: MAD collapses, fall back
        scale = pooled_std
    h = _shared_bandwidth(pooled)
    # 4h beyond the data range keeps the KDE tails (and a point-mass leg's spread) on the grid
    grid = np.linspace(pooled.min() - 4 * h, pooled.max() + 4 * h, 512)
    f_hc = _leg_density(b_hc, h, grid)
    f_wet = _leg_density(b_wet, h, grid)
    ovl = float(np.clip(np.trapz(np.minimum(f_hc, f_wet), grid), 0.0, 1.0))
    d_mode = (grid[np.argmax(f_hc)] - grid[np.argmax(f_wet)]) / scale
    d_med = (np.median(b_hc) - np.median(b_wet)) / scale
    return dict(sAUC=sauc, OVL=ovl, d_mode=float(d_mode),
                d_med=float(d_med), n_hc=n1, n_wet=n2)


def block_bootstrap_sauc(x, y, b, hc_mask, wet_mask, block_m=250.0, n_boot=200, seed=0):
    """Spatial block bootstrap: neighbouring points share amplitude, so resampling
    points would understate the uncertainty. Blocks are block_m x block_m squares."""
    x = np.asarray(x); y = np.asarray(y); b = np.asarray(b)
    bx = np.floor((x - x.min()) / block_m).astype(int)
    by = np.floor((y - y.min()) / block_m).astype(int)
    block_id = bx * (by.max() + 1) + by
    blocks = np.unique(block_id[hc_mask | wet_mask])
    members = {k: np.where(block_id == k)[0] for k in blocks}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(blocks, size=len(blocks), replace=True)
        idx = np.concatenate([members[k] for k in pick])
        h = idx[hc_mask[idx]]; w = idx[wet_mask[idx]]
        if len(h) < 2 or len(w) < 2:
            continue                 # degenerate resample: one leg vanished
        vals.append(separation_stats(b[h], b[w])["sAUC"])
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, 0.05)), float(np.quantile(vals, 0.95)))


def contact_profile(z, b, n_steps=25, n_min=30):
    """sAUC as a function of the split elevation. Peaks at the true contact when a
    fluid effect exists; used to flag a proposed contact far from the amplitude break."""
    z = np.asarray(z); b = np.asarray(b)
    zg = np.linspace(np.quantile(z, 0.02), np.quantile(z, 0.98), n_steps)
    out = np.full(n_steps, np.nan)
    for i, zc in enumerate(zg):
        hc = z >= zc
        if hc.sum() >= n_min and (~hc).sum() >= n_min:
            out[i] = separation_stats(b[hc], b[~hc])["sAUC"]
    return zg, out


def measure_prospect(points, geom, z_contact, s_dir=-1, n_min=30, min_frac=0.05,
                     n_boot=200, seed=0):
    inside = points_in_polygon(points["x"], points["y"], geom)
    if inside.sum() == 0:
        raise ValueError("no points inside the container polygon (CRS mismatch?)")
    z = points["z"][inside]
    if (z > 0).any():
        raise ValueError("z must be elevation (positive up, all negative); found positive z")
    b = s_dir * points["fluid"][inside]
    hc, wet, flags = split_legs(z, z_contact, n_min=n_min, min_frac=min_frac)
    out = dict(n_container=int(inside.sum()), n_hc=int(hc.sum()), n_wet=int(wet.sum()),
               flags=flags, level_override=None,
               sAUC=np.nan, OVL=np.nan, d_mode=np.nan, d_med=np.nan,
               sAUC_lo=np.nan, sAUC_hi=np.nan, sAUC_lith=np.nan,
               profile_z=None, profile_sauc=None, sAUC_at_contact_max=np.nan, zeta_gap=np.nan)
    if "no-wet-leg" in flags:
        out["level_override"] = LEVEL_OVERRIDE_NO_WET
    if flags:
        return out
    out.update(separation_stats(b[hc], b[wet]))
    out["sAUC_lo"], out["sAUC_hi"] = block_bootstrap_sauc(
        points["x"][inside], points["y"][inside], b, hc, wet, n_boot=n_boot, seed=seed)
    if "lith" in points:
        out["sAUC_lith"] = separation_stats(s_dir * points["lith"][inside][hc],
                                            s_dir * points["lith"][inside][wet])["sAUC"]
    zg, sg = contact_profile(z, b, n_min=n_min)
    out["profile_z"], out["profile_sauc"] = zg.tolist(), sg.tolist()
    if np.isfinite(sg).any():
        out["sAUC_at_contact_max"] = float(np.nanmax(sg))
        z_best = zg[np.nanargmax(sg)]
        relief = float(z.max() - z.min()) or 1.0
        out["zeta_gap"] = float(abs(z_best - z_contact) / relief)   # in relative-depth units
    return out


def measurement_report(m, ref_sauc):
    """The numbers the VLM reads (spec §5.2.2), with sAUC's percentile among the TRAINING
    fold's rated prospects -- the context that turns a number into a level."""
    ref = np.asarray(ref_sauc, dtype=float)
    ref = ref[np.isfinite(ref)]
    lines = ["Measurements (container points only, split at the proposed contact):"]
    flags = m.get("flags") or []
    if isinstance(flags, str):                 # measurements.csv stores flags ';'-joined
        flags = [f for f in flags.split(";") if f]
    if flags:
        lines.append(f"  flags: {', '.join(flags)}  "
                     f"(n_hc = {m['n_hc']}, n_wet = {m['n_wet']})")
        if "no-wet-leg" in flags:
            lines.append("  No down-dip wet leg identified -> rubric level 2 (W) by definition.")
        return "\n".join(lines)
    pct_txt = (f"{100.0 * (ref < m['sAUC']).mean():.0f}th percentile of rated prospects"
               if len(ref) else "percentile not available (no reference fold)")
    lines += [
        f"  sAUC = {m['sAUC']:.2f}  (P[HC point brighter than wet point]; 0.5 = identical, "
        f"1 = fully separated the expected way, 0 = separated the WRONG way); "
        f"90% block-bootstrap interval [{m['sAUC_lo']:.2f}, {m['sAUC_hi']:.2f}]; "
        f"{pct_txt}",
        f"  overlap coefficient OVL = {m['OVL']:.2f}  (1 = identical curves, 0 = disjoint)",
        f"  peak shift = {m['d_mode']:+.2f} robust SD, median shift = {m['d_med']:+.2f} robust SD "
        f"(positive = expected direction)",
        f"  n_hc = {m['n_hc']}, n_wet = {m['n_wet']}",
        (f"  lithology-volume control sAUC = {m['sAUC_lith']:.2f} (diagnostic only)"
         if np.isfinite(m.get("sAUC_lith", float("nan")))
         else "  lithology-volume control: not available (no lithology volume)"),
        (f"  contact check: proposed contact is {m['zeta_gap']:.2f} of relief from the amplitude break"
         if np.isfinite(m.get("zeta_gap", float("nan")))
         else "  contact check: not available"),
        "  no flags",
    ]
    return "\n".join(lines)
