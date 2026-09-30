"""Phase 0 inventory for the lateral-contrast slice (spec §9 phase 0).

Reads the delivered files under <data_dir>/points/<id>.txt and <data_dir>/polygons/<id>.shp
plus a ratings table, validates each prospect, and writes prospects.csv. `measure` then
caches the §4.4 measurements to measurements.csv so the CV harness and the VLM never
recompute them. `verify-direction` is the per-survey polarity check the spec insists on.
"""
import csv
import os
import sys

import numpy as np

from dhi_gpc import _RATING_MAPS
from dhi_io import load_yaml, null_values_of, points_in_polygon, read_container_polygon, read_petrel_points
from lateral_contrast import measure_prospect, split_legs

ATTR = "lateral_amplitude_contrast"
LETTER_TO_LEVEL = _RATING_MAPS[ATTR]
PROSPECT_COLS = ["prospect_id", "survey", "s_dir", "points_file", "shp_file", "z_contact",
                 "level", "letter", "drilled", "Success", "n_container", "n_hc", "n_wet",
                 "status", "issues"]
SCALAR_KEYS = ["sAUC", "OVL", "d_mode", "d_med", "n_hc", "n_wet", "n_container",
               "sAUC_lo", "sAUC_hi", "sAUC_lith", "sAUC_at_contact_max", "zeta_gap"]


def _survey_dir(surveys, survey):
    s = surveys.get("surveys") or {}
    return int((s.get(survey) or {}).get("s_dir", surveys["defaults"]["s_dir"]))


def build_inventory(data_dir, ratings_csv, column_map, surveys_yaml, out_csv="prospects.csv"):
    surveys = load_yaml(surveys_yaml)
    rows = []
    seen_ids = set()  # every downstream consumer keys by prospect_id -- a silent last-wins
                       # duplicate would corrupt folds and joins, so flag the repeat as error
    with open(ratings_csv, newline="") as f:
        for r in csv.DictReader(f):
            pid = r["prospect_id"].strip()
            rec = dict(prospect_id=pid, survey=r["survey"].strip(),
                       s_dir=_survey_dir(surveys, r["survey"].strip()),
                       points_file=os.path.join(data_dir, "points", f"{pid}.txt"),
                       shp_file=os.path.join(data_dir, "polygons", f"{pid}.shp"),
                       z_contact=r["z_contact"], level="", letter=r[ATTR].strip(),
                       drilled=r.get("drilled", ""), Success=r.get("Success", ""),
                       n_container="", n_hc="", n_wet="", status="ok", issues=[])
            issues = rec["issues"]
            if pid in seen_ids:
                issues.append("duplicate prospect_id")
            seen_ids.add(pid)
            if rec["letter"] not in LETTER_TO_LEVEL:
                issues.append(f"bad letter '{rec['letter']}'")
            else:
                rec["level"] = LETTER_TO_LEVEL[rec["letter"]]
            try:
                rec["z_contact"] = float(rec["z_contact"])
                if not np.isfinite(rec["z_contact"]):   # a literal "nan" parses; it is still missing
                    raise ValueError
                if rec["z_contact"] > 0:
                    issues.append("z_contact must be negative elevation")
            except ValueError:
                issues.append("z_contact missing")
            if not os.path.exists(rec["points_file"]):
                issues.append("points file missing")
            if not os.path.exists(rec["shp_file"]):
                issues.append("shapefile missing")
            if issues:
                rec["status"] = "error"
            else:
                try:
                    nulls = null_values_of(column_map)
                    pts, n_null = read_petrel_points(rec["points_file"], column_map,
                                                     null_values=nulls, return_dropped=True)
                    if n_null:
                        issues.append(f"{n_null} rows dropped as null ({nulls})")
                    # Controller ruling (Task 3 review): a NaN in z would silently land in
                    # the wet leg and a NaN in fluid fails deep inside scipy -- reject
                    # non-finite values right here, before any downstream math sees them.
                    bad_cols = [role for role in ("x", "y", "z", "fluid", "lith")
                                if role in pts and not np.isfinite(pts[role]).all()]
                    if len(pts["x"]) == 0:
                        issues.append("no points left after dropping nulls"); rec["status"] = "error"
                    elif bad_cols:
                        issues.append(f"non-finite values in {', '.join(bad_cols)}")
                        rec["status"] = "error"
                    else:
                        geom = read_container_polygon(rec["shp_file"])
                        inside = points_in_polygon(pts["x"], pts["y"], geom)
                        z = pts["z"][inside]
                        rec["n_container"] = int(inside.sum())
                        if inside.sum() == 0:
                            issues.append("no points inside polygon (CRS?)"); rec["status"] = "error"
                        else:
                            if (z > 0).any():
                                issues.append("positive z inside container"); rec["status"] = "error"
                            hc, wet, flags = split_legs(z, rec["z_contact"])
                            rec["n_hc"], rec["n_wet"] = int(hc.sum()), int(wet.sum())
                            if flags:
                                issues.extend(flags)
                                if rec["status"] == "ok":
                                    rec["status"] = "warn"
                            if not (z.min() <= rec["z_contact"] <= z.max()):
                                issues.append("z_contact outside container elevation range")
                                if rec["status"] == "ok":
                                    rec["status"] = "warn"
                except Exception as e:                       # noqa: BLE001 - report, don't abort the sweep
                    issues.append(f"read error: {e}"); rec["status"] = "error"
            rows.append(rec)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PROSPECT_COLS); w.writeheader()
        for rec in rows:
            w.writerow({**rec, "issues": "; ".join(rec["issues"])})
    n = {s: sum(r["status"] == s for r in rows) for s in ("ok", "warn", "error")}
    print(f"inventory: {len(rows)} prospects  ok={n['ok']} warn={n['warn']} error={n['error']} -> {out_csv}")
    for r in rows:
        if r["issues"]:
            print(f"  {r['status']:5s} {r['prospect_id']}: {'; '.join(r['issues'])}")
    return rows


def load_prospects(path="prospects.csv"):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["level"] = int(r["level"]) if r["level"] else None
        r["s_dir"] = int(r["s_dir"])
        # An error-status row can carry a blank or unparseable z_contact ("", "nan", "unknown",
        # "TBD" ... -- that's WHY it's error). A bad single row must not crash the load of the
        # whole file, so anything float() rejects becomes NaN here rather than a ValueError.
        try:
            r["z_contact"] = float(r["z_contact"])
        except (TypeError, ValueError):
            r["z_contact"] = float("nan")
        r["issues"] = [s for s in r["issues"].split("; ") if s]
    return rows


def measure_all(prospects, column_map, out_csv="measurements.csv", n_boot=200):
    out = []
    for r in prospects:
        if r["status"] == "error":
            continue
        pts = read_petrel_points(r["points_file"], column_map, null_values=null_values_of(column_map))
        geom = read_container_polygon(r["shp_file"])
        m = measure_prospect(pts, geom, r["z_contact"], s_dir=r["s_dir"], n_boot=n_boot)
        row = {"prospect_id": r["prospect_id"], **{k: m[k] for k in SCALAR_KEYS},
               "flags": ";".join(m["flags"]), "level_override": m["level_override"] or ""}
        out.append(row)
    if not out:
        raise ValueError("no measurable prospects (every row is status=error)")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print(f"measured {len(out)} prospects -> {out_csv}")
    return out


def load_measurements(path="measurements.csv"):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in SCALAR_KEYS:
            r[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
        for k in ("n_hc", "n_wet", "n_container"):
            # counts round-trip through CSV as floats; the VLM report must read "n_hc = 312"
            if np.isfinite(r[k]):
                r[k] = int(r[k])
        r["level_override"] = int(r["level_override"]) if r["level_override"] else None
    return rows


def verify_direction(measurements, prospects, min_level=4):
    lvl = {p["prospect_id"]: (p["level"], p["survey"]) for p in prospects}
    acc = {}
    for m in measurements:
        level, survey = lvl[m["prospect_id"]]
        if level is not None and level >= min_level and np.isfinite(m["sAUC"]):
            acc.setdefault(survey, []).append(m["sAUC"])
    out = {s: float(np.mean(v)) for s, v in acc.items()}
    for s, v in sorted(out.items()):
        flag = "  <-- POLARITY? high-rated prospects separate the WRONG way; flip s_dir for this survey" if v < 0.5 else ""
        print(f"  {s}: mean sAUC of level>={min_level} prospects = {v:.2f} (n={len(acc[s])}){flag}")
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    colmap = load_yaml("column_map.yaml") if os.path.exists("column_map.yaml") else \
        {"x": "X", "y": "Y", "z": "Z", "fluid": "Fluid_MinAmp", "lith": "Lith_MinAmp"}
    if cmd == "build":
        build_inventory(sys.argv[2], sys.argv[3], colmap, "surveys.yaml")
    elif cmd == "measure":
        measure_all(load_prospects(), colmap)
    elif cmd == "verify-direction":
        verify_direction(load_measurements(), load_prospects())
