"""Survey-grouped repeated CV for the lateral-contrast raters (spec §7.1, §7.4).

Groups = survey, k = min(10, n_surveys). Everything learned lives inside the training
fold: the ordinal fit, the exemplar pool for the VLM, the percentile reference for the
measurement report, and the stacking model. OOF levels/probs are averaged over repeats
and written as JSON next to the metrics.
"""
import json
import os
import random
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedGroupKFold

from dhi_io import load_yaml, null_values_of, read_container_polygon, read_petrel_points, points_in_polygon
from inventory import load_measurements, load_prospects
from lateral_contrast import measurement_report, split_legs
from lateral_render import render_panel
from ordinal_metrics import summary
from ordinal_model import OrdinalModel
from vlm_rater import VLMRater, audit_rationale, select_exemplars, self_consistency

FEATURE_SETS = {"sauc": ["sAUC"], "sauc+ovl+dmode": ["sAUC", "OVL", "d_mode"]}


def make_folds(levels, groups, k=None, n_repeats=3, seed=0):
    levels = np.asarray(levels); groups = np.asarray(groups)
    n_groups = len(set(groups))
    if n_groups < 2:
        raise ValueError("grouped CV needs at least 2 surveys")
    k = k or min(10, n_groups)
    if k == n_groups:
        # k == n_groups means leave-one-survey-out: every group is its own fold, a
        # deterministic partition of the groups regardless of shuffle -- repeating it
        # would just re-run the identical folds, multiplying VLM calls for nothing.
        n_repeats = 1
    out = []
    for r in range(n_repeats):
        skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed + r)
        for f, (tr, te) in enumerate(skf.split(np.zeros(len(levels)), levels, groups)):
            out.append((r, f, tr, te))
    return out


def align(rows, meas):
    m = {x["prospect_id"]: x for x in meas}
    keep = [r for r in rows if r["status"] != "error" and r["prospect_id"] in m and r["level"]]
    return keep, [m[r["prospect_id"]] for r in keep]


def features(meas_rows, which="sauc"):
    cols = FEATURE_SETS[which]
    return np.array([[x[c] for c in cols] for x in meas_rows], dtype=np.float64)


def spearman_check(rows, meas):
    rows, meas = align(rows, meas)
    ok = [i for i, x in enumerate(meas) if np.isfinite(x["sAUC"])]
    rho = spearmanr([meas[i]["sAUC"] for i in ok], [rows[i]["level"] for i in ok]).correlation
    print(f"Spearman(sAUC, level) = {rho:.3f} on n={len(ok)} (acceptance 1: high => contact/sign OK)")
    return float(rho)


def _accumulate(n, K=5):
    return np.zeros((n, K)), np.zeros(n)


def _finalize(probs_acc, cnt, override, what):
    """Shared scoring tail so all three raters score the SAME population: average over
    repeats, then apply the rubric override (no-wet-leg -> level 2) AFTER averaging so it
    cannot be diluted, then score anything still unscored as uniform and say so."""
    probs = probs_acc / np.maximum(cnt, 1)[:, None]
    for i in np.where(override > 0)[0]:
        probs[i] = 0.0; probs[i, override[i] - 1] = 1.0
    missing = (cnt == 0) & (override == 0)
    if missing.any():
        print(f"WARNING [{what}]: {int(missing.sum())} prospects had no usable features/panel and no override; scored uniform")
        probs[missing] = 0.2
    level = (np.cumsum(probs, 1) >= 0.5).argmax(1) + 1        # median level: ordinal-appropriate
    return probs, level


def cv_rater_a(rows, meas, which, folds, **model_kw):
    rows, meas = align(rows, meas)
    y = np.array([r["level"] for r in rows]); g = [r["survey"] for r in rows]
    X = features(meas, which)
    override = np.array([x["level_override"] or 0 for x in meas])
    usable = np.isfinite(X).all(1)
    probs_acc, cnt = _accumulate(len(rows))
    for rep, f, tr, te in folds:
        tr = tr[usable[tr]]
        m = OrdinalModel(**model_kw).fit(X[tr], y[tr], [g[i] for i in tr])
        te_u = te[usable[te]]
        if len(te_u):
            probs_acc[te_u] += m.predict_proba(X[te_u], [g[i] for i in te_u]); cnt[te_u] += 1
    probs, level = _finalize(probs_acc, cnt, override, "rater A")
    return dict(oof_level=level.tolist(), oof_probs=probs.tolist(), metrics=summary(y, level, probs),
                prospect_id=[r["prospect_id"] for r in rows])


def render_all(rows, meas, column_map, panels_dir):
    os.makedirs(panels_dir, exist_ok=True)
    m = {x["prospect_id"]: x for x in meas}
    out = {}
    skipped = {}                       # reason -> count, reported once so no skip is silent
    for r in rows:
        if r["status"] == "error":
            skipped["status-error"] = skipped.get("status-error", 0) + 1
            continue
        if r["prospect_id"] not in m:
            skipped["no-measurement"] = skipped.get("no-measurement", 0) + 1
            continue
        pts = read_petrel_points(r["points_file"], column_map, null_values=null_values_of(column_map))
        inside = points_in_polygon(pts["x"], pts["y"], read_container_polygon(r["shp_file"]))
        z, a = pts["z"][inside], pts["fluid"][inside]
        hc, wet, flags = split_legs(z, r["z_contact"])
        if flags:
            for fl in flags:
                skipped[fl] = skipped.get(fl, 0) + 1
            skipped["_prospects"] = skipped.get("_prospects", 0) + 1
            continue
        st = m[r["prospect_id"]]
        # s_dir polarity-normalises the panel so the HC peak sits left on EVERY survey
        # (the orientation the prompt and rubric.yaml promise the VLM)
        h = render_panel(a[hc], a[wet], st, os.path.join(panels_dir, f"{r['prospect_id']}_h.png"),
                         s_dir=r["s_dir"])
        v = render_panel(a[hc], a[wet], st, os.path.join(panels_dir, f"{r['prospect_id']}_v.png"),
                         orientation="vertical", s_dir=r["s_dir"])
        out[r["prospect_id"]] = (h, v)
    n_skip = (skipped.pop("_prospects", 0) + skipped.get("status-error", 0)
              + skipped.get("no-measurement", 0))
    if n_skip:
        why = ", ".join(f"{k}: {v}" for k, v in sorted(skipped.items()))
        print(f"render_all: skipped {n_skip} flagged prospects ({why})")
    return out


def cv_rater_c(rows, meas, folds, rater, rubric, panels, rung="L1", k_ex=4, n_spread=0):
    rows, meas = align(rows, meas)
    y = np.array([r["level"] for r in rows])
    override = np.array([x["level_override"] or 0 for x in meas])
    probs_acc, cnt = _accumulate(len(rows))
    audits, agreements, abstained = {}, {}, {}
    attr = rubric["attributes"]["lateral_amplitude_contrast"]
    for rep, f, tr, te in folds:
        pool = [dict(prospect_id=rows[i]["prospect_id"], sAUC=meas[i]["sAUC"], level=int(y[i]),
                     letter=rows[i].get("letter", ""), survey=rows[i]["survey"],
                     panel_path=panels[rows[i]["prospect_id"]][0],
                     # percentile reference built from the TRAINING fold's sAUC values only,
                     # so a test-fold prospect's own rank among rated wells never leaks in
                     report_text=measurement_report(meas[i], [meas[j]["sAUC"] for j in tr]))
                for i in tr if rows[i]["prospect_id"] in panels and np.isfinite(meas[i]["sAUC"])]
        ref = np.array([meas[i]["sAUC"] for i in tr])
        test_ids = {rows[i]["prospect_id"] for i in te}
        for i in te:
            pid = rows[i]["prospect_id"]
            if override[i] > 0:
                # rubric override (e.g. no-wet-leg -> level 2) is applied post-hoc in
                # _finalize; skip the VLM call entirely, even if a stale panel file exists,
                # so an overridden prospect never costs a request or enters the pool/audit
                continue
            if pid not in panels:
                continue
            report = measurement_report(meas[i], ref) if rung == "L1" else None
            ex = select_exemplars(meas[i]["sAUC"], pool, k=k_ex, prefer_other_survey=rows[i]["survey"]) \
                if rung == "L1" else []
            # exemplars are drawn from `pool` (training-fold prospects only); this assertion
            # is the leakage backstop -- if a test-fold id ever appears here, fail loud
            # rather than silently score against evidence the model was tested on
            assert not ({e["prospect_id"] for e in ex} & test_ids), "exemplar leakage from the test fold"
            tags = [f"r{rep}"] + [f"r{rep}s{s}" for s in range(n_spread)]
            results = [rater.rate(pid, [panels[pid][0]], report, ex, rubric, temperature=None, tag=tags[0])]
            for s in range(n_spread):
                ex_s = list(ex); random.Random(s).shuffle(ex_s)       # reshuffled exemplar order per sample
                results.append(rater.rate(pid, [panels[pid][(s + 1) % 2]], report, ex_s, rubric,
                                          temperature=None, tag=tags[s + 1]))
            sc = self_consistency(results)
            if sc["n_abstain"]:
                abstained[pid] = abstained.get(pid, 0) + sc["n_abstain"]
            if sc["mean_probs"] is not None:
                # every sample abstained -> no evidence: leave the prospect unscored so that
                # _finalize scores it uniform and says so, instead of averaging in a guess
                probs_acc[i] += np.array(sc["mean_probs"]); cnt[i] += 1
            agreements[pid] = sc["agreement"]
            if attr.get("levels"):
                # spec §7.4: the audit runs on 100 % of ratings -- every response that fed
                # mean_probs, not only the canonical one; each issue names its sample tag
                audits[pid] = [f"{t}: {msg}" for t, res in zip(tags, results)
                               for msg in audit_rationale(res, report or "", attr)]
    if abstained:
        print(f"rater C: {sum(abstained.values())} abstaining responses on {len(abstained)} prospects "
              f"(dropped from the average, counted against agreement)")
    probs, level = _finalize(probs_acc, cnt, override, "rater C")
    n_bad = sum(1 for v in audits.values() if v)
    return dict(oof_level=level.tolist(), oof_probs=probs.tolist(), metrics=summary(y, level, probs),
                prospect_id=[r["prospect_id"] for r in rows], audit_failures=n_bad, audits=audits,
                agreement=agreements, abstentions=abstained)


def stack(rows, meas, folds, vlm_probs, **model_kw):
    """Calibrator: OrdinalModel on [sAUC, E_vlm[level]] fit in-fold (spec §5.3)."""
    rows, meas = align(rows, meas)
    y = np.array([r["level"] for r in rows]); g = [r["survey"] for r in rows]
    override = np.array([x["level_override"] or 0 for x in meas])
    vp = np.asarray(vlm_probs, dtype=np.float64)
    e_vlm = (vp * np.arange(1, 6)).sum(1)
    # exactly-uniform Rater C probs = "no panel, no override" (the _finalize fallback): they
    # carry no VLM evidence, so mark them unusable rather than feed the stack E_vlm = 3
    e_vlm[np.all(vp == 0.2, axis=1)] = np.nan
    X = np.column_stack([features(meas, "sauc")[:, 0], e_vlm])
    usable = np.isfinite(X).all(1)
    probs_acc, cnt = _accumulate(len(rows))
    for rep, f, tr, te in folds:
        tr = tr[usable[tr]]; te_u = te[usable[te]]
        if not len(te_u):
            continue     # nothing scorable in this fold (e.g. a held-out survey that is all overrides)
        m = OrdinalModel(**model_kw).fit(X[tr], y[tr], [g[i] for i in tr])
        probs_acc[te_u] += m.predict_proba(X[te_u], [g[i] for i in te_u]); cnt[te_u] += 1
    probs, level = _finalize(probs_acc, cnt, override, "stack")
    return dict(oof_level=level.tolist(), oof_probs=probs.tolist(), metrics=summary(y, level, probs),
                prospect_id=[r["prospect_id"] for r in rows])


def _save(name, out):
    path = f"rate_oof_lateral_{name}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["metrics"], indent=1)); print("->", path)


if __name__ == "__main__":
    cmd = sys.argv[1]
    rows, meas = align(load_prospects(), load_measurements())   # one shared ordering for all raters
    colmap = load_yaml("column_map.yaml")
    folds = make_folds([r["level"] for r in rows], [r["survey"] for r in rows], n_repeats=3)
    if cmd == "spearman":
        spearman_check(rows, meas)
    elif cmd == "panels":
        render_all(rows, meas, colmap, "panels"); print("panels/ written")
    elif cmd == "cv-a":
        which = sys.argv[2] if len(sys.argv) > 2 else "sauc"
        _save(f"A_{which}", cv_rater_a(rows, meas, which, folds))
    elif cmd == "cv-c":
        rung = sys.argv[2] if len(sys.argv) > 2 else "L1"
        panels = {p: (f"panels/{p}_h.png", f"panels/{p}_v.png") for p in
                  [r["prospect_id"] for r in rows] if os.path.exists(f"panels/{p}_h.png")}
        out = cv_rater_c(rows, meas, folds, VLMRater(), load_yaml("rubric.yaml"), panels, rung=rung,
                         n_spread=6 if rung == "L1" else 0)
        print(f"rationale audit failures: {out['audit_failures']}")
        _save(f"C_{rung}", out)
    elif cmd == "stack":
        with open("rate_oof_lateral_C_L1.json") as f:
            c = json.load(f)
        # oof_probs is consumed by POSITION below; guard against a same-count reorder
        # silently attaching one prospect's VLM probabilities to a different prospect
        assert c["prospect_id"] == [r["prospect_id"] for r in rows], \
            "rate_oof_lateral_C_L1.json was produced from a different prospect ordering; re-run cv-c"
        _save("stack_A+C", stack(rows, meas, folds, c["oof_probs"]))
