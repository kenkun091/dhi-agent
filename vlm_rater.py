"""Rater C: vision-language rating of lateral amplitude contrast (spec §5.2, L0/L1).

L0 = rubric text + visual rubric + the prospect's panel. L1 = + measurement report +
retrieved in-fold exemplars. Raw output is NEVER the shipped level (spec §5.3): it is
evaluated raw and stacked through OrdinalModel in rate_cv.py.

Every result carries model / prompt / render / rubric versions. Responses are cached on
disk by a hash of the full request so CV re-runs are free and reproducible. The client
is injectable (tests use a fake; no network in tests).
"""
import base64
import hashlib
import json
import math
import os
import re

import numpy as np

from lateral_render import RENDER_VERSION

PROMPT_VERSION = "1"
DEFAULT_MODEL = "claude-fable-5-1"
ATTR = "lateral_amplitude_contrast"


class VLMParseError(ValueError):
    pass


def _img_block(path, max_side=1568):
    """PNG -> base64 block. Images wider than the API's 1568 px long edge are downscaled
    here (the rubric screenshot is ~2000 px) so we do not ship 2 MB per call to be resized
    server-side anyway."""
    from io import BytesIO
    from PIL import Image
    im = Image.open(path)
    if max(im.size) > max_side:
        r = max_side / max(im.size)
        im = im.convert("RGB").resize((int(im.size[0] * r), int(im.size[1] * r)))
        buf = BytesIO(); im.save(buf, format="PNG"); raw = buf.getvalue()
    else:
        with open(path, "rb") as f:
            raw = f.read()
    data = base64.standard_b64encode(raw).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def _system_text(attr):
    lines = ["You are a seismic DHI interpreter rating ONE attribute, lateral amplitude contrast, "
             "against the rubric below. Rate only what the histogram panel and the numbers show.",
             "", "RUBRIC GUIDELINE: " + attr["guideline"].strip(), "",
             "COLOURS: " + attr["notes"]["colours"], "DIRECTION: " + attr["notes"]["direction"].strip(),
             "SPLIT: " + attr["notes"]["split"], "", "LEVELS (cite the cue id you rely on):"]
    for k in sorted(attr["levels"]):
        lv = attr["levels"][k]
        lines.append(f"  level {k} ({lv['letter']}, cue {lv['cue']}): {lv['text']}")
    lines += ["", "RULES: use the full 1-5 scale; when the evidence is between two levels, spread "
              "probability; cite cue ids and the measurement values you used; never guess from "
              "prospect identity; if the panel is unreadable, set flags: ['abstain'].",
              "", "OUTPUT: reply with ONLY a JSON object: "
              '{"level": int 1-5, "probs": [5 floats summing to 1], "rationale": "<= 80 words", '
              '"cues_cited": ["LAC-..."], "flags": []}']
    return "\n".join(lines)


def build_messages(attr, panel_paths, report_text, exemplars, rubric_image_path):
    system = _system_text(attr)
    content = [{"type": "text", "text": "Reference illustration of the five levels (blue = wet leg, green = HC leg):"},
               _img_block(rubric_image_path)]
    for e in exemplars:
        content.append({"type": "text", "text": f"EXEMPLAR rated level {e['level']} ({e['letter']}) by a geoscientist:"
                        + (f"\n{e['report_text']}" if e.get("report_text") else "")})
        content.append(_img_block(e["panel_path"]))
    content.append({"type": "text", "text": "PROSPECT TO RATE:" + (f"\n{report_text}" if report_text else "")})
    for p in panel_paths:
        content.append(_img_block(p))
    content.append({"type": "text", "text": "Return the JSON object now."})
    return system, [{"role": "user", "content": content}]


def parse_response(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise VLMParseError("no JSON object in response")
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise VLMParseError(f"bad JSON: {e}") from None
    # Strict typing on purpose (spec: reject, never coerce): json.loads accepts bare
    # NaN/Infinity tokens and Python's int()/float()/list() happily coerce floats,
    # numeral strings, bools and even individual characters of a string into "valid"
    # shapes, which would let garbage reach calibration silently.
    level = d.get("level"); probs = d.get("probs")
    if not isinstance(level, int) or isinstance(level, bool) or not 1 <= level <= 5:
        raise VLMParseError(f"level must be an int in 1..5, got {level!r}")
    if (not isinstance(probs, list) or len(probs) != 5
            or any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or p < 0 for p in probs)
            or abs(sum(probs) - 1) > 0.02):
        raise VLMParseError(f"probs must be 5 finite non-negative numbers summing to 1, got {probs!r}")
    cues = d.get("cues_cited", []); flags = d.get("flags", [])
    if not isinstance(cues, list) or not isinstance(flags, list):
        raise VLMParseError("cues_cited and flags must be lists")
    return dict(level=level, probs=[float(p) / sum(probs) for p in probs],   # renormalise only within the sanctioned ±0.02
                rationale=str(d.get("rationale", "")), cues_cited=[str(c) for c in cues], flags=[str(f) for f in flags])


class VLMRater:
    def __init__(self, client=None, model=DEFAULT_MODEL, cache_dir=".vlm_cache", max_retries=2):
        if client is None:
            import anthropic                      # key from ANTHROPIC_API_KEY (never in the repo)
            client = anthropic.Anthropic()
        self.client = client; self.model = model; self.cache_dir = cache_dir
        self.max_retries = max_retries
        os.makedirs(cache_dir, exist_ok=True)

    def _key(self, system, messages, temperature, tag):
        h = hashlib.sha256(json.dumps([self.model, PROMPT_VERSION, RENDER_VERSION, system, messages,
                                       temperature, tag]).encode()).hexdigest()
        return os.path.join(self.cache_dir, h + ".json")

    def rate(self, prospect_id, panel_paths, report_text, exemplars, rubric, temperature=None, tag=""):
        attr = rubric["attributes"][ATTR]
        system, messages = build_messages(attr, panel_paths, report_text, exemplars, attr["illustration"])
        path = self._key(system, messages, temperature, tag)
        meta = dict(prospect_id=prospect_id, model=self.model, prompt_version=PROMPT_VERSION,
                    render_version=RENDER_VERSION, rubric_version=rubric["version"],
                    exemplar_ids=[e.get("prospect_id") for e in exemplars], temperature=temperature)
        if os.path.exists(path):
            with open(path) as f:
                return {**json.load(f), **meta, "cached": True}
        last = None
        for _ in range(self.max_retries + 1):
            # The Claude 5 family rejects `temperature` outright (HTTP 400), so it is sent only when
            # a caller asks for it explicitly on a model that supports it. Spread samples get their
            # variation from default sampling + rendering/exemplar changes, kept apart by `tag`.
            req = dict(model=self.model, max_tokens=600, system=system, messages=messages)
            if temperature is not None:
                req["temperature"] = temperature
            resp = self.client.messages.create(**req)
            text = "".join(c.text for c in resp.content if getattr(c, "type", "") == "text")
            try:
                parsed = parse_response(text)
            except VLMParseError as e:
                last = e; continue
            result = {**parsed, "raw": text}
            with open(path, "w") as f:
                json.dump(result, f)
            return {**result, **meta, "cached": False}
        raise VLMParseError(f"{prospect_id}: no parseable reply after {self.max_retries + 1} attempts: {last}")


def select_exemplars(query_sauc, pool, k=4, prefer_other_survey=None):
    """Nearest-but-level-diverse (spec §5.2.2): walk candidates by |sAUC| distance and take
    the nearest example of each level not yet shown (so the model sees the scale, not four
    copies of the nearest level), then fill the remaining slots by distance. Ties prefer a
    survey other than the query's."""
    cands = [c for c in pool if np.isfinite(c["sAUC"])]
    cands.sort(key=lambda c: (abs(c["sAUC"] - query_sauc), c["survey"] == prefer_other_survey))
    chosen, used = [], set()
    for c in cands:
        if len(chosen) >= k:
            break
        if c["level"] not in used:
            chosen.append(c); used.add(c["level"])
    for c in cands:
        if len(chosen) >= k:
            break
        if c not in chosen:
            chosen.append(c)
    return chosen


def self_consistency(results):
    levels = np.array([r["level"] for r in results])
    mode = int(np.bincount(levels).argmax())
    return dict(level_mode=mode, agreement=float(np.mean(levels == mode)),
                mean_probs=np.mean([r["probs"] for r in results], axis=0).tolist())


def audit_rationale(result, report_text, attr):
    """Automatic hallucination check (spec §7.4): every number the rationale cites must appear in
    the report (2-decimal match), every cue id must exist in the rubric."""
    # Compare SIGNED: direction is the level-1 question, so "peak shift -1.20" against a
    # report "+1.20" is a real error. The lookbehind stops a hyphen that follows a digit or
    # dot (a range like "0.88-0.95") being read as a minus sign; "+1.20" is captured
    # unsigned (1.2), which is the same value.
    num = re.compile(r"(?<![\d.])-?\d+\.\d+")
    issues = []
    report_nums = {round(float(x), 2) for x in num.findall(report_text or "")}
    for x in num.findall(result.get("rationale", "")):
        if round(float(x), 2) not in report_nums:
            issues.append(f"cited number {x} not in report")
    valid = {lv["cue"] for lv in attr["levels"].values()}
    for c in result.get("cues_cited", []):
        if c not in valid:
            issues.append(f"unknown cue id {c}")
    return issues
