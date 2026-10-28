"""Vision branch: per-frame pose features, windows and the transparent rule.

The 2D hip centre is a posture proxy, not a physical centre of mass. All
distances are normalised by an upright torso-length reference so that the rule
does not depend on raw pixel thresholds. Lying down is not a fall by itself:
the rule needs a fast descent, persistence of a low posture and no recovery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from homeease.core.config import VisionRuleCfg
from homeease.ml.fall.accel import ScoreResult
from homeease.ml.pose.layout import L_ANK, L_HIP, L_KNEE, L_SH, R_ANK, R_HIP, R_KNEE, R_SH, CORE, PoseObs
from homeease.schemas.evidence import ScoreSemantics

FCOLS = [
    "t", "valid", "hip_x", "hip_y", "sh_x", "sh_y", "tl", "torso_deg", "aspect", "knee_y", "ankle_y",
    "conf", "visible", "target_conf", "ambiguous", "persons", "knee_angle", "bbox_h", "n_core",
]
C = {name: i for i, name in enumerate(FCOLS)}


def _mean_pts(kp: np.ndarray, idx: list[int], min_score: float, w: int, h: int) -> tuple[float, float] | None:
    pts = [kp[i] for i in idx if not np.isnan(kp[i, 0]) and kp[i, 2] >= min_score]
    if not pts:
        return None
    a = np.array(pts)
    return float(a[:, 0].mean() * w), float(a[:, 1].mean() * h)


def _angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))))


def frame_features(obs: PoseObs, min_score: float = 0.3) -> np.ndarray:
    row = np.full(len(FCOLS), np.nan)
    row[C["t"]] = obs.t_ms
    row[C["visible"]] = 1.0 if obs.target_visible else 0.0
    row[C["target_conf"]] = obs.target_conf
    row[C["ambiguous"]] = 1.0 if ("target_ambiguous" in obs.flags or "target_switched" in obs.flags) else 0.0
    row[C["persons"]] = obs.person_count
    row[C["valid"]] = 0.0
    if not obs.target_visible:
        return row
    kp, w, h = obs.kp, obs.width, obs.height
    hip = _mean_pts(kp, [L_HIP, R_HIP], min_score, w, h)
    sh = _mean_pts(kp, [L_SH, R_SH], min_score, w, h)
    core_ok = [i for i in CORE if not np.isnan(kp[i, 0]) and kp[i, 2] >= min_score]
    row[C["n_core"]] = len(core_ok)
    if core_ok:
        row[C["conf"]] = float(np.mean([kp[i, 2] for i in core_ok]))
    allv = (~np.isnan(kp[:, 0])) & (kp[:, 2] >= min_score)
    if allv.sum() >= 3:
        xs, ys = kp[allv, 0] * w, kp[allv, 1] * h
        bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
        row[C["aspect"]] = bw / max(bh, 1.0)
        row[C["bbox_h"]] = bh
    if hip is None or sh is None:
        return row
    row[C["valid"]] = 1.0
    row[C["hip_x"]], row[C["hip_y"]] = hip
    row[C["sh_x"]], row[C["sh_y"]] = sh
    dx, dy = sh[0] - hip[0], sh[1] - hip[1]
    row[C["tl"]] = float(np.hypot(dx, dy))
    # 0 deg = shoulders straight above hips, 90 = horizontal, 180 = inverted
    row[C["torso_deg"]] = float(np.degrees(np.arctan2(abs(dx), -dy)))
    knee = _mean_pts(kp, [L_KNEE, R_KNEE], min_score, w, h)
    ank = _mean_pts(kp, [L_ANK, R_ANK], min_score, w, h)
    if knee is not None:
        row[C["knee_y"]] = knee[1]
    if ank is not None:
        row[C["ankle_y"]] = ank[1]
    angs = []
    for hi, ki, ai in ((L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK)):
        if all(not np.isnan(kp[i, 0]) and kp[i, 2] >= min_score for i in (hi, ki, ai)):
            pts = [np.array([kp[i, 0] * w, kp[i, 1] * h]) for i in (hi, ki, ai)]
            angs.append(_angle3(*pts))
    angs = [a for a in angs if np.isfinite(a)]
    if angs:
        row[C["knee_angle"]] = float(np.mean(angs))
    return row


@dataclass
class VisionWindow:
    t: np.ndarray  # (n,) grid times ms
    f: np.ndarray  # (n, F) features sampled on the grid (NaN where missing)
    mask: np.ndarray  # (n,) 1 = target valid at this grid slot
    fps: float
    flags: list[str] = field(default_factory=list)

    def col(self, name: str) -> np.ndarray:
        return self.f[:, C[name]]


def build_window(
    rows: np.ndarray, t_end: float, length_ms: float, fps: float, hold_ms: float = 130.0
) -> VisionWindow:
    """Sample per-frame rows on a uniform grid ending at t_end.

    Each grid slot takes the nearest frame within ``max(0.75*dt+20, hold_ms)``
    (so a 5 fps idle stream still fills a 15 fps grid); farther slots are masked.
    Only frames at or before ``t_end`` are used (causal)."""
    dt = 1000.0 / fps
    n = int(round(length_ms / dt))
    grid = t_end - dt * np.arange(n - 1, -1, -1)
    f = np.full((n, len(FCOLS)), np.nan)
    mask = np.zeros(n, dtype=np.int8)
    tol = max(0.75 * dt + 20.0, hold_ms)
    if rows is not None and len(rows):
        rows = rows[rows[:, 0] <= t_end + 1e-6]
    if rows is not None and len(rows):
        ts = rows[:, 0]
        idx = np.searchsorted(ts, grid)
        for k, g in enumerate(grid):
            best, bd = -1, 1e18
            for j in (idx[k] - 1, idx[k]):
                if 0 <= j < len(ts):
                    d = abs(ts[j] - g)
                    if d < bd:
                        best, bd = j, d
            if best >= 0 and bd <= tol:
                f[k] = rows[best]
                mask[k] = 1 if rows[best, C["valid"]] == 1 else 0
    return VisionWindow(grid, f, mask, fps)


def _tl_ref(win: VisionWindow, upto: float | None = None) -> float | None:
    m = win.mask.astype(bool)
    if upto is not None:
        m &= win.t <= upto
    tl, torso = win.col("tl"), win.col("torso_deg")
    up = m & (torso < 30)
    if up.sum() >= 3:
        return float(np.nanmedian(tl[up]))
    if m.sum() >= 1:
        return float(np.nanmax(tl[m]))
    return None


def vision_window_features(
    win: VisionWindow,
    cfg: VisionRuleCfg,
    descent_search: tuple[float, float] | None = None,
    truncated: bool = False,
    tl_ref: float | None = None,
) -> dict[str, Any]:
    t, m = win.t, win.mask.astype(bool)
    out: dict[str, Any] = {"valid": False}
    vis = np.nan_to_num(win.col("visible"), nan=0.0)
    out["visible_frac"] = float(vis.mean()) if len(vis) else 0.0
    out["valid_frac"] = float(m.mean()) if len(m) else 0.0
    if m.sum() < 5:
        out["reason"] = "person_not_visible"
        return out
    ref = tl_ref or _tl_ref(win)
    if not ref or ref < 5:
        out["reason"] = "no_scale_reference"
        return out
    yn = win.col("hip_y") / ref
    torso = win.col("torso_deg")
    aspect = win.col("aspect")
    fps = win.fps
    D = max(2, int(round(cfg.descent_max_s * fps)))
    n = len(t)
    drop = np.full(n, np.nan)
    start_idx = np.full(n, -1)
    for k in range(n):
        if not m[k]:
            continue
        lo = max(0, k - D)
        seg = np.where(m[lo : k + 1], yn[lo : k + 1], np.nan)
        if np.all(np.isnan(seg)):
            continue
        j = lo + int(np.nanargmin(seg))
        drop[k] = yn[k] - yn[j]
        start_idx[k] = j
    persist_min = 0.3 if truncated else cfg.persist_min_s
    lo_t, hi_t = t[0] + 300.0, t[-1] - persist_min * 1000.0
    if descent_search is not None:
        lo_t, hi_t = max(lo_t, descent_search[0]), min(hi_t, descent_search[1])
    cand = (t >= lo_t) & (t <= hi_t) & m & ~np.isnan(drop)
    cond = cand & ((drop >= cfg.descent_tl) | ((drop >= cfg.descent_with_tilt_tl) & (torso >= cfg.lying_torso_deg)))
    out.update(valid=True, tl_ref=ref, max_drop_tl=float(np.nanmax(np.where(cand, drop, np.nan))) if cand.any() else 0.0)
    # target lost right after a fast drop (common for single-stage pose models on lying people)
    lost_after_drop = False
    if not cond.any():
        weak = cand & (drop >= 0.5)
        if weak.any():
            k = int(np.where(weak)[0][np.nanargmax(drop[weak])])
            after = t > t[k]
            if after.sum() >= 3 and m[after].mean() < 0.3:
                j = int(start_idx[k])
                dt_s = max((t[k] - t[j]) / 1000.0, 1.0 / fps)
                if drop[k] / dt_s >= cfg.min_speed_tl_s:
                    lost_after_drop = True
                    cond = np.zeros(n, dtype=bool)
                    cond[k] = True
        if not lost_after_drop:
            out.update(descent=False)
            return out
    k = int(np.where(cond)[0][np.nanargmax(drop[cond])])
    j = int(start_idx[k])
    dt_s = max((t[k] - t[j]) / 1000.0, 1.0 / fps)
    # peak descent speed: largest drop over any ~0.4 s sub-window of the descent
    # (a fall often starts slowly; the average over the whole descent hides its fast part)
    n04 = max(1, int(round(0.4 * fps)))
    seg = yn[j : k + 1]
    segm = m[j : k + 1]
    peak = 0.0
    for a in range(len(seg)):
        if not segm[a]:
            continue
        for b in range(a + 1, min(len(seg), a + n04 + 1)):
            if segm[b]:
                peak = max(peak, (seg[b] - seg[a]) / max((b - a) / fps, 1.0 / fps))
    speed = float(max(peak, drop[k] / dt_s))
    ref_y = yn[j]
    post = (t >= t[k] + 300.0) & (t <= t[k] + 300.0 + cfg.persist_window_s * 1000.0)
    post_v = post & m
    post_obs_s = float(post_v.sum() / fps)
    drop_rel = yn - ref_y
    # "low posture" must not include an upright seated person (sit-to-stand, chairs):
    # tilted torso, wide box, a large drop with a leaning torso, or a drop close to floor level.
    low = (
        (torso >= cfg.lying_torso_deg)
        | (np.nan_to_num(aspect) >= 1.1)
        | ((drop_rel >= cfg.low_hip_tl) & (torso >= 35.0))
        | (drop_rel >= cfg.floor_hip_tl)
    )
    low_frac = float(low[post_v].mean()) if post_v.any() else 0.0
    upright_back = post_v & ((yn - ref_y) <= cfg.recovery_tl) & (torso < 35)
    run, best_run = 0, 0
    for v in upright_back:
        run = run + 1 if v else 0
        best_run = max(best_run, run)
    recovered = best_run >= max(1, int(round(cfg.recovery_s * fps)))
    out.update(
        descent=True,
        descent_start_ms=float(t[j]),
        descent_end_ms=float(t[k]),
        drop_tl=float(drop[k]),
        speed_tl_s=speed,
        post_obs_s=post_obs_s,
        low_frac=low_frac,
        torso_post_mean=float(np.nanmean(torso[post_v])) if post_v.any() else float("nan"),
        aspect_post_mean=float(np.nanmean(aspect[post_v])) if post_v.any() else float("nan"),
        recovered=bool(recovered),
        lost_after_drop=bool(lost_after_drop),
        post_visible_frac=float(m[post].mean()) if post.any() else 0.0,
    )
    return out


def quality_of(win: VisionWindow) -> tuple[float, list[str]]:
    flags = list(win.flags)
    m = win.mask.astype(bool)
    vis_frac = float(m.mean()) if len(m) else 0.0
    conf = win.col("conf")
    mean_conf = float(np.nanmean(conf[m])) if m.any() else 0.0
    tconf = win.col("target_conf")
    mean_tc = float(np.nanmean(tconf[m])) if m.any() else 0.0
    amb = np.nan_to_num(win.col("ambiguous"), nan=0.0)
    q = vis_frac * min(1.0, mean_conf / 0.6) * mean_tc
    if vis_frac < 0.6:
        flags.append("person_not_visible")
    if m.any() and mean_conf < 0.5:
        flags.append("low_pose_confidence")
    if amb.mean() > 0.2:
        q *= 0.3
        flags.append("target_ambiguous")
    return float(np.clip(q, 0.0, 1.0)), sorted(set(flags))


class VisionRuleScorer:
    channel = "vision"
    model_id = "vision_rule"
    model_version = "1.0"
    score_semantics = ScoreSemantics.heuristic_score
    calibration_id = None
    lost_after_drop_cap = 0.55

    def __init__(self, cfg: VisionRuleCfg):
        self.cfg = cfg
        self.strong_threshold = cfg.strong_score
        self.candidate_threshold = 0.35
        self.window_ms = cfg.window_s * 1000.0
        self.step_ms = float(cfg.step_ms)
        self.fps = cfg.fps

    def score(
        self, win: VisionWindow, descent_search: tuple[float, float] | None = None, truncated: bool = False
    ) -> ScoreResult:
        c = self.cfg
        f = vision_window_features(win, c, descent_search, truncated)
        q, flags = quality_of(win)
        if not f.get("valid"):
            return ScoreResult(False, 0.0, False, None, f, [f.get("reason", "invalid_window")], q, flags)
        if not f.get("descent"):
            return ScoreResult(True, 0.0, False, None, f, ["no_rapid_descent"], q, flags)
        reasons = ["rapid_descent"]
        if truncated:
            reasons.append("truncated_post_window")
        persist_min = 0.3 if truncated else c.persist_min_s
        fast = f["speed_tl_s"] >= c.min_speed_tl_s
        if not fast:
            reasons.append("descent_slow")
        persistent = f["post_obs_s"] >= persist_min and f["low_frac"] >= c.persist_fraction
        if f["recovered"]:
            reasons.append("recovered_to_standing")
        elif persistent:
            reasons.append("persistent_low_posture")
        d_term = float(np.clip(f["drop_tl"] / 1.3, 0, 1))
        v_term = float(np.clip(f["speed_tl_s"] / 2.0, 0, 1))
        tilt = f.get("torso_post_mean")
        tilt_term = float(np.clip((tilt if np.isfinite(tilt) else 0.0) / 90.0, 0, 1))
        score = 0.3 * d_term + 0.25 * v_term + 0.3 * f["low_frac"] + 0.15 * tilt_term
        candidate = bool(fast and not f["recovered"] and persistent)
        if f["lost_after_drop"]:
            reasons.append("target_lost_after_drop")
            score = min(self.lost_after_drop_cap, 0.3 * d_term + 0.25 * v_term + 0.2)
            candidate = bool(fast)
            q *= 0.7
        elif truncated and not f["recovered"] and f["post_obs_s"] < persist_min and fast and f["drop_tl"] >= c.descent_tl:
            # End of the observed stream right after a fast, deep descent: persistence could not be
            # observed. Report a weak candidate (never strong on its own) instead of silence.
            reasons.append("persistence_not_observed")
            score = min(self.lost_after_drop_cap - 0.05, 0.3 * d_term + 0.25 * v_term + 0.15)
            candidate = True
            q *= 0.7
        if f["recovered"]:
            score *= 0.4
        return ScoreResult(True, float(np.clip(score, 0, 1)), candidate, f["descent_end_ms"], f, reasons, q, flags)

    def assess_contradiction(self, win: VisionWindow, t_ref: float) -> ScoreResult:
        """Channel view around an accelerometer candidate at ``t_ref``."""
        res = self.score(win, descent_search=(t_ref - 2000.0, t_ref + 1500.0))
        if res.candidate:
            return res
        t, m = win.t, win.mask.astype(bool)
        around = (t >= t_ref - 500.0) & (t <= t_ref + 3000.0)
        if not around.any():
            return ScoreResult(False, 0.0, False, None, res.features, ["no_observation_window"], 0.0, res.flags)
        vis_frac = float(m[around].mean())
        amb = np.nan_to_num(win.col("ambiguous"), nan=0.0)
        if vis_frac < 0.5:
            return ScoreResult(False, res.score, False, None, res.features, ["person_not_visible"], res.quality, res.flags + ["person_not_visible"])
        if amb[around].mean() > 0.3:
            return ScoreResult(False, res.score, False, None, res.features, ["target_ambiguous"], res.quality * 0.3, res.flags + ["target_ambiguous"])
        ref = _tl_ref(win)
        torso = win.col("torso_deg")
        sel = around & m
        upright_torso = float((torso[sel] < 40).mean()) if sel.any() else 0.0
        reasons = list(res.reasons)
        contradicts = False
        if res.features.get("recovered"):
            contradicts = True
            reasons.append("recovered_to_standing")
        elif upright_torso >= 0.8 and not res.features.get("descent"):
            contradicts = True
            reasons.append("target_upright_no_descent")
        res.contradicts = contradicts
        res.reasons = sorted(set(reasons))
        res.features["upright_torso_frac"] = upright_torso
        res.features["tl_ref"] = ref
        return res
