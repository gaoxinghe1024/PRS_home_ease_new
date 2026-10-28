"""Seeded synthetic recordings for software, policy and stress testing.

Synthetic data exercise the real ingestion, alignment, tracking, inference and
event code paths. They do NOT demonstrate real-world fall-detection accuracy.
Expected outcomes are written to ``labels.json`` (sidecar) and are never
read by inference; generator parameters are randomised (speed, noise, scale,
impact strength, occlusion, timing) from a fixed seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from homeease.datasets.recording import DetectionFrame, write_recording
from homeease.schemas.observations import AccelBatch, AccelSample, TimeSyncReport, TimeSyncSample

GEN_VERSION = "synthetic-1.1"
W, H = 640, 480
G = 9.80665


# --------------------------------------------------------------------------- pose model
@dataclass
class BodyState:
    cx: float
    hy: float
    theta: float = 0.0  # torso tilt from vertical (deg)
    phi: float = 0.0  # thigh flexion (0 standing, 90 seated)
    arm: float = 0.0  # arm swing amplitude (deg)
    visible: float = 1.0
    conf: float = 0.9
    lower_occluded: float = 0.0


def skeleton(s: BodyState, S: float, t: float, view: str = "front") -> np.ndarray:
    """Return (17,2) pixel coordinates of a COCO-17 skeleton (front or side view)."""
    ph = math.radians(s.phi)
    sw = math.radians(s.arm) * math.sin(2 * math.pi * 1.8 * t)
    pts = np.zeros((17, 2))
    hipL, hipR = np.array([0.2 * S, 0.0]), np.array([-0.2 * S, 0.0])
    shL, shR = np.array([0.3 * S, -S]), np.array([-0.3 * S, -S])
    pts[11], pts[12] = hipL, hipR
    pts[5], pts[6] = shL, shR
    pts[0] = [0.0, -1.45 * S]
    pts[1], pts[2] = [0.07 * S, -1.52 * S], [-0.07 * S, -1.52 * S]
    pts[3], pts[4] = [0.16 * S, -1.47 * S], [-0.16 * S, -1.47 * S]
    for sh, el, wr, sgn in ((5, 7, 9, 1), (6, 8, 10, -1)):
        a = sgn * sw
        pts[el] = pts[sh] + 0.62 * S * np.array([math.sin(a), math.cos(a)])
        pts[wr] = pts[el] + 0.55 * S * np.array([math.sin(a * 1.2), math.cos(a * 1.2)])
    for hp, kn, an, sgn in ((11, 13, 15, 1), (12, 14, 16, -1)):
        lateral = 1.0 * S * math.sin(ph) if view == "side" else 0.12 * S * math.sin(ph) * sgn
        knee = pts[hp] + np.array([lateral, 1.0 * S * math.cos(ph)])
        pts[kn] = knee
        pts[an] = knee + np.array([0.0, 1.0 * S])
    th = math.radians(s.theta)
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    pts = pts @ R.T
    pts[:, 0] += s.cx
    pts[:, 1] += s.hy
    return pts


def _ease(kind: str, u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    if kind == "linear":
        return u
    if kind == "accel":  # gravity-like
        return u * u
    return u * u * (3 - 2 * u)


@dataclass
class Motion:
    """Keyframed body motion plus aligned accelerometer events and labels."""

    state: BodyState
    S: float
    floor: float
    t: float = 0.0
    view: str = "front"
    kf: list[tuple[float, BodyState, str]] = field(default_factory=list)
    accel_events: list[dict[str, Any]] = field(default_factory=list)
    labels: dict[str, list[Any]] = field(default_factory=lambda: {"falls": [], "adl": [], "reps": [], "phases": []})

    def __post_init__(self) -> None:
        self.kf.append((0.0, BodyState(**self.state.__dict__), "smooth"))

    @property
    def stand_hy(self) -> float:
        return self.floor - 2.0 * self.S

    @property
    def sit_hy(self) -> float:
        return self.floor - 1.05 * self.S

    @property
    def lie_hy(self) -> float:
        return self.floor - 0.18 * self.S

    def to(self, dur: float, ease: str = "smooth", **ch: float) -> "Motion":
        st = BodyState(**{**self.kf[-1][1].__dict__, **ch})
        self.t += dur
        self.kf.append((self.t, st, ease))
        return self

    def hold(self, dur: float) -> "Motion":
        return self.to(dur, "linear")

    def phase(self, name: str, t0: float, t1: float) -> None:
        self.labels["phases"].append({"phase": name, "start_ms": t0 * 1000, "end_ms": t1 * 1000})

    # ---- composite actions (pose + accel + labels)
    def walk(self, dur: float, dx: float) -> "Motion":
        t0 = self.t
        self.to(dur, "linear", cx=self.kf[-1][1].cx + dx, arm=18.0)
        self.to(0.3, "smooth", arm=0.0)
        self.accel_events.append({"kind": "walk", "t0": t0, "t1": self.t})
        self.labels["adl"].append({"activity": "walk", "start_ms": t0 * 1000, "end_ms": self.t * 1000})
        return self

    def sit_down(self, dur: float = 1.6, chair_drop: bool = False) -> "Motion":
        t0 = self.t
        self.to(dur * 0.5, "smooth", theta=22.0, phi=45.0, hy=(self.stand_hy + self.sit_hy) / 2)
        self.to(dur * 0.5, "accel" if chair_drop else "smooth", theta=4.0, phi=88.0, hy=self.sit_hy)
        self.accel_events.append({"kind": "transition", "t0": t0, "t1": self.t, "impact": 1.6 if chair_drop else 1.2})
        self.labels["adl"].append({"activity": "sit_down", "start_ms": t0 * 1000, "end_ms": self.t * 1000})
        return self

    def stand_up(self, dur: float = 1.4) -> "Motion":
        t0 = self.t
        self.to(dur * 0.5, "smooth", theta=25.0, phi=45.0, hy=(self.stand_hy + self.sit_hy) / 2)
        self.to(dur * 0.5, "smooth", theta=2.0, phi=0.0, hy=self.stand_hy)
        self.accel_events.append({"kind": "transition", "t0": t0, "t1": self.t, "impact": 1.15})
        self.labels["adl"].append({"activity": "stand_up", "start_ms": t0 * 1000, "end_ms": self.t * 1000})
        return self

    def fall(self, rng: np.random.Generator, dur: float | None = None, impact_g: float | None = None) -> "Motion":
        dur = dur or float(rng.uniform(0.55, 0.95))
        onset = self.t
        side = 1 if rng.random() < 0.5 else -1
        self.to(dur, "accel", theta=side * float(rng.uniform(70, 92)), phi=float(rng.uniform(0, 25)), hy=self.lie_hy, arm=0.0, conf=0.7)
        impact = self.t
        self.accel_events.append({"kind": "fall", "t": impact, "impact": impact_g or float(rng.uniform(2.6, 5.0))})
        self.labels["falls"].append({"onset_ms": onset * 1000, "impact_ms": impact * 1000, "end_ms": (impact + 1.0) * 1000})
        return self

    def lie_down(self) -> "Motion":
        t0 = self.t
        self.sit_down(1.8)
        self.to(2.4, "smooth", theta=-88.0, phi=10.0, hy=self.lie_hy + 5)
        self.labels["adl"].append({"activity": "lie_down", "start_ms": t0 * 1000, "end_ms": self.t * 1000})
        self.accel_events.append({"kind": "transition", "t0": t0 + 1.8, "t1": self.t, "impact": 1.15, "orient": [0.95, 0.1, 0.3]})
        return self

    def get_up_from_lying(self) -> "Motion":
        t0 = self.t
        self.to(2.0, "smooth", theta=0.0, phi=88.0, hy=self.sit_hy)
        self.stand_up(1.6)
        self.accel_events.append({"kind": "transition", "t0": t0, "t1": self.t, "impact": 1.2, "orient": "default"})
        return self

    def bend(self) -> "Motion":
        t0 = self.t
        self.to(1.0, "smooth", theta=72.0, phi=15.0, hy=self.stand_hy + 0.35 * self.S)
        self.hold(1.0)
        self.to(1.0, "smooth", theta=0.0, phi=0.0, hy=self.stand_hy)
        self.accel_events.append({"kind": "transition", "t0": t0, "t1": self.t, "impact": 1.25})
        self.labels["adl"].append({"activity": "bend", "start_ms": t0 * 1000, "end_ms": self.t * 1000})
        return self

    def sit_to_stand_rep(self, rng: np.random.Generator, rise: float | None = None, hold: float | None = None, desc: float | None = None, rest: float | None = None) -> "Motion":
        rise = rise or float(rng.uniform(0.9, 1.6))
        hold = hold or float(rng.uniform(0.4, 1.4))
        desc = desc or float(rng.uniform(1.0, 1.7))
        rest = rest or float(rng.uniform(0.5, 1.4))
        t0 = self.t
        self.to(rise * 0.5, "smooth", theta=28.0, phi=50.0, hy=(self.stand_hy + self.sit_hy) / 2)
        self.to(rise * 0.5, "smooth", theta=2.0, phi=0.0, hy=self.stand_hy)
        t1 = self.t
        self.hold(hold)
        t2 = self.t
        self.to(desc * 0.5, "smooth", theta=24.0, phi=45.0, hy=(self.stand_hy + self.sit_hy) / 2)
        self.to(desc * 0.5, "smooth", theta=4.0, phi=88.0, hy=self.sit_hy)
        t3 = self.t
        self.hold(rest)
        self.phase("RISING", t0, t1)
        self.phase("STANDING", t1, t2)
        self.phase("DESCENDING", t2, t3)
        self.phase("SITTING", t3, self.t)
        self.labels["reps"].append({"start_ms": t0 * 1000, "stand_ms": t1 * 1000, "end_ms": t3 * 1000})
        self.accel_events.append({"kind": "transition", "t0": t0, "t1": t3, "impact": 1.15})
        return self

    def state_at(self, t: float) -> BodyState:
        kf = self.kf
        if t <= kf[0][0]:
            return kf[0][1]
        for i in range(1, len(kf)):
            if t <= kf[i][0]:
                (ta, a, _), (tb, b, ease) = kf[i - 1], kf[i]
                u = _ease(ease, (t - ta) / max(tb - ta, 1e-6))
                return BodyState(**{k: getattr(a, k) + (getattr(b, k) - getattr(a, k)) * u for k in a.__dict__})
        return kf[-1][1]


def render_frames(
    persons: list[Motion],
    duration_s: float,
    fps: float,
    rng: np.random.Generator,
    camera_off: list[tuple[float, float]] = (),
    occlusion: list[tuple[float, float, str]] = (),
    backend: str = "synthetic",
) -> list[DetectionFrame]:
    frames = []
    n = int(duration_s * fps)
    for i in range(n):
        t = i / fps
        if any(a <= t < b for a, b in camera_off):
            continue
        persons_kp = []
        for m in persons:
            s = m.state_at(t)
            if s.visible < 0.5:
                continue
            pts = skeleton(s, m.S, t, m.view) + rng.normal(0, 1.5, size=(17, 2))
            kp = np.full((17, 3), np.nan)
            conf = np.clip(rng.normal(s.conf, 0.06, size=17), 0.05, 0.99)
            kp[:, 0] = pts[:, 0] / W
            kp[:, 1] = pts[:, 1] / H
            kp[:, 2] = conf
            out = (kp[:, 0] < 0) | (kp[:, 0] > 1) | (kp[:, 1] < 0) | (kp[:, 1] > 1)
            kp[out] = np.nan
            drop = rng.random(17) < 0.03
            kp[drop] = np.nan
            for a, b, kind in occlusion:
                if a <= t < b:
                    if kind == "full":
                        kp[:] = np.nan
                    elif kind == "lower":
                        kp[11:] = np.nan
            if np.sum(~np.isnan(kp[:, 0])) >= 3:
                persons_kp.append(kp)
        frames.append(DetectionFrame(t * 1000.0, i, W, H, persons_kp, backend, round(float(rng.uniform(4, 12)), 2)))
    return frames


# ---------------------------------------------------------------------- accel model
def _unit(v: Any) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    return a / np.linalg.norm(a)


DEFAULT_DIR = _unit([0.15, -0.97, 0.18])


def synth_accel(
    events: list[dict[str, Any]],
    duration_s: float,
    rng: np.random.Generator,
    carried: Callable[[float], bool] = lambda t: True,
    table_events: list[dict[str, Any]] = (),
    fs: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_s, acc m/s^2 incl. gravity) with jittered sampling."""
    n = int(duration_s * fs)
    t = np.arange(n) / fs + rng.normal(0, 0.0015, n)
    t = np.sort(np.clip(t, 0, None))
    keep = rng.random(n) > 0.005
    t = t[keep]
    gdir = np.tile(DEFAULT_DIR, (len(t), 1))
    dyn = np.zeros((len(t), 3))
    carried_mask = np.array([carried(x) for x in t])
    orient_changes: list[tuple[float, np.ndarray]] = []
    for ev in events:
        k = ev["kind"]
        if k == "walk":
            sel = (t >= ev["t0"]) & (t <= ev["t1"])
            ph = rng.uniform(0, 2 * np.pi)
            dyn[sel, 1] += 0.22 * np.sin(2 * np.pi * 1.8 * t[sel])
            dyn[sel, 0] += 0.12 * np.sin(2 * np.pi * 0.9 * t[sel] + ph)
            dyn[sel] += rng.normal(0, 0.04, (sel.sum(), 3))
        elif k == "transition":
            mid, wdt = (ev["t0"] + ev["t1"]) / 2, max(0.2, (ev["t1"] - ev["t0"]) / 3)
            bump = np.exp(-(((t - mid) / wdt) ** 2))
            dyn += np.outer(bump, _unit(rng.normal(size=3)) * 0.25)
            imp = ev.get("impact", 1.2) - 1.0
            if imp > 0:
                dyn += np.outer(np.exp(-(((t - ev["t1"]) / 0.03) ** 2)), _unit(rng.normal(size=3)) * imp)
            if "orient" in ev:
                orient_changes.append((ev["t1"], DEFAULT_DIR if ev["orient"] == "default" else _unit(ev["orient"])))
        elif k == "fall":
            ti, A = ev["t"], ev["impact"]
            ff = (t >= ti - 0.35) & (t < ti - 0.03)
            dyn[ff] -= 0.6 * DEFAULT_DIR
            d = _unit(rng.normal(size=3))
            env = np.exp(-(((t - ti) / 0.025) ** 2)) * A
            osc = np.where(t > ti, np.exp(-(t - ti) / 0.08) * np.sin(2 * np.pi * 9 * (t - ti)) * 0.8, 0)
            dyn += np.outer(env + osc, d)
            orient_changes.append((ti + 0.05, _unit(rng.normal(size=3) * [1, 0.3, 1])))
        elif k == "clap":
            for tc in np.arange(ev["t0"], ev["t1"], 0.4):
                dyn += np.outer(np.exp(-(((t - tc) / 0.015) ** 2)) * rng.uniform(1.8, 2.8), _unit(rng.normal(size=3)))
            sel = (t >= ev["t0"]) & (t <= ev["t1"] + 1.5)
            dyn[sel] += rng.normal(0, 0.12, (sel.sum(), 3))
        elif k == "hit":
            dyn += np.outer(np.exp(-(((t - ev["t"]) / 0.012) ** 2)) * rng.uniform(2.5, 3.8), _unit(rng.normal(size=3)))
            sel = (t > ev["t"] + 0.1) & (t < ev["t"] + 2.5)
            dyn[sel] += rng.normal(0, 0.15, (sel.sum(), 3)) + 0.2 * np.sin(2 * np.pi * 1.2 * t[sel])[:, None]
    for tc, nd in sorted(orient_changes, key=lambda x: x[0]):
        gdir[t >= tc] = nd
    acc = gdir + dyn
    # phone not carried: on a table, flat, tiny noise (table events may apply)
    table_dir = _unit([0.01, 0.02, 1.0])
    acc[~carried_mask] = table_dir
    for ev in table_events:
        if ev["kind"] == "drop":
            ti = ev["t"]
            ff = (t >= ti - 0.33) & (t < ti - 0.02)
            acc[ff] = acc[ff] * 0.03
            acc += np.outer(np.exp(-(((t - ti) / 0.02) ** 2)) * rng.uniform(4.0, 6.5), _unit(rng.normal(size=3)))
            acc += np.outer(np.exp(-(((t - ti - 0.17) / 0.02) ** 2)) * 1.4, _unit(rng.normal(size=3)))
            acc[t >= ti + 0.25] = _unit([0.02, 0.01, 1.0])
    acc += rng.normal(0, 0.012, acc.shape)
    acc = np.clip(acc, -7.9, 7.9)
    return t, acc * G


def make_batches(
    t_s: np.ndarray,
    acc: np.ndarray,
    *,
    device_id: str,
    subject_id: str,
    session_id: str,
    device_offset_ms: float,
    drift_ppm: float = 0.0,
    batch_ms: float = 500.0,
    placement: str = "wrist",
    clock_jump: tuple[float, float] | None = None,
    source_mode: str = "synthetic",
) -> list[AccelBatch]:
    batches: list[AccelBatch] = []
    rec_ms = t_s * 1000.0
    dev_ms = rec_ms * (1 + drift_ppm * 1e-6) + device_offset_ms
    if clock_jump is not None:
        dev_ms = dev_ms + np.where(rec_ms >= clock_jump[0] * 1000.0, clock_jump[1], 0.0)
    if len(rec_ms) == 0:
        return batches
    edges = np.arange(0, rec_ms[-1] + batch_ms, batch_ms)
    seq = 0
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (rec_ms >= a) & (rec_ms < b)
        if not sel.any():
            continue
        samples = [AccelSample(t_ms=round(float(d), 3), ax=float(x), ay=float(y), az=float(z)) for d, (x, y, z) in zip(dev_ms[sel], acc[sel])]
        batches.append(
            AccelBatch(
                device_id=device_id, subject_id=subject_id, session_id=session_id, source_mode=source_mode,
                collector_type="synthetic", sequence_no=seq, placement=placement, unit="m_s2", includes_gravity=True,
                timebase="device_monotonic_ms", clock_sync_id="sync-1", reported_rate_hz=50.0, samples=samples,
                sent_at_device_ms=float(dev_ms[sel][-1] + 5),
            )
        )
        seq += 1
    return batches


def sync_samples(rng: np.random.Generator, device_offset_ms: float, at_rec_ms: float, n: int = 8) -> list[TimeSyncSample]:
    out = []
    t = at_rec_ms
    for _ in range(n):
        up, down = rng.uniform(5, 40), rng.uniform(5, 40)
        proc = rng.uniform(0.2, 2.0)
        t0_rec = t
        t1 = t0_rec + up
        t2 = t1 + proc
        t3_rec = t2 + down
        out.append(TimeSyncSample(t0_device_ms=t0_rec + device_offset_ms, t1_server_mono_ms=t1, t2_server_mono_ms=t2, t3_device_ms=t3_rec + device_offset_ms))
        t = t3_rec + rng.uniform(20, 60)
    return out


# ------------------------------------------------------------------------ scenarios
@dataclass
class Scenario:
    name: str
    description: str
    duration_s: float
    persons: list[Motion]
    expect: dict[str, Any]
    carried: Callable[[float], bool] = lambda t: True
    table_events: list[dict[str, Any]] = field(default_factory=list)
    camera_off: list[tuple[float, float]] = field(default_factory=list)
    occlusion: list[tuple[float, float, str]] = field(default_factory=list)
    phone_outage: list[tuple[float, float, str]] = field(default_factory=list)  # (t0, t1, 'drop'|'backlog')
    phone_off_after: float | None = None
    clock_jump: tuple[float, float] | None = None
    exercise: dict[str, Any] | None = None
    fps: float = 15.0
    placement: str = "wrist"


def _person(rng: np.random.Generator, cx: float = 320.0, seated: bool = False) -> Motion:
    S = float(rng.uniform(58, 78))
    floor = float(rng.uniform(430, 455))
    m = Motion(BodyState(cx=cx, hy=0.0), S, floor)
    hy = m.sit_hy if seated else m.stand_hy
    st = BodyState(cx=cx, hy=hy, theta=4.0 if seated else 0.0, phi=88.0 if seated else 0.0)
    m.state = st
    m.kf = [(0.0, BodyState(**st.__dict__), "smooth")]
    return m


def build_scenario(name: str, rng: np.random.Generator) -> Scenario:
    if name == "normal_day":
        m = _person(rng, 200)
        m.hold(3).walk(4, 180).hold(2).sit_down().hold(4).stand_up().hold(2).bend().hold(2).walk(3, -150)
        m.hold(2).lie_down().hold(6).get_up_from_lying().hold(3)
        return Scenario(name, "S01 everyday activities incl. deliberate lying down", m.t + 1, [m], {"max_level": "PENDING", "no_emergency": True})
    if name == "phone_drop_standing":
        m = _person(rng, 300)
        m.hold(4).walk(3, 60).hold(8)
        drop_t = 9.0
        return Scenario(
            name, "S02 phone dropped while the person keeps standing", m.t + 1, [m],
            {"max_level": "NORMAL", "rejected_candidate": True},
            carried=lambda t, d=drop_t: t < d, table_events=[{"kind": "drop", "t": drop_t}],
        )
    if name == "fall_both":
        m = _person(rng, 220)
        m.hold(3).walk(3, 120).hold(1)
        m.fall(rng)
        m.hold(12)
        return Scenario(name, "S03 controlled fall seen by both channels", m.t + 1, [m], {"max_level": "EMERGENCY", "events": 1})
    if name == "fall_phone_on_table":
        m = _person(rng, 260)
        m.hold(3).walk(3, 100).hold(1)
        m.fall(rng)
        m.hold(12)
        return Scenario(name, "S04 fall while the phone lies on a table", m.t + 1, [m], {"max_level": "PENDING", "reason": "accel_no_impact"}, carried=lambda t: False, placement="table")
    if name == "fall_camera_occluded":
        m = _person(rng, 260)
        m.hold(3).walk(3, 100).hold(1)
        m.fall(rng)
        m.hold(12)
        return Scenario(name, "S05 fall while the camera is occluded", m.t + 1, [m], {"max_level": "PENDING", "coverage": "DEGRADED"}, occlusion=[(5.0, m.t + 1, "full")])
    if name == "no_channels":
        m = _person(rng, 260)
        m.hold(6).walk(3, 80).hold(25)
        return Scenario(name, "S06 phone and camera both stop", m.t + 1, [m], {"coverage_final": "UNAVAILABLE"}, camera_off=[(10.0, m.t + 1)], phone_off_after=10.0)
    if name == "clock_offset":
        m = _person(rng, 240)
        m.hold(3).walk(3, 100).hold(1)
        m.fall(rng)
        m.hold(12)
        return Scenario(name, "S07 device clock jumps after sync; old batches re-sent late", m.t + 1, [m], {"no_emergency": True, "flags": ["timestamp_untrusted"]}, clock_jump=(4.0, 2500.0))
    if name == "visitor":
        m = _person(rng, 200)
        v = _person(rng, 440)
        m.hold(3).walk(2, 60).hold(1)
        m.fall(rng)
        m.hold(10)
        v.hold(2).walk(4, -150).hold(12)
        return Scenario(name, "S08 visitor in frame; identity association uncertain", m.t + 1, [m, v], {"no_emergency_from_visitor": True})
    if name == "exercise_sts":
        m = _person(rng, 320, seated=True)
        m.hold(2.0)
        m.phase("SITTING", 0, 2.0)
        n = int(rng.integers(6, 11))
        for _ in range(n):
            m.sit_to_stand_rep(rng)
        m.hold(2)
        return Scenario(name, "S12 correct full sit-to-stand cycles", m.t + 1, [m], {"reps": n}, exercise={"target": 10}, fps=15.0)
    if name == "exercise_hard":
        m = _person(rng, 320, seated=False)
        m.hold(2.0)
        m.sit_down(1.5)
        m.hold(1.5)
        reps = 0
        for i in range(4):
            if i == 1:
                # half rise: does not reach standing
                t0 = m.t
                m.to(0.7, "smooth", theta=25.0, phi=55.0, hy=(m.stand_hy * 0.35 + m.sit_hy * 0.65))
                m.to(0.8, "smooth", theta=4.0, phi=88.0, hy=m.sit_hy)
                m.hold(1.0)
                m.labels["adl"].append({"activity": "half_rise", "start_ms": t0 * 1000, "end_ms": m.t * 1000})
            elif i == 2:
                # stand with jitter, then sit: counts as one full cycle
                t0 = m.t
                m.to(0.7, "smooth", theta=26.0, phi=45.0, hy=(m.stand_hy + m.sit_hy) / 2)
                m.to(0.7, "smooth", theta=2.0, phi=0.0, hy=m.stand_hy)
                t1 = m.t
                for _ in range(4):
                    m.to(0.25, "smooth", phi=12.0, hy=m.stand_hy + 0.12 * m.S)
                    m.to(0.25, "smooth", phi=0.0, hy=m.stand_hy)
                t2 = m.t
                m.to(0.7, "smooth", theta=24.0, phi=45.0, hy=(m.stand_hy + m.sit_hy) / 2)
                m.to(0.7, "smooth", theta=4.0, phi=88.0, hy=m.sit_hy)
                m.labels["reps"].append({"start_ms": t0 * 1000, "stand_ms": t1 * 1000, "end_ms": m.t * 1000})
                m.phase("STANDING", t1, t2)
                m.hold(1.0)
                reps += 1
            else:
                m.sit_to_stand_rep(rng)
                reps += 1
        m.hold(2)
        return Scenario(name, "S13 starts standing, half rise, standing jitter", m.t + 1, [m], {"reps": reps}, exercise={"target": 5})
    if name == "exercise_zero":
        m = _person(rng, 320, seated=True)
        m.hold(20)
        return Scenario(name, "S13 zero-repetition session (sits still)", m.t + 1, [m], {"reps": 0}, exercise={"target": 5})
    if name == "exercise_occlusion":
        m = _person(rng, 320, seated=True)
        m.hold(2.0)
        for _ in range(3):
            m.sit_to_stand_rep(rng)
        occ0 = m.t + 0.3
        m.sit_to_stand_rep(rng)  # this rep is occluded mid-way
        occ1 = occ0 + 3.0
        for _ in range(3):
            m.sit_to_stand_rep(rng)
        m.hold(2)
        return Scenario(name, "S14 occlusion during exercise; occluded cycle not counted", m.t + 1, [m], {"reps_max": 6, "reps_min": 5}, exercise={"target": 10}, occlusion=[(occ0, occ1, "full")])
    if name == "exercise_fall":
        m = _person(rng, 320, seated=True)
        m.hold(2.0)
        for _ in range(3):
            m.sit_to_stand_rep(rng)
        m.stand_up(1.4)
        m.hold(1.0)
        m.fall(rng)
        m.hold(12)
        return Scenario(name, "S15 fall during an exercise session", m.t + 1, [m], {"max_level": "EMERGENCY", "exercise_suspended": True}, exercise={"target": 10})
    if name == "network_outage":
        m = _person(rng, 260)
        m.hold(3).walk(4, 100).hold(30)
        return Scenario(name, "S16 phone network outage with backlog", m.t + 1, [m], {"coverage_during_outage": "DEGRADED"}, phone_outage=[(8.0, 22.0, "backlog")])
    if name == "clap_hard_negative":
        m = _person(rng, 300)
        m.hold(3).walk(2, 40).hold(8)
        m.accel_events.append({"kind": "clap", "t0": 6.0, "t1": 8.0})
        m.accel_events.append({"kind": "hit", "t": 10.5})
        return Scenario(name, "Hard negatives: clapping and hitting a table", m.t + 1, [m], {"no_emergency": True})
    raise KeyError(name)


SCENARIOS = [
    "normal_day", "phone_drop_standing", "fall_both", "fall_phone_on_table", "fall_camera_occluded", "no_channels",
    "clock_offset", "visitor", "exercise_sts", "exercise_hard", "exercise_zero", "exercise_occlusion",
    "exercise_fall", "network_outage", "clap_hard_negative",
]


def generate_scenario(name: str, seed: int, out_dir: str | Path, subject_id: str = "elder-01") -> Path:
    rng = np.random.default_rng(seed)
    sc = build_scenario(name, rng)
    dur = sc.duration_s
    frames = render_frames(sc.persons, dur, sc.fps, rng, sc.camera_off, sc.occlusion)
    target = sc.persons[0]
    t_s, acc = synth_accel(target.accel_events, dur, rng, sc.carried, sc.table_events)
    if sc.phone_off_after is not None:
        keep = t_s < sc.phone_off_after
        t_s, acc = t_s[keep], acc[keep]
    device_offset = float(rng.uniform(1e5, 9e5))
    drift = float(rng.uniform(-30, 30))
    batches = make_batches(
        t_s, acc, device_id="phone-01", subject_id=subject_id, session_id=f"cap-{name}-{seed}",
        device_offset_ms=device_offset, drift_ppm=drift, placement=sc.placement, clock_jump=sc.clock_jump,
    )
    arrivals: dict[str, float] = {}
    for b in batches:
        last_rec = (b.samples[-1].t_ms - device_offset) / (1 + drift * 1e-6)
        if sc.clock_jump and last_rec >= sc.clock_jump[0] * 1000:
            last_rec -= sc.clock_jump[1]
        arr = last_rec + float(rng.uniform(60, 160))
        for o0, o1, kind in sc.phone_outage:
            if o0 * 1000 <= last_rec < o1 * 1000:
                arr = o1 * 1000 + float(rng.uniform(100, 2000)) if kind == "backlog" else -1
        if arr >= 0:
            arrivals[str(b.sequence_no)] = round(arr, 2)
    batches = [b for b in batches if str(b.sequence_no) in arrivals]
    # The collector syncs its clock before streaming starts (recording time < 0).
    ss = sync_samples(rng, device_offset, -1500.0)
    sync_received = max(s.t3_device_ms for s in ss) - device_offset + 5.0
    report = TimeSyncReport(session_id=f"cap-{name}-{seed}", sync_id="sync-1", samples=ss)
    exercise = None
    if sc.exercise:
        exercise = {"start_ms": 500.0, "target": sc.exercise["target"], "definition": "full_cycle"}
    labels = {
        "scenario": name,
        "description": sc.description,
        "falls": target.labels["falls"],
        "adl": target.labels["adl"],
        "reps": target.labels["reps"],
        "phases": target.labels["phases"],
        "rep_count": len(target.labels["reps"]),
        "expect": sc.expect,
        "faults": {
            "camera_off": sc.camera_off, "occlusion": sc.occlusion, "phone_outage": sc.phone_outage,
            "clock_jump": sc.clock_jump, "phone_off_after": sc.phone_off_after,
        },
        "label_source": "generator",
    }
    manifest = {
        "recording_id": f"syn-{name}-{seed}",
        "dataset": "synthetic",
        "source_mode": "synthetic",
        "generator": GEN_VERSION,
        "seed": seed,
        "scenario": name,
        "subject_id": subject_id,
        "group_id": f"synthetic-{seed}",
        "room_id": "living-room",
        "duration_ms": dur * 1000.0,
        "placement": sc.placement,
        "units": "m_s2",
        "accel_includes_gravity": True,
        "sample_rate_hz": 50.0,
        "fps": sc.fps,
        "joint_layout": "coco17-v1",
        "timebase": {"camera": "recording_ms", "accel": "device_monotonic_ms"},
        "sync_method": "simulated ping/pong",
        "clock_sync": [{"device_id": "phone-01", "report": report.model_dump(), "received_at_ms": round(sync_received, 3)}],
        "accel_arrivals_ms": arrivals,
        "camera_device_id": "camera-01",
        "exercise_session": exercise,
        "available_modalities": ["accel", "pose"],
        "conditions": {"lighting": "synthetic", "occlusion": bool(sc.occlusion), "phone_placement": sc.placement, "persons": len(sc.persons)},
        "license": "generated (no external data)",
    }
    return write_recording(Path(out_dir) / manifest["recording_id"], manifest, batches, frames, labels)


def generate_suite(out_dir: str | Path, seed: int = 42, scenarios: list[str] | None = None) -> list[Path]:
    out = []
    for i, name in enumerate(scenarios or SCENARIOS):
        out.append(generate_scenario(name, seed + i, out_dir))
    return out


# ------------------------------------------------------------ exercise sets
def generate_exercise_session(seed: int, out_dir: str | Path, conditions: dict[str, Any] | None = None) -> Path:
    """One sit-to-stand session with randomised tempo, chair height, view, partial reps,
    standing start, occlusion and zero-rep sessions. Phase labels go to labels.json."""
    rng = np.random.default_rng(seed)
    c = dict(conditions or {})
    view = c.get("view") or ("side" if rng.random() < 0.5 else "front")
    tempo = float(c.get("tempo", rng.uniform(0.7, 1.6)))
    n_reps = int(c.get("reps", rng.integers(0, 13)))
    start_standing = bool(c.get("start_standing", rng.random() < 0.2))
    m = _person(rng, float(rng.uniform(250, 390)), seated=not start_standing)
    m.view = view
    chair = float(c.get("chair", rng.uniform(0.9, 1.25)))
    m.floor = m.floor  # chair height changes the seated hip height
    sit_hy = m.floor - 1.05 * m.S * chair
    if not start_standing:
        st = BodyState(cx=m.kf[0][1].cx, hy=sit_hy, theta=4.0, phi=88.0)
        m.kf = [(0.0, st, "smooth")]
    m.hold(1.5)
    if start_standing:
        t0 = m.t
        m.to(1.4 * tempo, "smooth", theta=4.0, phi=88.0, hy=sit_hy)
        m.labels["adl"].append({"activity": "initial_sit", "start_ms": t0 * 1000, "end_ms": m.t * 1000})
        m.hold(1.0)
    m.phase("SITTING", 0, m.t)
    half_rate = float(c.get("half_rate", 0.12))
    jitter_rate = float(c.get("jitter_rate", 0.15))
    reps_done = 0
    while reps_done < n_reps:
        if rng.random() < half_rate:
            t0 = m.t
            m.to(0.6 * tempo, "smooth", theta=25.0, phi=55.0, hy=(m.stand_hy * 0.3 + sit_hy * 0.7))
            m.phase("RISING", t0, m.t)
            t1 = m.t
            m.to(0.7 * tempo, "smooth", theta=4.0, phi=88.0, hy=sit_hy)
            m.phase("DESCENDING", t1, m.t)
            t2 = m.t
            m.hold(float(rng.uniform(0.5, 1.2)))
            m.phase("SITTING", t2, m.t)
            m.labels["adl"].append({"activity": "half_rise", "start_ms": t0 * 1000, "end_ms": m.t * 1000})
            continue
        r = float(rng.uniform(0.9, 1.5)) * tempo
        h = float(rng.uniform(0.3, 1.2)) * tempo
        d = float(rng.uniform(1.0, 1.6)) * tempo
        rest = float(rng.uniform(0.4, 1.2)) * tempo
        t0 = m.t
        m.to(r * 0.5, "smooth", theta=28.0, phi=50.0, hy=(m.stand_hy + sit_hy) / 2)
        m.to(r * 0.5, "smooth", theta=2.0, phi=0.0, hy=m.stand_hy)
        t1 = m.t
        if rng.random() < jitter_rate:
            for _ in range(3):
                m.to(0.2, "smooth", phi=10.0, hy=m.stand_hy + 0.1 * m.S)
                m.to(0.2, "smooth", phi=0.0, hy=m.stand_hy)
        m.hold(h)
        t2 = m.t
        m.to(d * 0.5, "smooth", theta=24.0, phi=45.0, hy=(m.stand_hy + sit_hy) / 2)
        m.to(d * 0.5, "smooth", theta=4.0, phi=88.0, hy=sit_hy)
        t3 = m.t
        m.hold(rest)
        m.phase("RISING", t0, t1)
        m.phase("STANDING", t1, t2)
        m.phase("DESCENDING", t2, t3)
        m.phase("SITTING", t3, m.t)
        m.labels["reps"].append({"start_ms": t0 * 1000, "stand_ms": t1 * 1000, "end_ms": t3 * 1000})
        reps_done += 1
    m.hold(1.5)
    m.phase("SITTING", m.t - 1.5, m.t)
    dur = m.t + 0.5
    occl = []
    if c.get("occlusion", rng.random() < 0.2) and dur > 8:
        o0 = float(rng.uniform(3, dur - 4))
        occl = [(o0, o0 + float(rng.uniform(1.0, 3.0)), "full")]
    fps = float(c.get("fps", 15.0))
    frames = render_frames([m], dur, fps, rng, occlusion=occl)
    # repetitions overlapping an occlusion are not countable from video: mark them
    countable = [r for r in m.labels["reps"] if not any(a * 1000 < r["end_ms"] and b * 1000 > r["start_ms"] for a, b, _ in occl)]
    labels = {
        "scenario": "exercise_set", "reps": m.labels["reps"], "rep_count": len(m.labels["reps"]), "rep_count_visible": len(countable),
        "phases": m.labels["phases"], "adl": m.labels["adl"], "falls": [], "label_source": "generator",
        "conditions": {"view": view, "tempo": round(tempo, 2), "chair": round(chair, 2), "start_standing": start_standing, "occlusion": bool(occl), "zero_reps": n_reps == 0},
    }
    manifest = {
        "recording_id": f"syn-ex-{seed}", "dataset": "synthetic_exercise", "source_mode": "synthetic", "generator": GEN_VERSION, "seed": seed,
        "subject_id": f"synthetic-{seed}", "group_id": f"synthetic-{seed}", "fps": fps, "joint_layout": "coco17-v1", "duration_ms": dur * 1000.0,
        "available_modalities": ["pose"], "exercise_session": {"start_ms": 0.0, "target": 10, "definition": "full_cycle"},
        "conditions": labels["conditions"], "license": "generated (no external data)", "kind": "exercise",
    }
    return write_recording(Path(out_dir) / manifest["recording_id"], manifest, None, frames, labels)
