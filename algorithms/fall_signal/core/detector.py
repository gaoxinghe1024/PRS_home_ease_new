# -*- coding: utf-8 -*-
"""Fall detector: wraps the best combination into two call forms ready for deployment.

  1. detect_offline(t, a, model)  - offline detection of a whole recording (sample data input / history replay)
     Reproduces the evaluation protocol of `experiments/exp03_alg.py` exactly: resample + filter the whole recording -> all sliding windows
     -> per-window probability -> threshold hits -> event aggregation.

  2. StreamingDetector            - real-time stream from a continuously sending sensor
     Keeps a sliding raw buffer of STREAM_BUFFER_S seconds; whenever the time axis advances by one step (0.25 s)
     a 2 s window "ending now" is evaluated; a hit enters the event state machine, and once hits pause for longer than
     the merge gap and the buffer covers 1 s after the alarm, the event is closed -> stored.

**Real-time note**: the filter of the best combination is the zero-phase (non-causal) filtfilt, which cannot be used on a live
stream without delay. This implementation takes an extra FILTER_PAD_S=0.25 s of context on both sides of each window and
evaluates 0.25 s late, so streaming results are almost identical to offline whole-recording filtering, at a fixed 0.25 s detection delay.

Both forms output FallEvent dicts, including the 3 s acceleration slice needed by the report.
"""
import time
import numpy as np

from . import config as C
from . import preprocess as P
from . import windowize as W

PAD = C.FILTER_PAD_S


# ---------------------------------------------------------------------------
# Report slice
# ---------------------------------------------------------------------------
def snippet_from_signal(t_src, ax, ay, az, t_alarm, fs=C.RESAMPLE_FS,
                        pre=C.REPORT_PRE_S, post=C.REPORT_POST_S):
    """Return the **filtered** acceleration curve (pre+post) seconds around the alarm, ready to plot.

    Implementation: resample [t_alarm-(pre+pad), t_alarm+(post+pad)] and apply the same butter5
    zero-phase filter (consistent with detection), then output the middle part [t_alarm-pre, t_alarm+post].
    t is in seconds relative to the alarm; mag = sqrt(ax^2+ay^2+az^2).
    """
    t_src = np.asarray(t_src, np.float64)
    if len(t_src) < 12:
        return {"t": [], "ax": [], "ay": [], "az": [], "mag": []}
    lo = max(t_alarm - pre - PAD, float(t_src[0]))
    hi = min(t_alarm + post + PAD, float(t_src[-1]))
    seg = np.arange(lo, hi + 1e-9, 1.0 / fs)
    if len(seg) < 12:
        seg = np.linspace(lo, hi, 12)
    A = np.stack([np.interp(seg, t_src, np.asarray(v, np.float64))
                  for v in (ax, ay, az)], axis=1)
    A = P.apply_butter5(A, fs)
    keep = (seg >= t_alarm - pre - 1e-9) & (seg <= t_alarm + post + 1e-9)
    rel = seg[keep] - t_alarm
    X, Y, Z = A[keep, 0], A[keep, 1], A[keep, 2]
    Mag = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    r = lambda v: [round(float(x), 5) for x in v]
    return {"t": [round(float(x), 4) for x in rel],
            "ax": r(X), "ay": r(Y), "az": r(Z), "mag": r(Mag)}


# ---------------------------------------------------------------------------
# Offline whole-recording detection
# ---------------------------------------------------------------------------
def detect_offline(t_raw, a_raw, model, threshold=C.ALARM_THRESHOLD, fs=C.RESAMPLE_FS):
    """Detect a whole recording. Returns (events, info).

    events: list[dict] - each element is one fall record ready to store (including the snippet).
    info  : {'n_windows','probs'(list),'t_end'(list),'duration_s','n_hits'}
    """
    if len(a_raw) < 10:
        return [], {"n_windows": 0, "probs": [], "t_end": [], "duration_s": 0.0, "n_hits": 0}

    norm = P.normalize(t_raw, a_raw, fs=fs)
    sig = np.stack([norm["ax"], norm["ay"], norm["az"], norm["mag"]], axis=1)
    wins, t0_idx = W.make_windows(sig, fs)
    if len(wins) == 0:
        return [], {"n_windows": 0, "probs": [], "t_end": [],
                    "duration_s": float(norm["t"][-1] - norm["t"][0]), "n_hits": 0}

    probs = model.predict_proba(wins)
    t_end = np.asarray([norm["t"][i + wins.shape[1] - 1] for i in t0_idx], np.float64)
    hit_mask = probs >= threshold
    hit_t, hit_p = t_end[hit_mask], probs[hit_mask]

    events = []
    for ev in W.merge_hits(hit_t):
        m = (hit_t >= ev["t_win_start"] - 1e-9) & (hit_t <= ev["t_win_end"] + 1e-9)
        sc = hit_p[m]
        events.append(_make_event(
            t_alarm=ev["t_alarm"], t_win_start=ev["t_win_start"], t_win_end=ev["t_win_end"],
            n_windows=ev["n_windows"], scores=sc, threshold=threshold,
            snippet=snippet_from_signal(norm["t"], norm["ax"], norm["ay"], norm["az"],
                                        ev["t_alarm"], fs=fs),
            source="sim",
        ))
    info = {"n_windows": int(len(wins)), "probs": [round(float(x), 5) for x in probs],
            "t_end": [round(float(x), 4) for x in t_end],
            "duration_s": float(norm["t"][-1] - norm["t"][0]), "n_hits": int(hit_mask.sum())}
    return events, info


def _make_event(t_alarm, t_win_start, t_win_end, n_windows, scores, threshold, snippet, source):
    sc = np.asarray(scores, np.float64)
    return {
        "source": source,
        "t_alarm": float(t_alarm),
        "t_win_start": float(t_win_start),
        "t_win_end": float(t_win_end),
        "n_windows": int(n_windows),
        "score": float(sc[0]) if len(sc) else 0.0,        # probability of the earliest hit window (basis of the alarm)
        "score_max": float(sc.max()) if len(sc) else 0.0,
        "score_mean": float(sc.mean()) if len(sc) else 0.0,
        "threshold": float(threshold),
        "snippet": snippet,
    }


# ---------------------------------------------------------------------------
# Streaming detector
# ---------------------------------------------------------------------------
class StreamingDetector:
    """Detector for a continuous acceleration stream (sensor mode).

    Usage:
        det = StreamingDetector(model)
        for t, ax, ay, az in sensor_stream:      # feed sample by sample
            for ev in det.push(t, ax, ay, az):
                save(ev)                          # ev is a closed fall event
        for ev in det.flush():                    # finish when the stream ends
            save(ev)
    """

    def __init__(self, model, threshold=C.ALARM_THRESHOLD, fs=C.RESAMPLE_FS):
        self.model = model
        self.threshold = float(threshold)
        self.fs = float(fs)
        self.win_n = W.n_samples(fs, C.WIN_S)
        self.win_span = (self.win_n - 1) / self.fs   # time span between the first and last sample of a window
        self.step_s = C.STEP_S
        self.merge_gap = C.MERGE_GAP_WIN * C.STEP_S
        self.pad_s = PAD
        # Event closing delay: make sure the data post seconds after the alarm has arrived (for filtering the report slice)
        self.close_delay = max(self.merge_gap, C.REPORT_POST_S)
        # Raw sliding buffer
        self._t = []
        self._a = []
        self.t0 = None
        self.t_now = None
        self.next_eval = None                  # end time of the next window to evaluate
        self.cur = None                        # current open event
        self.pending_alarms = []               # alarms raised immediately on a hit (for live frontend hints)
        self.recent = []                       # the latest (t_end, prob) pairs for the frontend
        self.log = []                          # log of all window evaluations (for the full probability curve in the frontend)
        self.log_base = 0                      # length of the prefix dropped from log (absolute index offset)
        self.last_prob = None
        self.n_eval = 0

    # ------------------------------------------------------------------ input
    def push(self, t, ax, ay, az):
        """Feed one sample (t in seconds, increasing). Returns the events **closed** by this call."""
        t = float(t)
        if self.t0 is None:
            self.t0 = t
            self.next_eval = t + self.win_span   # aligned exactly with the offline sample-index window end points
        if self.t_now is not None and t < self.t_now:
            t = self.t_now                       # never go back in time
        self.t_now = t
        self._t.append(t)
        self._a.append((float(ax), float(ay), float(az)))

        out = []
        guard = 0
        while (self.next_eval is not None and
               self.next_eval + self.pad_s <= t + 1e-9 and guard < 1000):
            guard += 1
            self._eval_window(self.next_eval)
            self.next_eval += self.step_s
        ev = self._maybe_close()
        if ev is not None:
            out.append(ev)
        self._trim()
        return out

    # -------------------------------------------------------------- internals
    def _eval_window(self, t_eval):
        """Evaluate the 2 s window ending at t_eval.

        On the normal streaming path `push` calls this only once `t_eval + pad_s` has arrived, so the
        backward context is always a full pad_s; when `flush` evaluates the tail windows it is shorter than pad_s,
        so `min(t_eval + pad_s, t_now)` is used everywhere (as much as is available), and the last window of the
        stream is not dropped (which would miss a fall).
        """
        if not self._t:
            return
        if self._t[0] > t_eval - self.win_span + 1e-9:
            return                               # the samples for the 2 s window have not all arrived yet
        hi_use = min(t_eval + self.pad_s, self.t_now)
        if hi_use < t_eval - 1e-9:
            return
        # Take as much forward context as available (slightly shorter than pad_s at the start of a stream)
        ctx_lo = max(t_eval - self.win_span - self.pad_s, self._t[0])
        seg_t, seg_a = self._slice(ctx_lo, hi_use)
        if len(seg_t) < 12:
            return
        n_before = max(0, int(round((t_eval - self.win_span - ctx_lo) * self.fs)))
        n_after = int(round((hi_use - t_eval) * self.fs))
        grid = t_eval + np.arange(-(self.win_n - 1 + n_before), n_after + 1) / self.fs
        A = np.stack([np.interp(grid, seg_t, seg_a[:, c]) for c in range(3)], axis=1)
        A = P.apply_butter5(A, self.fs)
        w3 = A[n_before:n_before + self.win_n]
        win = np.concatenate([w3, np.linalg.norm(w3, axis=1)[:, None]], axis=1)
        win = win[None, :, :].astype(np.float32)
        p = float(self.model.predict_proba(win)[0])
        self.last_prob = p
        self.n_eval += 1
        rec = {"t": round(float(t_eval - self.t0), 3), "p": round(p, 4)}
        self.recent.append(rec)
        if len(self.recent) > 400:
            self.recent = self.recent[-400:]
        self.log.append(rec)
        if len(self.log) > 20000:                # long-run protection: keep only the latest 10,000 windows
            self.log = self.log[10000:]
            self.log_base += 10000

        if p >= self.threshold:
            if self.cur is None:
                self.cur = {"t_alarm": t_eval, "last_hit": t_eval,
                            "t_win_start": t_eval - self.win_span, "n": 0,
                            "scores": [], "wall_alarm": time.time()}
                # Alarm immediately on a hit (live output); the record is stored once the event closes
                self.pending_alarms.append({
                    "t_alarm": float(t_eval), "wall_alarm": self.cur["wall_alarm"],
                    "score": p, "threshold": self.threshold,
                    "t_win_start": t_eval - self.win_span,
                })
            self.cur["last_hit"] = t_eval
            self.cur["n"] += 1
            self.cur["scores"].append(p)
            self.cur["t_win_end"] = t_eval

    def _maybe_close(self):
        """Hits paused longer than the closing delay -> close the event."""
        c = self.cur
        if c is None:
            return None
        if self.t_now >= c["last_hit"] + self.close_delay - 1e-9:
            ev = self._finalize(c, truncated=False)
            self.cur = None
            return ev
        return None

    def _finalize(self, c, truncated):
        t_alarm = c["t_alarm"]
        seg_t, seg_a = self._slice(t_alarm - (C.REPORT_PRE_S + self.pad_s),
                                  t_alarm + C.REPORT_POST_S + self.pad_s)
        if len(seg_t) >= 12:
            snip = snippet_from_signal(seg_t, seg_a[:, 0], seg_a[:, 1], seg_a[:, 2],
                                       t_alarm, fs=self.fs)
        else:
            snip = {"t": [], "ax": [], "ay": [], "az": [], "mag": []}
        ev = _make_event(
            t_alarm=t_alarm, t_win_start=c["t_win_start"],
            t_win_end=c.get("t_win_end", c["last_hit"]), n_windows=c["n"],
            scores=c["scores"], threshold=self.threshold, snippet=snip, source="sensor",
        )
        ev["wall_alarm"] = c.get("wall_alarm")
        ev["truncated"] = bool(truncated)
        ev["stream_alarm"] = round(float(t_alarm - self.t0), 3)
        return ev

    def flush(self, complete_tail=True):
        """End of stream: evaluate the tail windows and close and output the open event.

        `complete_tail=True` evaluates the tail windows that end at or before the current time but were not evaluated yet
        (the normal path waits for the pad_s backward context, which leaves the last 0.25 s of windows), so a complete
        replay gives the same result as offline detection; otherwise a fall right at the end of the recording would be missed.
        """
        out = []
        if complete_tail and self.next_eval is not None:
            while self.next_eval <= self.t_now + 1e-9:
                self._eval_window(self.next_eval)
                self.next_eval += self.step_s
        if self.cur is not None:
            out.append(self._finalize(self.cur, truncated=True))
            self.cur = None
        return out

    def pop_alarms(self):
        """Take out and clear the queue of immediate alarms (for live alarms)."""
        out = self.pending_alarms
        self.pending_alarms = []
        return out

    # ------------------------------------------------------------- buffer
    def _slice(self, lo, hi):
        t = np.asarray(self._t, np.float64)
        a = np.asarray(self._a, np.float64)
        m = (t >= lo - 1e-9) & (t <= hi + 1e-9)
        return t[m], a[m]

    def _trim(self):
        if self.t_now is None:
            return
        lo = self.t_now - C.STREAM_BUFFER_S
        i = 0
        for i in range(len(self._t)):
            if self._t[i] >= lo:
                break
        if i > 0:
            self._t = self._t[i:]
            self._a = self._a[i:]

    # ------------------------------------------------------------- state
    def status(self):
        return {
            "t_stream": round(float(self.t_now - self.t0), 3) if self.t_now is not None else 0.0,
            "last_prob": round(float(self.last_prob), 4) if self.last_prob is not None else None,
            "alarming": self.cur is not None,
            "n_eval": self.n_eval,
            "n_samples": len(self._t),
        }
