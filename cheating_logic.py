# ============================================================
# modules/cheating_logic.py — Weighted scoring + cooldown engine
# ============================================================

import time
from collections import deque
import config
from modules.utils import clamp, readable_timestamp


class CheatingEngine:
    """
    Combines signals from all detectors into a single 0–100 risk score.

    Weights (must sum to 100):
        GAZE    → config.GAZE_WEIGHT    (25)
        HEAD    → config.HEAD_WEIGHT    (25)
        OBJECT  → config.OBJECT_WEIGHT  (30)
        POSTURE → config.POSTURE_WEIGHT (20)

    Features:
    - Smoothing over a rolling window of scores
    - Cooldown per event type to avoid repeated alerts
    """

    # Event labels for each signal
    _EVENT_LABELS = {
        "gaze":    "Suspicious Gaze",
        "head":    "Head Turned",
        "object":  "Object Detected",
        "posture": "Abnormal Posture",
    }

    def __init__(self):
        self._score_history: deque = deque(
            maxlen=config.SMOOTHING_WINDOW
        )
        # Cooldown tracking: last alert time per event type
        self._last_alert: dict[str, float] = {k: 0.0 for k in self._EVENT_LABELS}

    # ── Public API ──────────────────────────────────────────

    def evaluate(
        self,
        gaze_result:    dict,
        head_result:    dict,
        object_result:  dict,
        posture_result: dict,
    ) -> dict:
        """
        Parameters
        ----------
        *_result : output dicts from each detector module

        Returns
        -------
        dict with keys:
            raw_score     : float  (current frame, unsmoothed)
            smooth_score  : float  (smoothed over window)
            level         : str    "LOW" | "MEDIUM" | "HIGH"
            events        : list[str]  active event labels
            new_alerts    : list[str]  events that just fired (post-cooldown)
        """
        now = time.time()

        # ── Per-signal weighted scores ──────────────────────
        g_score = gaze_result["score_frac"]    * config.GAZE_WEIGHT
        h_score = head_result["score_frac"]    * config.HEAD_WEIGHT
        o_score = object_result["score_frac"]  * config.OBJECT_WEIGHT
        p_score = posture_result["score_frac"] * config.POSTURE_WEIGHT

        raw = clamp(g_score + h_score + o_score + p_score)
        self._score_history.append(raw)
        smooth = clamp(sum(self._score_history) / len(self._score_history))

        # ── Active events ──────────────────────────────────
        alert_flags = {
            "gaze":    gaze_result["alert"],
            "head":    head_result["alert"],
            "object":  object_result["alert"],
            "posture": posture_result["alert"],
        }
        events = [self._EVENT_LABELS[k] for k, v in alert_flags.items() if v]

        # ── Cooldown-filtered new alerts ───────────────────
        new_alerts = []
        for k, fired in alert_flags.items():
            if fired:
                if now - self._last_alert[k] >= config.COOLDOWN_SECONDS:
                    self._last_alert[k] = now
                    new_alerts.append(self._EVENT_LABELS[k])

        # ── Risk level ─────────────────────────────────────
        if smooth >= config.RISK_HIGH_THRESHOLD:
            level = "HIGH"
        elif smooth >= config.RISK_MEDIUM_THRESHOLD:
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "raw_score":   round(raw,    1),
            "smooth_score":round(smooth, 1),
            "level":       level,
            "events":      events,
            "new_alerts":  new_alerts,
            "timestamp":   readable_timestamp(),
        }

    def reset(self):
        """Clear history and cooldowns."""
        self._score_history.clear()
        self._last_alert = {k: 0.0 for k in self._EVENT_LABELS}
