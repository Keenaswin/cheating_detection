# ============================================================
# modules/posture_analysis.py — Body posture via MediaPipe Pose
# ============================================================

import cv2
import numpy as np
import mediapipe as mp
import config
from modules.utils import DurationTracker

# Landmark indices (MediaPipe Pose)
_NOSE          = 0
_LEFT_SHOULDER = 11
_RIGHT_SHOULDER= 12


class PostureAnalyser:
    """
    Detects:
    - body leaning out of frame  (shoulders off-centre)
    - head tilted far down       (nose y > threshold)
    """

    def __init__(self):
        self._mp_pose = mp.solutions.pose
        self._pose    = self._mp_pose.Pose(
            model_complexity=0,           # fastest model
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._lean_duration = DurationTracker()
        self._down_duration = DurationTracker()

    # ── Public API ──────────────────────────────────────────

    def process(self, frame_bgr: np.ndarray) -> dict:
        """
        Returns
        -------
        dict with keys:
            leaning         : bool
            looking_down    : bool
            lean_duration   : float
            down_duration   : float
            alert           : bool
            score_frac      : float 0–1
        """
        h, w = frame_bgr.shape[:2]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)

        if not results.pose_landmarks:
            self._lean_duration.update(False)
            self._down_duration.update(False)
            return self._empty_result()

        lm = results.pose_landmarks.landmark

        nose   = lm[_NOSE]
        l_sh   = lm[_LEFT_SHOULDER]
        r_sh   = lm[_RIGHT_SHOULDER]

        # ── Lean detection ──────────────────────────────────
        # Compute centre of shoulders and compare to frame centre (0.5)
        shoulder_mid_x = (l_sh.x + r_sh.x) / 2
        lean = abs(shoulder_mid_x - 0.5) > config.POSTURE_LEAN_THRESHOLD

        # ── Head-down detection ─────────────────────────────
        looking_down = nose.y > config.POSTURE_DOWN_THRESHOLD

        lean_dur = self._lean_duration.update(lean)
        down_dur = self._down_duration.update(looking_down)

        lean_alert = lean and lean_dur >= config.POSTURE_ALERT_SECONDS
        down_alert = looking_down and down_dur >= config.POSTURE_ALERT_SECONDS
        alert      = lean_alert or down_alert

        # Score fraction: proportion of alert thresholds reached
        fracs = []
        if lean:
            fracs.append(min(lean_dur / (config.POSTURE_ALERT_SECONDS * 2), 1.0))
        if looking_down:
            fracs.append(min(down_dur / (config.POSTURE_ALERT_SECONDS * 2), 1.0))
        score_frac = max(fracs) if fracs else 0.0

        return {
            "leaning":      lean,
            "looking_down": looking_down,
            "lean_duration":lean_dur,
            "down_duration":down_dur,
            "alert":        alert,
            "score_frac":   score_frac,
        }

    def draw(self, frame: np.ndarray, result: dict) -> np.ndarray:
        colour = (0, 0, 255) if result["alert"] else (0, 200, 0)
        parts  = []
        if result["leaning"]:
            parts.append(f"LEAN({result['lean_duration']:.1f}s)")
        if result["looking_down"]:
            parts.append(f"DOWN({result['down_duration']:.1f}s)")
        label = "Posture: " + (", ".join(parts) if parts else "OK")
        cv2.putText(frame, label, (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        return frame

    def release(self):
        self._pose.close()

    # ── Private ─────────────────────────────────────────────

    @staticmethod
    def _empty_result():
        return {
            "leaning":       False,
            "looking_down":  False,
            "lean_duration": 0.0,
            "down_duration": 0.0,
            "alert":         False,
            "score_frac":    0.0,
        }
