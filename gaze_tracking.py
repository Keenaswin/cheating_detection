# ============================================================
# modules/gaze_tracking.py — Iris / gaze direction via MediaPipe Face Mesh
# ============================================================

import cv2
import numpy as np
import mediapipe as mp
import config
from modules.utils import DurationTracker

# MediaPipe iris landmark indices (left eye, right eye)
# These are the refined iris landmarks available when refine_landmarks=True
_LEFT_IRIS   = [474, 475, 476, 477]   # centre + cardinal pts
_RIGHT_IRIS  = [469, 470, 471, 472]

# Eye-corner landmarks used to compute the eye bounding box
_LEFT_EYE_CORNERS  = [33, 133]   # inner / outer corner
_RIGHT_EYE_CORNERS = [362, 263]


class GazeTracker:
    """
    Estimates gaze direction (LEFT / RIGHT / UP / CENTER) from MediaPipe
    iris landmarks and raises an alert when the deviation persists.
    """

    def __init__(self):
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,      # required for iris landmarks
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._duration = DurationTracker()
        self._last_direction = "CENTER"

    # ── Public API ──────────────────────────────────────────

    def process(self, frame_bgr: np.ndarray) -> dict:
        """
        Parameters
        ----------
        frame_bgr : BGR OpenCV frame

        Returns
        -------
        dict with keys:
            direction   : str  "LEFT" | "RIGHT" | "UP" | "CENTER" | "NO_FACE"
            offset_x    : float  normalised horizontal iris offset  (-1 … +1)
            offset_y    : float  normalised vertical   iris offset  (-1 … +1)
            alert       : bool   True when deviation sustained > threshold
            duration    : float  seconds of current deviation
            score_frac  : float  0–1 contribution to risk score
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self._duration.update(False)
            return self._empty_result()

        lm = results.multi_face_landmarks[0].landmark

        # ── Compute iris centre (average of 4 iris points) ─────────────
        def iris_centre(indices):
            pts = np.array([[lm[i].x, lm[i].y] for i in indices])
            return pts.mean(axis=0)

        left_iris  = iris_centre(_LEFT_IRIS)
        right_iris = iris_centre(_RIGHT_IRIS)

        # ── Compute eye bounding box ────────────────────────────────────
        def eye_box(corner_indices):
            pts = np.array([[lm[i].x, lm[i].y] for i in corner_indices])
            return pts[0], pts[1]   # left-corner, right-corner

        lc_left,  rc_left  = eye_box(_LEFT_EYE_CORNERS)
        lc_right, rc_right = eye_box(_RIGHT_EYE_CORNERS)

        # Offset = (iris_x - eye_midpoint_x) / half_eye_width
        def offset(iris, lc, rc):
            mid_x  = (lc[0] + rc[0]) / 2
            half_w = abs(rc[0] - lc[0]) / 2 + 1e-6
            ox = (iris[0] - mid_x) / half_w
            mid_y  = (lc[1] + rc[1]) / 2
            half_h = abs(rc[1] - lc[1]) / 2 + 1e-6
            oy = (iris[1] - mid_y) / half_h
            return ox, oy

        ox_l, oy_l = offset(left_iris,  lc_left,  rc_left)
        ox_r, oy_r = offset(right_iris, lc_right, rc_right)

        avg_ox = (ox_l + ox_r) / 2
        avg_oy = (oy_l + oy_r) / 2

        # ── Classify direction ─────────────────────────────────────────
        thr = config.GAZE_DEVIATION_THRESHOLD
        if avg_ox < -thr:
            direction = "LEFT"
        elif avg_ox > thr:
            direction = "RIGHT"
        elif avg_oy < -thr:
            direction = "UP"
        else:
            direction = "CENTER"

        self._last_direction = direction
        deviated = direction != "CENTER"
        duration  = self._duration.update(deviated)
        alert     = deviated and duration >= config.GAZE_ALERT_SECONDS

        # Score fraction: scale by how long they've been looking away (cap at 1)
        if deviated:
            score_frac = min(duration / (config.GAZE_ALERT_SECONDS * 2), 1.0)
        else:
            score_frac = 0.0

        return {
            "direction":  direction,
            "offset_x":   float(avg_ox),
            "offset_y":   float(avg_oy),
            "alert":      alert,
            "duration":   duration,
            "score_frac": score_frac,
        }

    def draw(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Annotate frame with gaze info."""
        d = result["direction"]
        colour = (0, 255, 0) if d == "CENTER" else (0, 0, 255)
        label  = f"Gaze: {d}"
        if result["alert"]:
            label += f" ({result['duration']:.1f}s) !"
        cv2.putText(frame, label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
        return frame

    def release(self):
        self._face_mesh.close()

    # ── Private ─────────────────────────────────────────────

    @staticmethod
    def _empty_result():
        return {
            "direction":  "NO_FACE",
            "offset_x":   0.0,
            "offset_y":   0.0,
            "alert":      False,
            "duration":   0.0,
            "score_frac": 0.0,
        }
