# ============================================================
# modules/head_pose.py — Head pose estimation via solvePnP + Face Mesh
# ============================================================

import cv2
import numpy as np
import mediapipe as mp
import config
from modules.utils import DurationTracker

# 3-D reference face model points (canonical face, unit scale)
_MODEL_POINTS = np.array([
    [0.0,    0.0,    0.0   ],   # Nose tip          (1)
    [0.0,   -330.0, -65.0  ],   # Chin              (152)
    [-225.0, 170.0, -135.0 ],   # Left eye corner   (263)
    [225.0,  170.0, -135.0 ],   # Right eye corner  (33)
    [-150.0,-150.0, -125.0 ],   # Left mouth corner (287)
    [150.0, -150.0, -125.0 ],   # Right mouth corner(57)
], dtype=np.float64)

# Corresponding Face Mesh landmark indices
_LANDMARK_INDICES = [1, 152, 263, 33, 287, 57]


class HeadPoseEstimator:
    """
    Estimates pitch, yaw, roll from a single camera frame using
    MediaPipe Face Mesh + OpenCV solvePnP.
    """

    def __init__(self):
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._duration = DurationTracker()

    # ── Public API ──────────────────────────────────────────

    def process(self, frame_bgr: np.ndarray) -> dict:
        """
        Returns
        -------
        dict with keys:
            pitch       : float  up/down  (degrees, + = looking down)
            yaw         : float  left/right (degrees, + = looking right)
            roll        : float  tilt     (degrees)
            alert       : bool
            duration    : float
            score_frac  : float  0–1
            rvec / tvec : rotation & translation vectors (for drawing)
        """
        h, w = frame_bgr.shape[:2]
        focal   = w                          # rough focal length estimate
        cx, cy  = w / 2, h / 2
        camera_matrix = np.array([
            [focal, 0,     cx],
            [0,     focal, cy],
            [0,     0,     1 ],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            self._duration.update(False)
            return self._empty_result()

        lm = results.multi_face_landmarks[0].landmark

        # Project 2-D image points from landmark normalised coords
        image_points = np.array([
            [lm[i].x * w, lm[i].y * h] for i in _LANDMARK_INDICES
        ], dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            _MODEL_POINTS, image_points,
            camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            self._duration.update(False)
            return self._empty_result()

        # Convert rotation vector to Euler angles
        rmat, _ = cv2.Rodrigues(rvec)
        pitch, yaw, roll = self._rotation_matrix_to_euler(rmat)

        deviated = (abs(yaw) > config.HEAD_YAW_THRESHOLD or
                    abs(pitch) > config.HEAD_PITCH_THRESHOLD)
        duration  = self._duration.update(deviated)
        alert     = deviated and duration >= config.HEAD_ALERT_SECONDS

        if deviated:
            score_frac = min(duration / (config.HEAD_ALERT_SECONDS * 2), 1.0)
        else:
            score_frac = 0.0

        return {
            "pitch":      float(pitch),
            "yaw":        float(yaw),
            "roll":       float(roll),
            "alert":      alert,
            "duration":   duration,
            "score_frac": score_frac,
            "rvec":       rvec,
            "tvec":       tvec,
        }

    def draw(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Draw pitch/yaw/roll text and a pose axis on the frame."""
        colour = (0, 0, 255) if result["alert"] else (0, 200, 0)
        label  = (f"Yaw:{result['yaw']:.1f}  "
                  f"Pitch:{result['pitch']:.1f}  "
                  f"Roll:{result['roll']:.1f}")
        if result["alert"]:
            label += " !"
        cv2.putText(frame, label, (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

        # Draw 3-D axis if we have pose vectors
        if result.get("rvec") is not None:
            self._draw_axis(frame, result["rvec"], result["tvec"])
        return frame

    def release(self):
        self._face_mesh.close()

    # ── Private ─────────────────────────────────────────────

    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray):
        """Convert 3×3 rotation matrix to (pitch, yaw, roll) in degrees."""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        if not singular:
            x = np.arctan2( R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2( R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0.0
        pitch = np.degrees(x)
        yaw   = np.degrees(y)
        roll  = np.degrees(z)
        return pitch, yaw, roll

    def _draw_axis(self, frame, rvec, tvec):
        h, w = frame.shape[:2]
        focal = w
        camera_matrix = np.array([
            [focal, 0,     w / 2],
            [0,     focal, h / 2],
            [0,     0,     1    ],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        axis_len = 100.0
        axis_3d = np.float32([
            [axis_len, 0, 0],
            [0, axis_len, 0],
            [0, 0, axis_len],
            [0, 0, 0       ],
        ])
        pts, _ = cv2.projectPoints(axis_3d, rvec, tvec,
                                   camera_matrix, dist_coeffs)
        pts = pts.astype(int)
        origin = tuple(pts[3].ravel())
        cv2.line(frame, origin, tuple(pts[0].ravel()), (0,   0,   255), 2)  # X red
        cv2.line(frame, origin, tuple(pts[1].ravel()), (0,   255, 0  ), 2)  # Y green
        cv2.line(frame, origin, tuple(pts[2].ravel()), (255, 0,   0  ), 2)  # Z blue

    @staticmethod
    def _empty_result():
        return {
            "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
            "alert": False, "duration": 0.0,
            "score_frac": 0.0, "rvec": None, "tvec": None,
        }
