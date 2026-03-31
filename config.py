# ============================================================
# config.py — Central configuration for Cheating Detection System
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR        = os.path.join(BASE_DIR, "logs")
SNAPSHOTS_DIR   = os.path.join(BASE_DIR, "snapshots")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
LOG_CSV         = os.path.join(LOGS_DIR, "events.csv")

# ── Camera ───────────────────────────────────────────────────
CAMERA_INDEX    = 0          # 0 = default webcam
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480
TARGET_FPS      = 20

# ── Gaze Tracking ────────────────────────────────────────────
GAZE_DEVIATION_THRESHOLD  = 0.15   # normalised iris offset to count as "off-centre"
GAZE_ALERT_SECONDS        = 2.0    # sustained deviation before flagging
GAZE_WEIGHT               = 25     # max contribution to risk score

# ── Head Pose ────────────────────────────────────────────────
HEAD_YAW_THRESHOLD        = 25     # degrees left / right
HEAD_PITCH_THRESHOLD      = 20     # degrees up / down
HEAD_ALERT_SECONDS        = 1.5
HEAD_WEIGHT               = 25

# ── Object Detection ─────────────────────────────────────────
YOLO_MODEL_NAME           = "yolov5s"   # will be downloaded via torch.hub
YOLO_CONF_THRESHOLD       = 0.40
YOLO_TARGET_CLASSES       = ["cell phone", "book"]
OBJECT_WEIGHT             = 30

# ── Posture Analysis ─────────────────────────────────────────
POSTURE_LEAN_THRESHOLD    = 0.25   # fraction of frame width for shoulder offset
POSTURE_DOWN_THRESHOLD    = 0.65   # nose y-fraction of frame height
POSTURE_ALERT_SECONDS     = 2.0
POSTURE_WEIGHT            = 20

# ── Cheating Logic ───────────────────────────────────────────
RISK_HIGH_THRESHOLD       = 70     # score >= this → HIGH alert
RISK_MEDIUM_THRESHOLD     = 40     # score >= this → MEDIUM alert
COOLDOWN_SECONDS          = 5      # minimum gap between same-event alerts
SMOOTHING_WINDOW          = 8      # frames used for score smoothing

# ── Logging ──────────────────────────────────────────────────
SNAPSHOT_ON_ALERT         = True
MIN_SCORE_FOR_SNAPSHOT    = 60

# ── Dashboard ────────────────────────────────────────────────
DASHBOARD_HOST            = "0.0.0.0"
DASHBOARD_PORT            = 5000
STREAM_JPEG_QUALITY       = 70     # 1-95, lower = faster
