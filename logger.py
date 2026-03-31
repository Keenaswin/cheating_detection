# ============================================================
# modules/logger.py — CSV event logger + snapshot saver
# ============================================================

import os
import csv
import cv2
import config
from modules.utils import timestamp_str, ensure_dirs


class EventLogger:
    """
    Persists events to a CSV file and optionally saves JPEG snapshots.
    """

    _CSV_HEADERS = ["timestamp", "event", "score", "level", "snapshot"]

    def __init__(self):
        ensure_dirs()
        self._init_csv()

    # ── Public API ──────────────────────────────────────────

    def log(self, engine_result: dict, frame=None):
        """
        Log every alert event in engine_result["new_alerts"].

        Parameters
        ----------
        engine_result : dict from CheatingEngine.evaluate()
        frame         : BGR numpy array (optional, for snapshots)
        """
        if not engine_result["new_alerts"]:
            return

        score     = engine_result["smooth_score"]
        level     = engine_result["level"]
        timestamp = engine_result["timestamp"]

        for event in engine_result["new_alerts"]:
            snapshot_path = ""
            if (config.SNAPSHOT_ON_ALERT and
                    frame is not None and
                    score >= config.MIN_SCORE_FOR_SNAPSHOT):
                snapshot_path = self._save_snapshot(frame, event)

            self._write_row(timestamp, event, score, level, snapshot_path)

    def log_raw(self, timestamp: str, event: str, score: float,
                level: str, frame=None):
        """Direct single-event logging (for testing / external calls)."""
        snapshot_path = ""
        if config.SNAPSHOT_ON_ALERT and frame is not None:
            snapshot_path = self._save_snapshot(frame, event)
        self._write_row(timestamp, event, score, level, snapshot_path)

    # ── Private ─────────────────────────────────────────────

    def _init_csv(self):
        """Create the CSV with headers if it doesn't exist."""
        if not os.path.exists(config.LOG_CSV):
            with open(config.LOG_CSV, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self._CSV_HEADERS)

    def _write_row(self, timestamp, event, score, level, snapshot_path):
        with open(config.LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, event, score, level, snapshot_path])

    def _save_snapshot(self, frame, event: str) -> str:
        safe_event = event.replace(" ", "_").replace("/", "-")
        filename   = f"{timestamp_str()}_{safe_event}.jpg"
        path       = os.path.join(config.SNAPSHOTS_DIR, filename)
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return path
