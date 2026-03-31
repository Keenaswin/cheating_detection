# ============================================================
# shared_state.py — Thread-safe global state for dashboard ↔ detector
# ============================================================

import threading
import cv2
import numpy as np
import config

_lock = threading.Lock()

# Default state
_state = {
    "score":     0.0,
    "level":     "LOW",
    "events":    [],
    "new_alerts":[],
    "timestamp": "--",
    "history":   [],      # list of recent engine_results (last 50)
}

_latest_frame: np.ndarray = None   # BGR frame


def update(engine_result: dict, frame: np.ndarray):
    """Called from the detection thread."""
    global _latest_frame
    with _lock:
        _state["score"]      = engine_result["smooth_score"]
        _state["level"]      = engine_result["level"]
        _state["events"]     = engine_result["events"]
        _state["new_alerts"] = engine_result["new_alerts"]
        _state["timestamp"]  = engine_result["timestamp"]

        if engine_result["new_alerts"]:
            _state["history"].append({
                "timestamp": engine_result["timestamp"],
                "events":    engine_result["new_alerts"],
                "score":     engine_result["smooth_score"],
                "level":     engine_result["level"],
            })
            # Keep only last 50 entries
            if len(_state["history"]) > 50:
                _state["history"] = _state["history"][-50:]

        _latest_frame = frame.copy()


def get_state() -> dict:
    """Called from the dashboard thread — returns a safe copy."""
    with _lock:
        return {
            "score":     _state["score"],
            "level":     _state["level"],
            "events":    list(_state["events"]),
            "new_alerts":list(_state["new_alerts"]),
            "timestamp": _state["timestamp"],
            "history":   list(_state["history"]),
        }


def get_frame_jpeg() -> bytes | None:
    """Return the latest frame encoded as JPEG bytes."""
    global _latest_frame
    with _lock:
        if _latest_frame is None:
            return None
        ret, buf = cv2.imencode(
            ".jpg", _latest_frame,
            [cv2.IMWRITE_JPEG_QUALITY, config.STREAM_JPEG_QUALITY]
        )
        return bytes(buf) if ret else None
