# ============================================================
# modules/utils.py — Shared helper utilities
# ============================================================

import os
import time
import cv2
import numpy as np
import config


def ensure_dirs():
    """Create all required directories if they do not exist."""
    for d in (config.LOGS_DIR, config.SNAPSHOTS_DIR, config.MODELS_DIR):
        os.makedirs(d, exist_ok=True)


def resize_frame(frame, width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT):
    """Resize frame to a fixed resolution."""
    return cv2.resize(frame, (width, height))


def draw_risk_bar(frame, score: float):
    """
    Overlay a colour-coded risk bar and score text on the frame.
    score : 0–100
    """
    h, w = frame.shape[:2]
    bar_x, bar_y = 10, h - 30
    bar_w, bar_h = w - 20, 20

    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (50, 50, 50), -1)

    # Filled portion
    filled = int(bar_w * score / 100)
    if score < config.RISK_MEDIUM_THRESHOLD:
        colour = (0, 200, 0)       # green
    elif score < config.RISK_HIGH_THRESHOLD:
        colour = (0, 165, 255)     # orange
    else:
        colour = (0, 0, 255)       # red

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                  colour, -1)

    # Label
    label = f"Risk: {score:.0f}/100"
    cv2.putText(frame, label, (bar_x + 4, bar_y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def draw_status_text(frame, lines: list):
    """
    Draw a list of (text, colour) tuples as status lines in the top-left.
    """
    y = 20
    for text, colour in lines:
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
        y += 18
    return frame


def timestamp_str():
    """Return a filesystem-safe timestamp string."""
    return time.strftime("%Y%m%d_%H%M%S")


def readable_timestamp():
    """Return a human-readable timestamp string."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def clamp(value, lo=0, hi=100):
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


class RateTimer:
    """
    Simple helper: returns True once per `interval` seconds.
    Useful for cooldown / sustained-alert logic.
    """
    def __init__(self, interval: float):
        self.interval = interval
        self._last = 0.0

    def ready(self) -> bool:
        now = time.time()
        if now - self._last >= self.interval:
            self._last = now
            return True
        return False

    def reset(self):
        self._last = 0.0


class DurationTracker:
    """
    Tracks how long a condition has been continuously True.
    Returns the elapsed seconds; resets when condition becomes False.
    """
    def __init__(self):
        self._start = None

    def update(self, condition: bool) -> float:
        if condition:
            if self._start is None:
                self._start = time.time()
            return time.time() - self._start
        else:
            self._start = None
            return 0.0
