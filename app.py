# ============================================================
# dashboard/app.py — Flask dashboard for the cheating detection system
# ============================================================

import sys
import os
import time

# Make sure the project root is on sys.path so we can import shared_state
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from flask import Flask, Response, jsonify, render_template, send_from_directory
import shared_state
import config


def create_app(frame_queue=None):
    """
    Factory function — called by main.py (which passes the frame_queue).
    Can also be called standalone: python dashboard/app.py
    """
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    static_dir   = os.path.join(os.path.dirname(__file__), "static")

    app = Flask(__name__,
                template_folder=template_dir,
                static_folder=static_dir)

    # ── Routes ──────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/state")
    def api_state():
        """JSON endpoint — polled by JavaScript every second."""
        state = shared_state.get_state()
        return jsonify(state)

    @app.route("/video_feed")
    def video_feed():
        """MJPEG stream endpoint."""
        return Response(
            _generate_mjpeg(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/snapshots/<path:filename>")
    def snapshot(filename):
        return send_from_directory(config.SNAPSHOTS_DIR, filename)

    # ── MJPEG generator ─────────────────────────────────────

    def _generate_mjpeg():
        """Yields JPEG frames as a multipart MJPEG stream."""
        import cv2
        import numpy as np

        blank = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        cv2.putText(blank, "Waiting for camera...", (80, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        _, blank_jpeg = cv2.imencode(".jpg", blank)
        blank_bytes   = bytes(blank_jpeg)

        while True:
            jpeg = shared_state.get_frame_jpeg()
            if jpeg is None:
                jpeg = blank_bytes

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )
            time.sleep(1 / config.TARGET_FPS)

    return app


# ── Standalone mode ─────────────────────────────────────────

if __name__ == "__main__":
    print("[Dashboard] Running in standalone mode (no live camera).")
    print(f"[Dashboard] Open → http://localhost:{config.DASHBOARD_PORT}")
    app = create_app()
    app.run(
        host=config.DASHBOARD_HOST,
        port=config.DASHBOARD_PORT,
        debug=False,
        use_reloader=False,
    )
