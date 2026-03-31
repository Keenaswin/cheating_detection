# ============================================================
# main.py — Entry point: captures webcam, runs all detectors,
#            publishes shared state for the Flask dashboard.
# ============================================================

import cv2
import time
import threading
import queue
import numpy as np

import config
from modules.utils      import ensure_dirs, resize_frame, draw_risk_bar, draw_status_text
from modules.gaze_tracking    import GazeTracker
from modules.head_pose        import HeadPoseEstimator
from modules.object_detection import ObjectDetector
from modules.posture_analysis import PostureAnalyser
from modules.cheating_logic   import CheatingEngine
from modules.logger           import EventLogger

# ── Shared state (thread-safe) ──────────────────────────────
# The Flask dashboard reads from this dict.
import shared_state

# ── Frame queue for the dashboard MJPEG stream ──────────────
frame_queue: queue.Queue = queue.Queue(maxsize=2)


# ════════════════════════════════════════════════════════════
# Core detection loop
# ════════════════════════════════════════════════════════════

def run_detection():
    """Main detection loop — runs in its own thread."""
    ensure_dirs()

    # ── Initialise modules ──────────────────────────────────
    print("[Main] Initialising detectors …")
    gaze_tracker  = GazeTracker()
    head_estimator= HeadPoseEstimator()
    object_detector = ObjectDetector()   # may print a download message
    posture_analyser= PostureAnalyser()
    cheating_engine = CheatingEngine()
    logger          = EventLogger()

    # ── Open camera ─────────────────────────────────────────
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[Main] ERROR: Cannot open camera (index {config.CAMERA_INDEX}).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)

    print("[Main] Camera open. Starting detection loop … (press Q to quit)")

    # YOLO runs slower — we only run it every N frames
    YOLO_EVERY_N = 5
    frame_count  = 0
    last_obj_result = {"detections": [], "alert": False, "score_frac": 0.0}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Main] WARNING: Frame capture failed, retrying …")
            time.sleep(0.05)
            continue

        frame = resize_frame(frame)
        frame_count += 1

        # ── Run detectors ───────────────────────────────────
        gaze_result    = gaze_tracker.process(frame)
        head_result    = head_estimator.process(frame)
        posture_result = posture_analyser.process(frame)

        if frame_count % YOLO_EVERY_N == 0:
            last_obj_result = object_detector.process(frame)
        object_result = last_obj_result

        # ── Cheating engine ─────────────────────────────────
        engine_result = cheating_engine.evaluate(
            gaze_result, head_result, object_result, posture_result
        )

        # ── Log events ──────────────────────────────────────
        logger.log(engine_result, frame)

        # ── Annotate frame ──────────────────────────────────
        annotated = frame.copy()
        gaze_tracker.draw(annotated, gaze_result)
        head_estimator.draw(annotated, head_result)
        object_detector.draw(annotated, object_result)
        posture_analyser.draw(annotated, posture_result)
        draw_risk_bar(annotated, engine_result["smooth_score"])

        # Draw alert banner
        level  = engine_result["level"]
        colour = {"LOW": (0,180,0), "MEDIUM": (0,140,255), "HIGH": (0,0,255)}[level]
        cv2.putText(annotated, f"Risk Level: {level}",
                    (10, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

        # ── Update shared state for dashboard ───────────────
        shared_state.update(engine_result, annotated)

        # ── Push to frame queue (non-blocking) ──────────────
        try:
            frame_queue.put_nowait(annotated)
        except queue.Full:
            pass  # dashboard consumer is slower than producer — drop frame

        # ── Local preview window ─────────────────────────────
        cv2.imshow("Cheating Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Cleanup ─────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    gaze_tracker.release()
    head_estimator.release()
    posture_analyser.release()
    print("[Main] Detection stopped.")


# ════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Start Flask dashboard in a background thread
    import importlib, sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dashboard"))

    try:
        from dashboard.app import create_app
        app = create_app(frame_queue)
        dash_thread = threading.Thread(
            target=lambda: app.run(
                host=config.DASHBOARD_HOST,
                port=config.DASHBOARD_PORT,
                debug=False,
                use_reloader=False,
            ),
            daemon=True,
        )
        dash_thread.start()
        print(f"[Main] Dashboard → http://localhost:{config.DASHBOARD_PORT}")
    except Exception as e:
        print(f"[Main] Dashboard could not start: {e}")

    # Run detection on the main thread
    run_detection()
