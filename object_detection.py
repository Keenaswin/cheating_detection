# ============================================================
# modules/object_detection.py — YOLOv5 phone / book detection
# ============================================================

import cv2
import numpy as np
import torch
import config


class ObjectDetector:
    """
    Wraps YOLOv5 (loaded via torch.hub) to detect suspicious objects.
    Auto-downloads the model on first run.
    """

    def __init__(self):
        self._model = None
        self._load_model()

    # ── Public API ──────────────────────────────────────────

    def process(self, frame_bgr: np.ndarray) -> dict:
        """
        Returns
        -------
        dict with keys:
            detections  : list of dicts  {label, conf, box:[x1,y1,x2,y2]}
            alert       : bool  any target class found
            score_frac  : float 0–1
        """
        if self._model is None:
            return self._empty_result()

        # YOLOv5 expects RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._model(rgb, size=320)      # smaller size = faster

        detections = []
        max_conf   = 0.0

        for *box, conf, cls_id in results.xyxy[0].tolist():
            label = self._model.names[int(cls_id)]
            if (label in config.YOLO_TARGET_CLASSES and
                    conf >= config.YOLO_CONF_THRESHOLD):
                detections.append({
                    "label": label,
                    "conf":  round(float(conf), 2),
                    "box":   [int(b) for b in box],
                })
                max_conf = max(max_conf, conf)

        alert      = len(detections) > 0
        score_frac = min(max_conf, 1.0) if alert else 0.0

        return {
            "detections": detections,
            "alert":      alert,
            "score_frac": score_frac,
        }

    def draw(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Draw bounding boxes + labels for each detection."""
        for det in result["detections"]:
            x1, y1, x2, y2 = det["box"]
            label = f"{det['label']} {det['conf']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if result["alert"]:
            cv2.putText(frame, "OBJECT DETECTED!", (10, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        return frame

    # ── Private ─────────────────────────────────────────────

    def _load_model(self):
        print("[ObjectDetector] Loading YOLOv5 model …")
        try:
            # torch.hub will cache the model in ~/.cache/torch/hub
            self._model = torch.hub.load(
                "ultralytics/yolov5",
                config.YOLO_MODEL_NAME,
                pretrained=True,
                verbose=False,
            )
            self._model.eval()
            # Run once on a dummy image to warm up
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self._model(dummy, size=320)
            print("[ObjectDetector] Model ready.")
        except Exception as exc:
            print(f"[ObjectDetector] WARNING: Could not load model – {exc}")
            print("[ObjectDetector] Object detection will be skipped.")
            self._model = None

    @staticmethod
    def _empty_result():
        return {"detections": [], "alert": False, "score_frac": 0.0}
