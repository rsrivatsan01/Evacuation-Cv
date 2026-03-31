# ─────────────────────────────────────────────
# src/detection.py
# YOLOv8 person detection wrapper.
# Loads the fine-tuned model once and runs
# inference on individual frames.
# ─────────────────────────────────────────────

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    YOLO_PERSON_WEIGHTS,
    YOLO_FALLBACK_WEIGHTS,
    DETECTION_CONFIDENCE,
    DETECTION_IOU,
    PERSON_CLASS_ID,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class PersonDetector:
    """
    Wraps YOLOv8 for person detection on video frames.

    Usage:
        detector = PersonDetector()
        detections = detector.detect(frame)
    """

    def __init__(
        self,
        weights_path: str = None,
        confidence: float = DETECTION_CONFIDENCE,
        iou: float = DETECTION_IOU,
    ):
        self.confidence = confidence
        self.iou        = iou

        # ── Load model weights ────────────────────────────
        if weights_path is None:
            weights_path = YOLO_PERSON_WEIGHTS

        if os.path.exists(weights_path):
            logger.info(f"Loading fine-tuned weights: {weights_path}")
            self.model = YOLO(weights_path)
        else:
            logger.warning(
                f"Fine-tuned weights not found at: {weights_path}\n"
                f"Falling back to pretrained: {YOLO_FALLBACK_WEIGHTS}"
            )
            self.model = YOLO(YOLO_FALLBACK_WEIGHTS)

        logger.info(f"PersonDetector ready | conf={confidence} | iou={iou}")

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Run YOLOv8 inference on a single frame.

        IMPORTANT: We pass the original frame directly to YOLO and let
        it handle resizing internally via imgsz=640. This ensures the
        returned bounding box coordinates are always in the original
        frame's coordinate space — no manual scaling needed.

        Args:
            frame : BGR numpy array (original resolution video frame)

        Returns:
            sv.Detections with bounding boxes in original frame coordinates.
        """
        results = self.model.predict(
            source  = frame,       # Original frame — no manual resize
            conf    = self.confidence,
            iou     = self.iou,
            imgsz   = 640,         # YOLO resizes internally, maps coords back
            classes = [PERSON_CLASS_ID],
            verbose = False,
            device  = 'cpu',
        )[0]

        # Convert to supervision Detections
        detections = sv.Detections.from_ultralytics(results)

        # Filter to person class only (safety check)
        if len(detections) > 0:
            mask       = detections.class_id == PERSON_CLASS_ID
            detections = detections[mask]

        return detections

    def detect_batch(self, frames: list) -> list:
        """
        Run inference on a list of frames.

        Args:
            frames : List of BGR numpy arrays (original resolution)

        Returns:
            List of sv.Detections objects
        """
        results_list = self.model.predict(
            source  = frames,
            conf    = self.confidence,
            iou     = self.iou,
            imgsz   = 640,
            classes = [PERSON_CLASS_ID],
            verbose = False,
            device  = 'cpu',
        )

        detections_list = []
        for results in results_list:
            detections = sv.Detections.from_ultralytics(results)
            if len(detections) > 0:
                mask       = detections.class_id == PERSON_CLASS_ID
                detections = detections[mask]
            detections_list.append(detections)

        return detections_list

    @property
    def model_info(self) -> dict:
        return {
            "confidence" : self.confidence,
            "iou"        : self.iou,
            "device"     : "cpu",
        }