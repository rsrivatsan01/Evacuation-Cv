# ─────────────────────────────────────────────
# src/pipeline.py
# Main background processing pipeline.
# Runs in a separate thread, processes video
# frames and exposes results to Flask app.
# ─────────────────────────────────────────────

import cv2
import time
import threading
import numpy as np
import os
import sys
from collections import deque
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    SAMPLE_VIDEO_PATH, REALTIME_SOURCE,
    VIDEO_STREAM_FPS, FLOORMAP_PATH,
)
from src.detection    import PersonDetector
from src.tracking     import PersonTracker
from src.zones        import ZoneManager, STATUS_COLORS
from src.features     import FeatureExtractor
from src.congestion   import CongestionInference
from src.pathplanning import VenueGraph, PersonRouter
from utils.visualizer import (
    build_annotators, draw_detections, draw_stats_panel,
    draw_alert_banner, draw_person_routing,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# Try importing path drawing — may not exist yet
try:
    from utils.visualizer import draw_path_on_frame, draw_path_on_floormap
    HAS_PATH_DRAW = True
except ImportError:
    HAS_PATH_DRAW = False


class EvacuationPipeline:
    """
    Background pipeline that processes video frames
    and exposes results for the Flask dashboard.

    Thread-safe via locks on shared state.
    """

    def __init__(self, video_source=None):
        self.video_source  = video_source or SAMPLE_VIDEO_PATH
        self._thread       = None
        self._running      = False
        self._lock         = threading.Lock()

        # ── Shared state (read by Flask routes) ──
        self._latest_frame    = None   # JPEG bytes of annotated frame
        self._latest_floormap = None   # JPEG bytes of floormap
        self._latest_status   = {}     # Dict for /status API
        self._frame_count     = 0
        self._fps             = 0.0
        self._initialized     = False

        # ── Components (initialized in thread) ───
        self.detector     = None
        self.tracker      = None
        self.zone_manager = None
        self.extractor    = None
        self.predictor    = None
        self.graph        = None
        self.router       = None
        self.box_ann      = None
        self.label_ann    = None

        self._fps_history  = deque(maxlen=30)
        self._path_result  = None

    # ── Lifecycle ─────────────────────────────────

    def start(self):
        """Start pipeline in background thread."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Pipeline started | source: {self.video_source}")

    def stop(self):
        """Stop the pipeline thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        logger.info("Pipeline stopped")

    def reload_zones(self):
        """Reload zones from JSON — called after dashboard edit."""
        if self.zone_manager:
            self.zone_manager.load()
            if self.extractor:
                self.extractor.zone_manager = self.zone_manager
            if self.predictor:
                self.predictor.zone_manager = self.zone_manager
                self.predictor._init_buffers()
            logger.info("Zones reloaded")

    # ── Initialization ─────────────────────────────

    def _init_components(self, frame_shape):
        """Initialize all ML components. Called once on first frame."""
        h, w = frame_shape[:2]

        self.zone_manager = ZoneManager()
        self.zone_manager.load()

        self.detector  = PersonDetector()
        self.tracker   = PersonTracker()
        self.extractor = FeatureExtractor(
            self.zone_manager, (h, w), log_csv=True
        )
        self.predictor = CongestionInference(self.zone_manager)
        self.predictor.load()

        self.graph  = VenueGraph()
        self.graph.load()

        self.router = PersonRouter(self.zone_manager, (h, w))

        self.box_ann, self.label_ann = build_annotators()
        self._initialized = True
        logger.info(f"Pipeline components initialized | frame={w}x{h}")

    # ── Main Loop ──────────────────────────────────

    def _run(self):
        """Main processing loop — runs in background thread."""
        while self._running:
            cap = self._open_source()
            if cap is None:
                time.sleep(2)
                continue

            try:
                self._process_video(cap)
            finally:
                cap.release()

            # If pre-recorded video ended — loop
            if self._running and not self._is_live_source():
                logger.info("Video ended — looping")
                time.sleep(0.1)
            else:
                break

    def _open_source(self) -> Optional[cv2.VideoCapture]:
        """Open video source — file or live stream."""
        src = self.video_source
        if isinstance(src, str) and not os.path.exists(src):
            logger.error(f"Video file not found: {src}")
            return None

        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            logger.error(f"Cannot open video source: {src}")
            return None

        return cap

    def _is_live_source(self) -> bool:
        """Returns True if source is webcam or RTSP."""
        src = self.video_source
        if isinstance(src, int):
            return True
        if isinstance(src, str) and src.startswith("rtsp://"):
            return True
        return False

    def _process_video(self, cap: cv2.VideoCapture):
        """Process frames from an open VideoCapture."""
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        frame_idx  = 0
        skip       = 2   # Process every 2nd frame

        # Load floormap
        floormap = self._load_floormap()

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % skip != 0:
                continue

            # Init components on first valid frame
            if not self._initialized:
                self._init_components(frame.shape)

            t0 = time.time()

            # ── Core pipeline ──────────────────
            detections  = self.detector.detect(frame)
            tracked     = self.tracker.update(detections)
            self.zone_manager.update_counts(tracked)
            features    = self.extractor.update(tracked, frame_idx)
            predictions = self.predictor.predict(features)
            routing     = self.router.assign(tracked, predictions)

            # ── Log features to CSV now that ML predictions are ready ──
            self.extractor.log_features(features, predictions)

            # ── Path planning ──────────────────
            if self.graph.loaded:
                self.graph.update_weights(predictions)
                self._path_result = self.graph.find_safest_path("center")

            # ── FPS ────────────────────────────
            elapsed = time.time() - t0
            fps     = 1.0 / max(elapsed, 1e-6)
            self._fps_history.append(fps)
            avg_fps = sum(self._fps_history) / len(self._fps_history)

            # ── Annotate frame ─────────────────
            annotated = self._annotate_frame(
                frame, tracked, routing,
                features, predictions, avg_fps, frame_idx
            )

            # ── Update floormap ────────────────
            if floormap is not None and HAS_PATH_DRAW and self._path_result:
                try:
                    fm_copy = floormap.copy()
                    fm_copy = draw_path_on_floormap(
                        fm_copy, self._path_result,
                        self.graph, predictions, self.zone_manager
                    )
                    self._update_floormap(fm_copy)
                except Exception:
                    self._update_floormap(floormap)
            elif floormap is not None:
                self._update_floormap(floormap)

            # ── Build status dict ──────────────
            status = self._build_status(
                features, predictions, avg_fps,
                frame_idx, tracked
            )

            # ── Write to shared state ──────────
            with self._lock:
                self._latest_frame  = self._encode_frame(annotated)
                self._latest_status = status
                self._frame_count   = frame_idx
                self._fps           = avg_fps

            # ── Throttle to target FPS ─────────
            target_delay = 1.0 / VIDEO_STREAM_FPS
            sleep_time   = target_delay - (time.time() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _annotate_frame(
        self, frame, tracked, routing,
        features, predictions, avg_fps, frame_idx
    ) -> np.ndarray:
        """Build fully annotated frame."""
        annotated = frame.copy()

        # Draw zones
        for zone in self.zone_manager.zones:
            x1, y1, x2, y2 = zone.bbox
            
            # Predict status
            pred_info = predictions.get(zone.id, {})
            pred_status = pred_info.get("status", zone.current_status)
            conf = pred_info.get("confidence")
            conf_str = f" {conf:.0%}" if conf else ""
            
            color = STATUS_COLORS.get(pred_status, (0, 200, 0))
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{zone.name} {zone.current_count}/{zone.capacity} {pred_status}{conf_str}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            
            label_y = max(y1, th + 8)
            cv2.rectangle(annotated, (x1, label_y-th-8), (x1+tw+6, label_y), color, -1)
            cv2.putText(annotated, label, (x1+3, label_y-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)

        # Draw detections
        annotated = draw_detections(annotated, tracked, self.box_ann, self.label_ann)

        # Draw per-person routing
        if routing:
            annotated = draw_person_routing(annotated, tracked, routing)

        # Stats panel
        annotated = draw_stats_panel(
            annotated, self.tracker.active_count,
            avg_fps, frame_idx, self.tracker.total_count
        )

        # Alert banner
        critical = [z.name for z in self.zone_manager.zones
                    if predictions.get(z.id, {}).get("status", z.current_status) == "CRITICAL"]
        if critical:
            annotated = draw_alert_banner(
                annotated,
                f"EVACUATION ALERT — Congestion at: {', '.join(critical)}"
            )

        return annotated

    def _build_status(self, features, predictions, fps, frame_idx, tracked) -> dict:
        """Build status dict for /status API."""
        zones_status = []
        for zone in self.zone_manager.zones:
            pred   = predictions.get(zone.id, {})
            status = pred.get("status", zone.current_status)
            probs  = pred.get("probs")
            zones_status.append({
                "id"         : zone.id,
                "name"       : zone.name,
                "count"      : zone.current_count,
                "capacity"   : zone.capacity,
                "status"     : status,
                "confidence" : pred.get("confidence"),
                "probs"      : probs,
            })

        zone_features = features.get("zones", [])

        return {
            "frame_id"       : frame_idx,
            "fps"            : round(fps, 1),
            "active_persons" : self.tracker.active_count,
            "total_persons"  : self.tracker.total_count,
            "global_count"   : features.get("global_count", 0),
            "global_speed"   : round(features.get("global_speed", 0), 2),
            "global_density" : round(features.get("global_density", 0), 4),
            "zones"          : zones_status,
            "zone_features"  : zone_features,
            "any_critical"   : self.zone_manager.any_critical(),
            "path"           : self._path_result,
            "lstm_ready"     : self.predictor.is_ready if self.predictor else False,
        }

    # ── Helpers ────────────────────────────────────

    def _load_floormap(self) -> Optional[np.ndarray]:
        """Load floor map image."""
        if os.path.exists(FLOORMAP_PATH):
            img = cv2.imread(FLOORMAP_PATH)
            if img is not None:
                return img
        # Return blank grey image if no floor map
        return np.ones((400, 700, 3), dtype=np.uint8) * 40

    def _update_floormap(self, img: np.ndarray):
        """Encode and store floor map."""
        with self._lock:
            self._latest_floormap = self._encode_frame(img)

    @staticmethod
    def _encode_frame(frame: np.ndarray) -> bytes:
        """Encode numpy frame to JPEG bytes."""
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()

    # ── Public accessors (thread-safe) ─────────────

    def get_frame(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_frame

    def get_floormap(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_floormap

    def get_status(self) -> dict:
        with self._lock:
            return dict(self._latest_status)

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._latest_frame is not None