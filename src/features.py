# ─────────────────────────────────────────────
# src/features.py
# Crowd feature extraction.
# Computes per-zone and global features every
# frame and logs them to CSV for LSTM training.
# ─────────────────────────────────────────────

import os
import sys
import time
import csv
import numpy as np
import supervision as sv
from collections import defaultdict, deque
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LOG_FEATURES_CSV, LOG_DIR
from src.zones import ZoneManager
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Config ────────────────────────────────────
SPEED_HISTORY_LEN  = 5     # Frames to smooth speed over
GLOBAL_LOG_CSV     = os.path.join(LOG_DIR, "global_features_log.csv")
ZONE_LOG_CSV       = LOG_FEATURES_CSV   # data/logs/features_log.csv


class FeatureExtractor:
    """
    Extracts crowd features per zone and globally from
    tracked detections every frame.

    Features per zone:
        - count     : persons inside zone
        - avg_speed : mean speed of persons in zone (px/frame)
        - density   : persons per 1000 px² of zone area

    Global features:
        - global_count   : total active persons in scene
        - global_speed   : mean speed of all tracked persons
        - global_density : persons per 1000 px² of full frame

    Usage:
        extractor = FeatureExtractor(zone_manager, frame_shape)
        features  = extractor.update(tracked_detections, frame_id)
    """

    def __init__(
        self,
        zone_manager : ZoneManager,
        frame_shape  : tuple,          # (height, width)
        log_csv      : bool = True,
    ):
        self.zone_manager  = zone_manager
        self.frame_h       = frame_shape[0]
        self.frame_w       = frame_shape[1]
        self.frame_area    = self.frame_h * self.frame_w
        self.log_csv       = log_csv

        # ── Position history per track ID ─────────────
        # track_id → deque of (x_center, y_bottom) positions
        self.position_history : Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=SPEED_HISTORY_LEN + 1)
        )

        # ── Speed history per track ID (smoothing) ────
        self.speed_history : Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=SPEED_HISTORY_LEN)
        )

        # ── CSV writers ───────────────────────────────
        self._zone_csv_file   = None
        self._global_csv_file = None
        self._zone_writer     = None
        self._global_writer   = None

        if self.log_csv:
            self._init_csv_writers()

        logger.info(
            f"FeatureExtractor ready | "
            f"frame={self.frame_w}x{self.frame_h} | "
            f"zones={len(zone_manager)} | "
            f"csv_logging={log_csv}"
        )

    # ── CSV Setup ─────────────────────────────────────

    def _init_csv_writers(self):
        """Open CSV files and write headers if new."""
        os.makedirs(LOG_DIR, exist_ok=True)

        # Zone-level CSV
        zone_new = not os.path.exists(ZONE_LOG_CSV)
        self._zone_csv_file = open(ZONE_LOG_CSV, "a", newline="")
        self._zone_writer   = csv.DictWriter(
            self._zone_csv_file,
            fieldnames=[
                "timestamp", "frame_id", "zone_id", "zone_name",
                "count", "avg_speed", "density", "status"
            ]
        )
        if zone_new:
            self._zone_writer.writeheader()

        # Global CSV
        global_new = not os.path.exists(GLOBAL_LOG_CSV)
        self._global_csv_file = open(GLOBAL_LOG_CSV, "a", newline="")
        self._global_writer   = csv.DictWriter(
            self._global_csv_file,
            fieldnames=[
                "timestamp", "frame_id",
                "global_count", "global_speed", "global_density"
            ]
        )
        if global_new:
            self._global_writer.writeheader()

        logger.info(f"CSV logging → {ZONE_LOG_CSV}")
        logger.info(f"CSV logging → {GLOBAL_LOG_CSV}")

    def close(self):
        """Close open CSV file handles. Call when pipeline ends."""
        if self._zone_csv_file:
            self._zone_csv_file.close()
        if self._global_csv_file:
            self._global_csv_file.close()

    # ── Speed Computation ─────────────────────────────

    def _update_positions(self, detections: sv.Detections):
        """
        Update position history for all currently tracked persons.
        Uses bottom-center of bounding box (feet position).
        """
        if detections.tracker_id is None or len(detections) == 0:
            return

        for i, bbox in enumerate(detections.xyxy):
            track_id = int(detections.tracker_id[i])
            bx = float((bbox[0] + bbox[2]) / 2)   # horizontal center
            by = float(bbox[3])                     # bottom edge (feet)
            self.position_history[track_id].append((bx, by))

    def _compute_speed(self, track_id: int) -> float:
        """
        Compute smoothed speed for a single track ID.
        Speed = Euclidean distance between oldest and newest
        position in history, divided by number of steps.

        Returns:
            Speed in pixels per frame (smoothed).
        """
        history = self.position_history[track_id]
        if len(history) < 2:
            return 0.0

        # Euclidean distance from oldest to newest position
        x0, y0 = history[0]
        x1, y1 = history[-1]
        dist    = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        steps   = len(history) - 1
        # Divide by 2 to compensate for the `skip=2` frame optimization in pipeline
        speed   = (dist / steps) / 2.0

        # Add to speed history for further smoothing
        self.speed_history[track_id].append(speed)
        return float(np.mean(self.speed_history[track_id]))

    def _get_speeds(self, detections: sv.Detections) -> Dict[int, float]:
        """
        Compute speed for all currently tracked persons.

        Returns:
            Dict mapping track_id → speed (px/frame)
        """
        speeds = {}
        if detections.tracker_id is None:
            return speeds
        for track_id in detections.tracker_id:
            speeds[int(track_id)] = self._compute_speed(int(track_id))
        return speeds

    # ── Zone Feature Computation ──────────────────────

    def _compute_zone_features(
        self,
        zone,
        detections : sv.Detections,
        speeds     : Dict[int, float],
    ) -> dict:
        """
        Compute features for a single exit zone.

        Args:
            zone       : ExitZone object
            detections : All tracked detections this frame
            speeds     : Dict of track_id → speed

        Returns:
            Feature dict for this zone.
        """
        zone_area = max(
            (zone.x2 - zone.x1) * (zone.y2 - zone.y1), 1
        )

        # Persons inside this zone
        zone_track_ids = list(zone.person_ids)
        count          = len(zone_track_ids)

        # Average speed of persons inside zone
        if count > 0 and speeds:
            zone_speeds = [
                speeds.get(tid, 0.0) for tid in zone_track_ids
            ]
            avg_speed = float(np.mean(zone_speeds))
        else:
            avg_speed = 0.0

        # Density: persons per 1000 px²
        # Cap the maximum mathematical area to ~12000 px^2 (the size of a typical door in the training set).
        # This allows the visual bounding box to remain large to track queues,
        # but prevents the LSTM's density calculation from being artificially diluted.
        effective_area = min(zone_area, 12000.0)
        density = (count / effective_area) * 1000

        return {
            "zone_id"   : zone.id,
            "zone_name" : zone.name,
            "count"     : count,
            "avg_speed" : round(avg_speed, 4),
            "density"   : round(density, 6),
            "status"    : zone.current_status,
        }

    # ── Global Feature Computation ────────────────────

    def _compute_global_features(
        self,
        detections : sv.Detections,
        speeds     : Dict[int, float],
    ) -> dict:
        """
        Compute global scene-level features.

        Returns:
            Feature dict for the whole scene.
        """
        count = len(detections)

        if count > 0 and speeds:
            avg_speed = float(np.mean(list(speeds.values())))
        else:
            avg_speed = 0.0

        density = (count / max(self.frame_area, 1)) * 1000

        return {
            "global_count"   : count,
            "global_speed"   : round(avg_speed, 4),
            "global_density" : round(density, 6),
        }

    # ── Main Update ───────────────────────────────────

    def update(
        self,
        detections : sv.Detections,
        frame_id   : int,
    ) -> dict:
        """
        Extract all features for the current frame.
        Call this AFTER zone_manager.update_counts() so zone
        person_ids are already populated.

        Args:
            detections : Tracked sv.Detections (with tracker_id)
            frame_id   : Current frame number

        Returns:
            Full feature dict for this frame (zones + global).
        """
        timestamp = time.time()

        # ── Update position history ───────────────────
        self._update_positions(detections)

        # ── Compute speeds ────────────────────────────
        speeds = self._get_speeds(detections)

        # ── Zone features ─────────────────────────────
        zone_features = []
        for zone in self.zone_manager.zones:
            zf = self._compute_zone_features(zone, detections, speeds)
            zone_features.append(zf)

        # ── Global features ───────────────────────────
        gf = self._compute_global_features(detections, speeds)

        # ── Assemble full feature dict ─────────────────
        features = {
            "frame_id"       : frame_id,
            "timestamp"      : timestamp,
            "global_count"   : gf["global_count"],
            "global_speed"   : gf["global_speed"],
            "global_density" : gf["global_density"],
            "zones"          : zone_features,
        }

        # ── Log to CSV manually from pipeline ─────────
        # (Removed automatic logging so pipeline can pass predictions)

        # ── Cleanup stale track IDs ───────────────────
        self._cleanup_stale_tracks(detections)

        return features

    # ── CSV Logging ───────────────────────────────────

    def log_features(self, features: dict, predictions: dict = None):
        """Write current frame features to both CSV files, tracking LSTM status."""
        if not self.log_csv:
            return
        ts       = features["timestamp"]
        frame_id = features["frame_id"]

        # Zone-level rows
        for zf in features["zones"]:
            # Override status with ML prediction if available
            pred_status = zf["status"]
            if predictions and zf["zone_id"] in predictions:
                pred_status = predictions[zf["zone_id"]].get("status", pred_status)
                
            self._zone_writer.writerow({
                "timestamp" : round(ts, 3),
                "frame_id"  : frame_id,
                "zone_id"   : zf["zone_id"],
                "zone_name" : zf["zone_name"],
                "count"     : zf["count"],
                "avg_speed" : zf["avg_speed"],
                "density"   : zf["density"],
                "status"    : pred_status,
            })

        # Global row
        self._global_writer.writerow({
            "timestamp"      : round(ts, 3),
            "frame_id"       : frame_id,
            "global_count"   : features["global_count"],
            "global_speed"   : features["global_speed"],
            "global_density" : features["global_density"],
        })

        # Flush periodically
        if frame_id % 100 == 0:
            self._zone_csv_file.flush()
            self._global_csv_file.flush()

    # ── Cleanup ───────────────────────────────────────

    def _cleanup_stale_tracks(self, detections: sv.Detections):
        """
        Remove position/speed history for track IDs no longer active.
        Prevents unbounded memory growth over long videos.
        """
        if detections.tracker_id is None:
            return

        active_ids = set(detections.tracker_id.tolist())
        stale_ids  = set(self.position_history.keys()) - active_ids

        for tid in stale_ids:
            self.position_history.pop(tid, None)
            self.speed_history.pop(tid, None)

    # ── Utility ───────────────────────────────────────

    def get_zone_series(self, zone_id: int, n_frames: int = 30) -> Optional[np.ndarray]:
        """
        Returns the last n_frames of features for a zone as a numpy array.
        Shape: (n_frames, 3) — [count, avg_speed, density]
        Used by LSTM in Phase 6 for real-time prediction.

        Note: This requires reading from the CSV. For real-time use,
        Phase 6 will maintain an in-memory ring buffer instead.
        """
        if not os.path.exists(ZONE_LOG_CSV):
            return None

        import pandas as pd
        try:
            df = pd.read_csv(ZONE_LOG_CSV)
            zone_df = df[df["zone_id"] == zone_id].tail(n_frames)
            if len(zone_df) < n_frames:
                return None
            return zone_df[["count", "avg_speed", "density"]].values.astype(np.float32)
        except Exception as e:
            logger.error(f"Failed to read zone series: {e}")
            return None