# ─────────────────────────────────────────────
# src/tracking.py
# ByteTrack multi-person tracker wrapper.
# Takes detections from detection.py and assigns
# persistent track IDs across frames.
# ─────────────────────────────────────────────

import numpy as np
import supervision as sv

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TRACK_THRESHOLD,
    TRACK_BUFFER,
    MATCH_THRESHOLD,
    FRAME_RATE,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class PersonTracker:
    """
    Wraps ByteTrack for persistent multi-person tracking.

    ByteTrack assigns a unique integer ID to each person
    and maintains that ID across frames even if detection
    is temporarily lost.

    Usage:
        tracker    = PersonTracker()
        detections = detector.detect(frame)
        tracked    = tracker.update(detections)
        # tracked.tracker_id contains persistent IDs
    """

    def __init__(
        self,
        track_threshold : float = TRACK_THRESHOLD,
        track_buffer    : int   = TRACK_BUFFER,
        match_threshold : float = MATCH_THRESHOLD,
        frame_rate      : int   = FRAME_RATE,
    ):
        """
        Args:
            track_threshold : Min confidence to initiate a new track.
            track_buffer    : Frames to keep a lost track alive before deleting.
            match_threshold : IoU threshold for matching detections to tracks.
            frame_rate      : FPS of the input video.
        """
        self.tracker = sv.ByteTrack(
            track_activation_threshold = track_threshold,
            lost_track_buffer          = track_buffer,
            minimum_matching_threshold = match_threshold,
            frame_rate                 = frame_rate,
        )

        self.frame_count    = 0
        self.active_ids     = set()
        self.total_seen_ids = set()

        logger.info(
            f"PersonTracker (ByteTrack) ready | "
            f"threshold={track_threshold} | buffer={track_buffer} frames"
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Update tracker with new detections and return tracked detections.

        Args:
            detections : sv.Detections from PersonDetector.detect()

        Returns:
            sv.Detections with .tracker_id field populated.
            Only returns currently active tracks.
        """
        self.frame_count += 1

        # Update ByteTrack with new detections
        tracked = self.tracker.update_with_detections(detections)

        # Update active ID sets for statistics
        if tracked.tracker_id is not None:
            current_ids = set(tracked.tracker_id.tolist())
            self.active_ids     = current_ids
            self.total_seen_ids.update(current_ids)

        return tracked

    def reset(self):
        """Reset tracker state — call when switching to a new video."""
        self.tracker.reset()
        self.frame_count    = 0
        self.active_ids     = set()
        self.total_seen_ids = set()
        logger.info("Tracker reset")

    @property
    def active_count(self) -> int:
        """Number of currently tracked persons."""
        return len(self.active_ids)

    @property
    def total_count(self) -> int:
        """Total unique persons seen since last reset."""
        return len(self.total_seen_ids)

    @property
    def stats(self) -> dict:
        """Returns current tracker statistics."""
        return {
            "frame_count"    : self.frame_count,
            "active_tracks"  : self.active_count,
            "total_unique"   : self.total_count,
            "active_ids"     : sorted(list(self.active_ids)),
        }