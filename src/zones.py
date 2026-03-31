# ─────────────────────────────────────────────
# src/zones.py
# Exit zone manager.
# Handles loading/saving zones, counting persons
# inside each zone, and computing congestion status.
# ─────────────────────────────────────────────

import json
import os
import numpy as np
import supervision as sv
from typing import List, Dict, Optional

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    CONGESTION_SAFE,
    CONGESTION_MODERATE,
    DEFAULT_EXIT_ZONES,
)
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Default zones file path ───────────────────
ZONES_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "zones.json"
)

# ── Congestion status constants ───────────────
STATUS_SAFE     = "SAFE"
STATUS_MODERATE = "MODERATE"
STATUS_CRITICAL = "CRITICAL"

# ── Status colors (BGR for OpenCV) ────────────
STATUS_COLORS = {
    STATUS_SAFE     : (0,   200, 0),    # Green
    STATUS_MODERATE : (0,   165, 255),  # Orange
    STATUS_CRITICAL : (0,   0,   255),  # Red
}


class ExitZone:
    """
    Represents a single exit zone in the venue.

    Attributes:
        id       : Unique integer ID
        name     : Human-readable name (e.g. "Exit A - North")
        bbox     : [x1, y1, x2, y2] pixel coordinates on the video frame
        capacity : Max persons before zone is considered congested
        color    : BGR color tuple for visualization
    """

    def __init__(
        self,
        id       : int,
        name     : str,
        bbox     : List[int],
        capacity : int   = 20,
        color    : List  = None,
    ):
        self.id       = id
        self.name     = name
        self.bbox     = bbox        # [x1, y1, x2, y2]
        self.capacity = capacity
        self.color    = color or [0, 255, 0]

        # Runtime state — updated every frame
        self.current_count  = 0
        self.current_status = STATUS_SAFE
        self.person_ids     = set()   # Track IDs currently inside zone

    @property
    def x1(self): return self.bbox[0]
    @property
    def y1(self): return self.bbox[1]
    @property
    def x2(self): return self.bbox[2]
    @property
    def y2(self): return self.bbox[3]

    @property
    def center(self):
        """Returns (cx, cy) center of the zone."""
        return (
            (self.x1 + self.x2) // 2,
            (self.y1 + self.y2) // 2,
        )

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point (x, y) falls inside this zone."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def contains_bottom_center(self, bbox: np.ndarray) -> bool:
        """
        Check if the bottom-center of a bounding box is inside this zone.
        Bottom-center is used instead of box center because it represents
        where the person's feet are — more accurate for zone membership.

        Args:
            bbox : [x1, y1, x2, y2] person bounding box
        """
        bx = (bbox[0] + bbox[2]) / 2   # horizontal center
        by = bbox[3]                    # bottom edge (feet)
        return self.contains_point(bx, by)

    def to_dict(self) -> dict:
        """Serialize zone to dict for JSON storage."""
        return {
            "id"       : self.id,
            "name"     : self.name,
            "bbox"     : self.bbox,
            "capacity" : self.capacity,
            "color"    : self.color,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExitZone":
        """Deserialize zone from dict."""
        return cls(
            id       = data["id"],
            name     = data["name"],
            bbox     = data["bbox"],
            capacity = data.get("capacity", 20),
            color    = data.get("color", [0, 255, 0]),
        )

    def __repr__(self):
        return (
            f"ExitZone(id={self.id}, name='{self.name}', "
            f"bbox={self.bbox}, capacity={self.capacity})"
        )


class ZoneManager:
    """
    Manages all exit zones for the venue.

    Responsibilities:
    - Load/save zones from JSON
    - Count persons inside each zone per frame
    - Compute congestion status per zone
    - Support runtime add/edit/delete of zones
    """

    def __init__(self, zones_file: str = ZONES_FILE):
        self.zones_file = zones_file
        self.zones      : List[ExitZone] = []
        self._next_id   = 1

    # ── Persistence ───────────────────────────────────

    def load(self) -> bool:
        """
        Load zones from JSON file.
        Falls back to DEFAULT_EXIT_ZONES from config if file not found.

        Returns:
            True if loaded from file, False if using defaults.
        """
        if os.path.exists(self.zones_file):
            try:
                with open(self.zones_file, "r") as f:
                    data = json.load(f)
                self.zones  = [ExitZone.from_dict(z) for z in data]
                self._next_id = max((z.id for z in self.zones), default=0) + 1
                logger.info(
                    f"Loaded {len(self.zones)} zones from {self.zones_file}"
                )
                return True
            except Exception as e:
                logger.error(f"Failed to load zones: {e}")

        # Fall back to defaults from config
        logger.warning(
            f"Zones file not found: {self.zones_file}\n"
            f"Using {len(DEFAULT_EXIT_ZONES)} default zones from config.py\n"
            f"Run: python tools/define_zones.py  to define your own zones"
        )
        self.zones = [
            ExitZone(
                id       = z["id"],
                name     = z["name"],
                bbox     = z["bbox"],
                capacity = z["capacity"],
            )
            for z in DEFAULT_EXIT_ZONES
        ]
        self._next_id = len(self.zones) + 1
        return False

    def save(self) -> bool:
        """
        Save current zones to JSON file.

        Returns:
            True on success, False on failure.
        """
        try:
            os.makedirs(os.path.dirname(self.zones_file), exist_ok=True)
            data = [z.to_dict() for z in self.zones]
            with open(self.zones_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(
                f"Saved {len(self.zones)} zones to {self.zones_file}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save zones: {e}")
            return False

    # ── Zone CRUD ─────────────────────────────────────

    def add_zone(
        self,
        name     : str,
        bbox     : List[int],
        capacity : int  = 20,
        color    : List = None,
    ) -> ExitZone:
        """Add a new exit zone and return it."""
        zone = ExitZone(
            id       = self._next_id,
            name     = name,
            bbox     = bbox,
            capacity = capacity,
            color    = color or [0, 255, 0],
        )
        self.zones.append(zone)
        self._next_id += 1
        logger.info(f"Added zone: {zone}")
        return zone

    def remove_zone(self, zone_id: int) -> bool:
        """Remove a zone by ID. Returns True if found and removed."""
        before = len(self.zones)
        self.zones = [z for z in self.zones if z.id != zone_id]
        removed = len(self.zones) < before
        if removed:
            logger.info(f"Removed zone ID={zone_id}")
        else:
            logger.warning(f"Zone ID={zone_id} not found")
        return removed

    def get_zone(self, zone_id: int) -> Optional[ExitZone]:
        """Get zone by ID."""
        for z in self.zones:
            if z.id == zone_id:
                return z
        return None

    def update_zone(
        self,
        zone_id  : int,
        name     : str  = None,
        bbox     : List = None,
        capacity : int  = None,
    ) -> bool:
        """Update zone properties. Returns True if found and updated."""
        zone = self.get_zone(zone_id)
        if zone is None:
            return False
        if name     is not None: zone.name     = name
        if bbox     is not None: zone.bbox     = bbox
        if capacity is not None: zone.capacity = capacity
        logger.info(f"Updated zone: {zone}")
        return True

    # ── Per-frame Analysis ────────────────────────────

    def update_counts(self, detections: sv.Detections) -> Dict[int, int]:
        """
        Count how many tracked persons are inside each zone
        and update zone status. Call this once per frame.

        Args:
            detections : sv.Detections with tracker_id populated

        Returns:
            Dict mapping zone_id → person count
        """
        # Reset all zone counts
        for zone in self.zones:
            zone.current_count = 0
            zone.person_ids    = set()

        if len(detections) == 0 or detections.tracker_id is None:
            for zone in self.zones:
                zone.current_status = STATUS_SAFE
            return {z.id: 0 for z in self.zones}

        # Check each detection against each zone
        for i, bbox in enumerate(detections.xyxy):
            track_id = int(detections.tracker_id[i])
            for zone in self.zones:
                if zone.contains_bottom_center(bbox):
                    zone.current_count += 1
                    zone.person_ids.add(track_id)

        # Update status for each zone
        counts = {}
        for zone in self.zones:
            zone.current_status = self.compute_status(
                zone.current_count, zone.capacity
            )
            counts[zone.id] = zone.current_count

        return counts

    @staticmethod
    def compute_status(count: int, capacity: int) -> str:
        """
        Compute congestion status based on person count vs capacity.

        Thresholds (from config.py):
            < CONGESTION_SAFE     → SAFE
            < CONGESTION_MODERATE → MODERATE
            else                  → CRITICAL
        """
        ratio = count / max(capacity, 1)
        if count < CONGESTION_SAFE:
            return STATUS_SAFE
        elif count < CONGESTION_MODERATE:
            return STATUS_MODERATE
        else:
            return STATUS_CRITICAL

    def get_status(self, zone: ExitZone) -> str:
        """Get current status string for a zone."""
        return zone.current_status

    def get_status_color(self, zone: ExitZone) -> tuple:
        """Get BGR color tuple for zone's current status."""
        return STATUS_COLORS.get(zone.current_status, (0, 255, 0))

    def any_critical(self) -> bool:
        """Returns True if any zone is at CRITICAL status."""
        return any(z.current_status == STATUS_CRITICAL for z in self.zones)

    def summary(self) -> List[dict]:
        """
        Returns a list of dicts summarizing all zones.
        Used by Flask API to send status to the dashboard.
        """
        return [
            {
                "id"       : z.id,
                "name"     : z.name,
                "count"    : z.current_count,
                "capacity" : z.capacity,
                "status"   : z.current_status,
                "color"    : z.color,
                "bbox"     : z.bbox,
            }
            for z in self.zones
        ]

    def __len__(self):
        return len(self.zones)

    def __repr__(self):
        return f"ZoneManager({len(self.zones)} zones)"