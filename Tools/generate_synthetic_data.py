# ─────────────────────────────────────────────
# tools/generate_synthetic_data.py
# Synthetic crowd feature data generator.
# Simulates realistic crowd flow patterns
# with SAFE / MODERATE / CRITICAL transitions.
#
# Usage:
#   python tools/generate_synthetic_data.py
#   python tools/generate_synthetic_data.py --frames 3000
#   python tools/generate_synthetic_data.py --zones 3 --frames 2000
# ─────────────────────────────────────────────

import os
import sys
import csv
import time
import random
import argparse
import numpy as np
from enum import Enum

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    LOG_DIR,
    LOG_FEATURES_CSV,
    CONGESTION_SAFE,
    CONGESTION_MODERATE,
)

GLOBAL_LOG_CSV = os.path.join(LOG_DIR, "global_features_log.csv")

# ── Crowd state machine ───────────────────────
class CrowdState(Enum):
    QUIET      = "quiet"       # Very few people
    BUILDING   = "building"    # Crowd growing
    RUSH       = "rush"        # Peak crowd
    DISPERSING = "dispersing"  # Crowd leaving
    INCIDENT   = "incident"    # Sudden surge (evacuation scenario)


# ── Zone simulation parameters ────────────────
ZONE_CONFIGS = [
    {
        "id"       : 1,
        "name"     : "Exit A - North",
        "capacity" : 20,
        "area_px2" : 15000,
        "color"    : [0, 255, 0],
    },
    {
        "id"       : 2,
        "name"     : "Exit B - South",
        "capacity" : 15,
        "area_px2" : 12000,
        "color"    : [0, 200, 255],
    },
    {
        "id"       : 3,
        "name"     : "Exit C - East",
        "capacity" : 10,
        "area_px2" : 8000,
        "color"    : [255, 128, 0],
    },
]

# ── Simulation constants ──────────────────────
FRAME_RATE      = 25       # Simulated FPS
STATE_DURATIONS = {        # How long each state lasts (in frames)
    CrowdState.QUIET      : (100, 300),
    CrowdState.BUILDING   : (50,  150),
    CrowdState.RUSH       : (80,  250),
    CrowdState.DISPERSING : (60,  180),
    CrowdState.INCIDENT   : (30,  100),
}
STATE_TRANSITIONS = {      # Possible next states from each state
    CrowdState.QUIET      : [CrowdState.BUILDING, CrowdState.QUIET],
    CrowdState.BUILDING   : [CrowdState.RUSH, CrowdState.DISPERSING],
    CrowdState.RUSH       : [CrowdState.DISPERSING, CrowdState.INCIDENT],
    CrowdState.DISPERSING : [CrowdState.QUIET, CrowdState.BUILDING],
    CrowdState.INCIDENT   : [CrowdState.RUSH, CrowdState.DISPERSING],
}
STATE_WEIGHTS = {          # Probability weights for transitions
    CrowdState.QUIET      : [0.6, 0.4],
    CrowdState.BUILDING   : [0.7, 0.3],
    CrowdState.RUSH       : [0.6, 0.4],
    CrowdState.DISPERSING : [0.5, 0.5],
    CrowdState.INCIDENT   : [0.4, 0.6],
}


class ZoneSimulator:
    """
    Simulates realistic crowd count, speed, and density
    for a single exit zone using a state machine.
    """

    def __init__(self, zone_config: dict, seed: int = None):
        self.id       = zone_config["id"]
        self.name     = zone_config["name"]
        self.capacity = zone_config["capacity"]
        self.area     = zone_config["area_px2"]
        self.color    = zone_config["color"]

        if seed is not None:
            np.random.seed(seed)

        # State machine
        self.state          = CrowdState.QUIET
        self.state_frame    = 0
        self.state_duration = self._sample_duration(CrowdState.QUIET)

        # Smooth count using exponential moving average
        self.smooth_count   = 0.0
        self.target_count   = 0.0

    def _sample_duration(self, state: CrowdState) -> int:
        lo, hi = STATE_DURATIONS[state]
        return random.randint(lo, hi)

    def _get_target_count(self, state: CrowdState) -> float:
        """
        Returns target person count for a given state.
        Counts are calibrated against zone capacity.
        """
        cap = self.capacity
        if state == CrowdState.QUIET:
            return np.random.uniform(0, cap * 0.3)          # 0–30% capacity
        elif state == CrowdState.BUILDING:
            return np.random.uniform(cap * 0.25, cap * 0.6) # 25–60% capacity
        elif state == CrowdState.RUSH:
            return np.random.uniform(cap * 0.55, cap * 1.2) # 55–120% capacity
        elif state == CrowdState.DISPERSING:
            return np.random.uniform(cap * 0.1, cap * 0.5)  # 10–50% capacity
        elif state == CrowdState.INCIDENT:
            return np.random.uniform(cap * 0.8, cap * 1.5)  # 80–150% capacity
        return 0.0

    def _get_speed(self, count: float, state: CrowdState) -> float:
        """
        Speed is inversely related to density — crowded zones move slower.
        Incident state causes higher speed (panic/rushing).
        """
        cap   = max(self.capacity, 1)
        ratio = count / cap

        if state == CrowdState.INCIDENT:
            base_speed = np.random.uniform(6.0, 15.0)
        elif state == CrowdState.RUSH:
            # High density = slow movement
            base_speed = np.random.uniform(0.5, 3.0) * (1 - ratio * 0.5)
        elif state == CrowdState.BUILDING:
            base_speed = np.random.uniform(2.0, 6.0)
        elif state == CrowdState.DISPERSING:
            base_speed = np.random.uniform(3.0, 8.0)
        else:  # QUIET
            base_speed = np.random.uniform(1.5, 5.0)

        # Add noise
        noise = np.random.normal(0, 0.5)
        return max(0.0, base_speed + noise)

    def _transition_state(self):
        """Move to the next state based on transition probabilities."""
        next_states  = STATE_TRANSITIONS[self.state]
        weights      = STATE_WEIGHTS[self.state]
        self.state   = random.choices(next_states, weights=weights)[0]
        self.state_duration = self._sample_duration(self.state)
        self.state_frame    = 0
        self.target_count   = self._get_target_count(self.state)

    def step(self) -> dict:
        """
        Advance simulation by one frame and return features.
        """
        # Check state transition
        self.state_frame += 1
        if self.state_frame >= self.state_duration:
            self._transition_state()

        # Smoothly move toward target count (EMA)
        alpha             = 0.08   # Smoothing factor — lower = smoother
        self.smooth_count += alpha * (self.target_count - self.smooth_count)

        # Add per-frame noise
        noise = np.random.normal(0, 0.3)
        raw_count = max(0.0, self.smooth_count + noise)

        # Discretize to integer
        count = int(round(raw_count))
        count = max(0, count)

        # Compute speed
        avg_speed = self._get_speed(raw_count, self.state)

        # Compute density (persons per 1000 px²)
        density = (raw_count / max(self.area, 1)) * 1000

        # Compute status
        status = self._compute_status(count)

        return {
            "zone_id"   : self.id,
            "zone_name" : self.name,
            "count"     : count,
            "avg_speed" : round(avg_speed, 4),
            "density"   : round(density, 6),
            "status"    : status,
            "state"     : self.state.value,   # for debugging only
        }

    def _compute_status(self, count: int) -> str:
        """
        Use capacity-relative thresholds so every zone can
        reach all 3 status levels regardless of its size.
        """
        ratio = count / max(self.capacity, 1)
        if ratio < 0.5:
            return "SAFE"
        elif ratio < 0.85:
            return "MODERATE"
        else:
            return "CRITICAL"


class GlobalSimulator:
    """
    Simulates global scene features (whole frame).
    Aggregates from zone simulators + adds scene-level noise.
    """

    def __init__(self, frame_area_px2: int = 921600):  # 1280x720
        self.frame_area = frame_area_px2

    def compute(self, zone_features: list) -> dict:
        """Compute global features from zone features."""
        # Global count = sum of all zone counts + people outside zones
        zone_total  = sum(zf["count"] for zf in zone_features)
        extra_people = max(0, int(np.random.normal(8, 3)))   # people not in any zone
        global_count = zone_total + extra_people

        # Global speed = weighted average
        speeds = [zf["avg_speed"] for zf in zone_features if zf["count"] > 0]
        if speeds:
            global_speed = float(np.mean(speeds)) + np.random.normal(0, 0.5)
        else:
            global_speed = abs(np.random.normal(3.0, 1.5))

        global_speed = max(0.0, global_speed)

        # Global density
        global_density = (global_count / max(self.frame_area, 1)) * 1000

        return {
            "global_count"   : global_count,
            "global_speed"   : round(global_speed, 4),
            "global_density" : round(global_density, 6),
        }


def generate(
    n_frames    : int,
    zone_configs: list,
    output_zone : str,
    output_global: str,
    seed        : int = 42,
):
    """
    Generate synthetic feature logs.

    Args:
        n_frames      : Total frames to simulate
        zone_configs  : List of zone config dicts
        output_zone   : Path for zone-level CSV
        output_global : Path for global CSV
        seed          : Random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(os.path.dirname(output_zone), exist_ok=True)

    # ── Init simulators ───────────────────────────
    zone_sims = [
        ZoneSimulator(cfg, seed=seed + i)
        for i, cfg in enumerate(zone_configs)
    ]
    global_sim = GlobalSimulator()

    # ── Open CSV writers ──────────────────────────
    zone_f   = open(output_zone,   "w", newline="")
    global_f = open(output_global, "w", newline="")

    zone_writer = csv.DictWriter(zone_f, fieldnames=[
        "timestamp", "frame_id", "zone_id", "zone_name",
        "count", "avg_speed", "density", "status"
    ])
    global_writer = csv.DictWriter(global_f, fieldnames=[
        "timestamp", "frame_id",
        "global_count", "global_speed", "global_density"
    ])
    zone_writer.writeheader()
    global_writer.writeheader()

    # ── Simulate ──────────────────────────────────
    base_time     = time.time()
    status_counts = {"SAFE": 0, "MODERATE": 0, "CRITICAL": 0}

    print(f"\n🔄 Generating {n_frames:,} frames of synthetic data...")
    print(f"   Zones     : {[z['name'] for z in zone_configs]}")
    print(f"   Output    : {output_zone}")
    print()

    for frame_id in range(1, n_frames + 1):
        timestamp = base_time + frame_id / FRAME_RATE

        # Step each zone simulator
        zone_features = [sim.step() for sim in zone_sims]

        # Compute global features
        gf = global_sim.compute(zone_features)

        # Write zone rows
        for zf in zone_features:
            zone_writer.writerow({
                "timestamp" : round(timestamp, 3),
                "frame_id"  : frame_id,
                "zone_id"   : zf["zone_id"],
                "zone_name" : zf["zone_name"],
                "count"     : zf["count"],
                "avg_speed" : zf["avg_speed"],
                "density"   : zf["density"],
                "status"    : zf["status"],
            })
            status_counts[zf["status"]] = status_counts.get(zf["status"], 0) + 1

        # Write global row
        global_writer.writerow({
            "timestamp"      : round(timestamp, 3),
            "frame_id"       : frame_id,
            "global_count"   : gf["global_count"],
            "global_speed"   : gf["global_speed"],
            "global_density" : gf["global_density"],
        })

        # Progress update
        if frame_id % 500 == 0 or frame_id == n_frames:
            pct = frame_id / n_frames * 100
            print(f"   Frame {frame_id:>5,}/{n_frames:,} ({pct:5.1f}%) — "
                  f"SAFE:{status_counts['SAFE']:,} | "
                  f"MODERATE:{status_counts['MODERATE']:,} | "
                  f"CRITICAL:{status_counts['CRITICAL']:,}")

    zone_f.close()
    global_f.close()

    # ── Summary ───────────────────────────────────
    total_zone_rows = n_frames * len(zone_configs)
    print(f"\n{'=' * 55}")
    print(f"  SYNTHETIC DATA GENERATION COMPLETE")
    print(f"{'=' * 55}")
    print(f"  Frames generated  : {n_frames:,}")
    print(f"  Zone rows total   : {total_zone_rows:,}")
    print(f"  Zones             : {len(zone_configs)}")
    print(f"\n  Status distribution (per zone row):")
    for status, count in status_counts.items():
        pct = count / total_zone_rows * 100
        bar = '█' * int(pct / 2)
        print(f"    {status:<12}: {count:>6,} ({pct:5.1f}%)  {bar}")
    print(f"\n  Sequences per zone (~30 frame window):")
    print(f"    ~{max(0, n_frames - 30):,} training sequences per zone")
    print(f"\n  Files saved:")
    print(f"    {output_zone}")
    print(f"    {output_global}")
    print(f"{'=' * 55}")

    return status_counts


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic crowd feature training data"
    )
    parser.add_argument(
        "--frames", type=int, default=3000,
        help="Number of frames to simulate (default: 3000)"
    )
    parser.add_argument(
        "--zones", type=int, default=None,
        help="Number of zones (1-3, default: use all defined zones)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing CSVs instead of overwriting"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Select zone configs
    zones = ZONE_CONFIGS
    if args.zones is not None:
        zones = ZONE_CONFIGS[:max(1, min(args.zones, len(ZONE_CONFIGS)))]

    # Use real zones.json if it exists
    zones_json = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "zones.json"
    )
    if os.path.exists(zones_json):
        import json
        with open(zones_json) as f:
            real_zones = json.load(f)
        if real_zones:
            # Build zone configs from real zones, keeping area estimates
            zones = []
            for z in real_zones:
                bbox     = z["bbox"]
                area_px2 = max((bbox[2]-bbox[0]) * (bbox[3]-bbox[1]), 1)
                zones.append({
                    "id"       : z["id"],
                    "name"     : z["name"],
                    "capacity" : z["capacity"],
                    "area_px2" : area_px2,
                    "color"    : z.get("color", [0, 255, 0]),
                })
            print(f"✅ Using {len(zones)} zones from data/zones.json")

    # Output paths
    out_zone   = LOG_FEATURES_CSV
    out_global = GLOBAL_LOG_CSV

    if args.append:
        print("📎 Append mode — adding to existing CSVs")
    else:
        # Clear existing files
        for f in [out_zone, out_global]:
            if os.path.exists(f):
                os.remove(f)
                print(f"🗑️  Cleared: {f}")

    generate(
        n_frames     = args.frames,
        zone_configs = zones,
        output_zone  = out_zone,
        output_global= out_global,
        seed         = args.seed,
    )