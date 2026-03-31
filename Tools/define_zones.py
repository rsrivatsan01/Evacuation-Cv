# ─────────────────────────────────────────────
# tools/define_zones.py
# Interactive exit zone drawing tool.
# Click and drag on the first frame of your
# video to define exit zones.
#
# Controls:
#   Click + Drag  → Draw a zone rectangle
#   's'           → Save zones to data/zones.json
#   'r'           → Redo last zone (delete and redraw)
#   'c'           → Clear all zones
#   'q'           → Quit without saving
#
# Usage:
#   python tools/define_zones.py
#   python tools/define_zones.py --video path/to/video.mp4
#   python tools/define_zones.py --frame 100  (use frame 100)
# ─────────────────────────────────────────────

import cv2
import json
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SAMPLE_VIDEO_PATH
from src.zones import ZoneManager, ZONES_FILE
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Zone colors (cycle through these) ─────────
ZONE_COLORS = [
    (0,   255, 0),    # Green
    (0,   200, 255),  # Yellow
    (255, 128, 0),    # Blue
    (0,   128, 255),  # Orange
    (255, 0,   128),  # Purple
]


class ZoneDrawer:
    """
    Interactive tool for drawing exit zones on a video frame.
    Uses OpenCV mouse callbacks for click-and-drag rectangle drawing.
    """

    def __init__(self, frame, existing_zones=None):
        self.base_frame     = frame.copy()
        self.display_frame  = frame.copy()
        self.zones          = existing_zones or []   # List of dicts

        # Drawing state
        self.drawing    = False
        self.start_x    = 0
        self.start_y    = 0
        self.current_x  = 0
        self.current_y  = 0
        self.temp_rect  = None    # Rectangle being drawn

        self._next_id   = max((z["id"] for z in self.zones), default=0) + 1

    def _get_color(self):
        idx = (len(self.zones)) % len(ZONE_COLORS)
        return ZONE_COLORS[idx]

    def mouse_callback(self, event, x, y, flags, param):
        """OpenCV mouse event handler."""

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_x = x
            self.start_y = y

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.current_x = x
            self.current_y = y
            self._refresh_display()

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            x1 = min(self.start_x, x)
            y1 = min(self.start_y, y)
            x2 = max(self.start_x, x)
            y2 = max(self.start_y, y)

            # Ignore tiny accidental clicks
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                logger.warning("Zone too small — ignored. Try dragging a larger area.")
                self._refresh_display()
                return

            # Prompt user for zone info in terminal
            print(f"\n✅ Zone rectangle drawn: [{x1}, {y1}, {x2}, {y2}]")
            name = input("   Enter exit name (e.g. 'Exit A - North'): ").strip()
            if not name:
                name = f"Exit {self._next_id}"

            try:
                capacity = int(input("   Enter capacity (max persons, e.g. 20): ").strip())
            except ValueError:
                capacity = 20
                print("   Invalid input — using default capacity: 20")

            color = list(self._get_color())
            zone = {
                "id"       : self._next_id,
                "name"     : name,
                "bbox"     : [x1, y1, x2, y2],
                "capacity" : capacity,
                "color"    : color,
            }
            self.zones.append(zone)
            self._next_id += 1

            print(f"   Zone saved: {name} | capacity={capacity}")
            print(f"   Total zones defined: {len(self.zones)}")
            print("\n   Draw next zone, or press 's' to save, 'r' to redo last, 'q' to quit")
            self._refresh_display()

    def _refresh_display(self):
        """Redraw all zones + current drag rectangle."""
        self.display_frame = self.base_frame.copy()

        # Draw all confirmed zones
        for zone in self.zones:
            x1, y1, x2, y2 = zone["bbox"]
            color = tuple(zone["color"])
            cv2.rectangle(self.display_frame, (x1, y1), (x2, y2), color, 2)

            # Zone label background
            label = f"[{zone['id']}] {zone['name']} (cap:{zone['capacity']})"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                self.display_frame,
                (x1, y1 - th - 8),
                (x1 + tw + 6, y1),
                color, -1
            )
            cv2.putText(
                self.display_frame, label,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, cv2.LINE_AA
            )

        # Draw current drag rectangle (dashed effect via color)
        if self.drawing:
            x1 = min(self.start_x, self.current_x)
            y1 = min(self.start_y, self.current_y)
            x2 = max(self.start_x, self.current_x)
            y2 = max(self.start_y, self.current_y)
            cv2.rectangle(
                self.display_frame, (x1, y1), (x2, y2),
                (255, 255, 255), 2
            )

        # Draw instructions overlay
        self._draw_instructions()

    def _draw_instructions(self):
        """Draw controls help text in bottom-left corner."""
        h, w = self.display_frame.shape[:2]
        instructions = [
            "ZONE DRAWING TOOL",
            "Click+Drag : Draw zone",
            "S          : Save zones",
            "R          : Redo last zone",
            "C          : Clear all",
            "Q          : Quit",
            f"Zones: {len(self.zones)}",
        ]
        panel_h = len(instructions) * 22 + 10
        panel_w = 220

        overlay = self.display_frame.copy()
        cv2.rectangle(
            overlay,
            (10, h - panel_h - 10),
            (10 + panel_w, h - 10),
            (0, 0, 0), -1
        )
        cv2.addWeighted(overlay, 0.7, self.display_frame, 0.3, 0, self.display_frame)

        for i, line in enumerate(instructions):
            y = h - panel_h + i * 22
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(
                self.display_frame, line,
                (18, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                color, 1, cv2.LINE_AA
            )

    def run(self) -> list:
        """
        Open the drawing window and return the list of defined zones
        when the user presses 's' to save or 'q' to quit.
        """
        window_name = "Define Exit Zones — Evacuation System"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 700)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        self._refresh_display()
        print("\n" + "=" * 55)
        print("  EXIT ZONE DRAWING TOOL")
        print("=" * 55)
        print("  Click and drag on the frame to draw exit zones.")
        print("  After drawing each zone, enter its name and capacity.")
        print("\n  Controls:")
        print("    S → Save zones to data/zones.json")
        print("    R → Redo last zone")
        print("    C → Clear all zones")
        print("    Q → Quit without saving")
        print("=" * 55 + "\n")

        while True:
            cv2.imshow(window_name, self.display_frame)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('s'):
                if not self.zones:
                    print("⚠️  No zones defined yet — draw at least one zone first.")
                    continue
                cv2.destroyAllWindows()
                return self.zones

            elif key == ord('r'):
                if self.zones:
                    removed = self.zones.pop()
                    self._next_id -= 1
                    print(f"↩️  Removed last zone: {removed['name']}")
                    self._refresh_display()
                else:
                    print("⚠️  No zones to redo.")

            elif key == ord('c'):
                self.zones.clear()
                self._next_id = 1
                print("🗑️  All zones cleared.")
                self._refresh_display()

            elif key == ord('q'):
                print("Quit without saving.")
                cv2.destroyAllWindows()
                return []

            # Handle window close button
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
        return []


def get_frame(video_path: str, frame_number: int = 0) -> "np.ndarray":
    """Extract a specific frame from the video for zone drawing."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = min(frame_number, total - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Could not read frame {frame_number} from {video_path}")

    logger.info(
        f"Using frame {frame_number}/{total} from {os.path.basename(video_path)} "
        f"({frame.shape[1]}x{frame.shape[0]})"
    )
    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Interactive exit zone drawing tool"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=SAMPLE_VIDEO_PATH,
        help="Path to input video",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame number to use as background (default: 0)",
    )
    parser.add_argument(
        "--load-existing",
        action="store_true",
        help="Load and edit existing zones from data/zones.json",
    )
    args = parser.parse_args()

    # ── Validate video ────────────────────────────────
    if not os.path.exists(args.video):
        print(f"❌ Video not found: {args.video}")
        print(f"   Please place your video at: {SAMPLE_VIDEO_PATH}")
        print(f"   Or pass --video path/to/video.mp4")
        sys.exit(1)

    # ── Extract frame ─────────────────────────────────
    print(f"Loading frame from: {args.video}")
    frame = get_frame(args.video, args.frame)

    # ── Load existing zones if requested ──────────────
    existing_zones = []
    if args.load_existing and os.path.exists(ZONES_FILE):
        with open(ZONES_FILE) as f:
            existing_zones = json.load(f)
        print(f"Loaded {len(existing_zones)} existing zones for editing.")

    # ── Run zone drawing tool ─────────────────────────
    drawer = ZoneDrawer(frame, existing_zones=existing_zones)
    zones  = drawer.run()

    if not zones:
        print("No zones saved.")
        sys.exit(0)

    # ── Save zones via ZoneManager ────────────────────
    zm = ZoneManager()
    zm.zones = []
    for z in zones:
        from src.zones import ExitZone
        zm.zones.append(ExitZone.from_dict(z))

    success = zm.save()
    if success:
        print(f"\n✅ {len(zones)} zone(s) saved to: {ZONES_FILE}")
        print("\nSaved zones:")
        for z in zones:
            print(f"  [{z['id']}] {z['name']} | bbox={z['bbox']} | capacity={z['capacity']}")
        print(f"\nYou can now run: python test_pipeline.py")
    else:
        print("❌ Failed to save zones.")
        sys.exit(1)


if __name__ == "__main__":
    main()