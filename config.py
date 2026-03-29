# ─────────────────────────────────────────────
# config.py — Central configuration for the
# AI Evacuation Optimization System
# All tunable parameters live here.
# ─────────────────────────────────────────────

import os

# ── Paths ─────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR       = os.path.join(BASE_DIR, "models")
DATA_DIR        = os.path.join(BASE_DIR, "data")
VIDEO_DIR       = os.path.join(DATA_DIR, "videos")
LOG_DIR         = os.path.join(DATA_DIR, "logs")
MAPS_DIR        = os.path.join(DATA_DIR, "maps")
STATIC_DIR      = os.path.join(BASE_DIR, "static")

# ── Model Weights ─────────────────────────────
YOLO_PERSON_WEIGHTS = os.path.join(MODEL_DIR, "yolov8_person.pt")
YOLO_EXIT_WEIGHTS   = os.path.join(MODEL_DIR, "yolov8_exit.pt")
LSTM_WEIGHTS        = os.path.join(MODEL_DIR, "lstm_congestion.pt")

# Fallback: use pretrained YOLOv8n until fine-tuned weights are ready
YOLO_FALLBACK_WEIGHTS = "yolov8n.pt"

# ── Video Input ───────────────────────────────
# Path to the sample test video (pre-recorded mode)
SAMPLE_VIDEO_PATH = os.path.join(VIDEO_DIR, "sample_crowd.mp4")

# For future real-time mode: set to 0 for webcam or an RTSP URL
REALTIME_SOURCE = 0

# ── Detection Parameters ──────────────────────
DETECTION_CONFIDENCE    = 0.4    # Minimum YOLO confidence threshold
DETECTION_IOU           = 0.5    # NMS IoU threshold
PERSON_CLASS_ID         = 0      # COCO class ID for 'person'
TARGET_FRAME_WIDTH      = 1280   # Resize input frames to this width
TARGET_FRAME_HEIGHT     = 720

# ── Tracking Parameters ───────────────────────
TRACK_THRESHOLD         = 0.5    # ByteTrack detection threshold
TRACK_BUFFER            = 30     # Frames to keep lost tracks alive
MATCH_THRESHOLD         = 0.8    # IoU match threshold for ByteTrack
FRAME_RATE              = 25     # Expected FPS of input video

# ── Zone / Exit Configuration ─────────────────
# Each exit zone is defined as:
# { "id": int, "name": str, "bbox": [x1, y1, x2, y2], "capacity": int }
# These are manually defined defaults — will be overridden by UI config tool
DEFAULT_EXIT_ZONES = [
    {"id": 1, "name": "Exit A - North", "bbox": [50,  20,  200, 80],  "capacity": 50},
    {"id": 2, "name": "Exit B - South", "bbox": [50,  620, 200, 680], "capacity": 50},
    {"id": 3, "name": "Exit C - East",  "bbox": [1100, 300, 1260, 420], "capacity": 30},
]

# ── Congestion Thresholds ─────────────────────
# People count per zone that triggers each level
CONGESTION_SAFE         = 10     # < 10 people  → SAFE
CONGESTION_MODERATE     = 25     # 10–25 people → MODERATE
# > 25 people                                  → CRITICAL

# ── LSTM Feature Window ───────────────────────
LSTM_SEQUENCE_LENGTH    = 30     # Number of past frames used per prediction
LSTM_INPUT_FEATURES     = 3      # [people_count, avg_speed, density]
LSTM_HIDDEN_SIZE        = 64
LSTM_NUM_LAYERS         = 2

# ── Path Planning ─────────────────────────────
# Nodes in the venue graph (zone names)
# Edges and weights are computed dynamically from congestion scores
VENUE_GRAPH_NODES = ["Entrance", "Corridor_A", "Corridor_B", "Hall_Center",
                     "Exit_A", "Exit_B", "Exit_C"]

# ── Flask Dashboard ───────────────────────────
FLASK_HOST              = "0.0.0.0"
FLASK_PORT              = 5000
FLASK_DEBUG             = False
VIDEO_STREAM_FPS        = 15     # FPS for MJPEG stream to browser

# ── Logging ───────────────────────────────────
LOG_FEATURES_CSV        = os.path.join(LOG_DIR, "features_log.csv")
LOG_CONGESTION_CSV      = os.path.join(LOG_DIR, "congestion_log.csv")
LOG_LEVEL               = "INFO"

# ── Floor Map ─────────────────────────────────
FLOORMAP_PATH           = os.path.join(STATIC_DIR, "images", "floormap.png")
FLOORMAP_WIDTH          = 800
FLOORMAP_HEIGHT         = 500
