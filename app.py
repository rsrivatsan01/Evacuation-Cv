# ─────────────────────────────────────────────
# app.py — Flask dashboard entry point
# Run with: python app.py
# Open:     http://localhost:5000
# ─────────────────────────────────────────────

import os
import sys
import json
import pandas as pd
from flask import (
    Flask, render_template, Response,
    jsonify, request, redirect, url_for
)
from werkzeug.utils import secure_filename

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG,
    SAMPLE_VIDEO_PATH, REALTIME_SOURCE,
    FLOORMAP_PATH, LOG_FEATURES_CSV, LOG_DIR,
)
from src.pipeline import EvacuationPipeline
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Flask app setup ───────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024   # 16MB upload limit
app.secret_key = "evacuation_system_secret"

ALLOWED_IMAGE_EXT = {'png', 'jpg', 'jpeg'}

# ── Read video source from config ─────────────
def get_video_source():
    """Return active video source from config."""
    src = os.environ.get("VIDEO_SOURCE", str(SAMPLE_VIDEO_PATH))
    try:
        return int(src)   # webcam index
    except ValueError:
        return src        # file path or RTSP URL

# ── Start pipeline ────────────────────────────
pipeline = EvacuationPipeline(video_source=get_video_source())
pipeline.start()

# ══════════════════════════════════════════════
# PAGES
# ══════════════════════════════════════════════

@app.route("/")
def index():
    """Live dashboard page."""
    return render_template("dashboard.html")


@app.route("/analytics")
def analytics():
    """Analytics page."""
    return render_template("analytics.html")


# ══════════════════════════════════════════════
# LIVE STREAM ROUTES
# ══════════════════════════════════════════════

def generate_frames(source: str):
    """Generator for MJPEG video stream."""
    while True:
        if source == "video":
            frame = pipeline.get_frame()
        else:
            frame = pipeline.get_floormap()

        if frame is None:
            # Send loading placeholder
            import numpy as np
            import cv2
            placeholder = np.zeros((360, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Initializing pipeline...",
                        (120, 180), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (100, 200, 100), 2)
            _, buf   = cv2.imencode('.jpg', placeholder)
            frame    = buf.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n'
            + frame +
            b'\r\n'
        )

        import time
        time.sleep(1.0 / 15)   # ~15 FPS stream


@app.route("/video_feed")
def video_feed():
    """MJPEG stream of annotated video."""
    return Response(
        generate_frames("video"),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route("/floormap_feed")
def floormap_feed():
    """MJPEG stream of floor map with path overlay."""
    return Response(
        generate_frames("floormap"),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ══════════════════════════════════════════════
# STATUS API
# ══════════════════════════════════════════════

@app.route("/status")
def status():
    """JSON API — returns current pipeline status."""
    return jsonify(pipeline.get_status())


@app.route("/api/zones")
def get_zones():
    """Return current zone definitions."""
    zones_path = os.path.join("data", "zones.json")
    if os.path.exists(zones_path):
        with open(zones_path) as f:
            zones = json.load(f)
        return jsonify({"zones": zones})
    return jsonify({"zones": []})


# ══════════════════════════════════════════════
# ZONE MANAGEMENT API
# ══════════════════════════════════════════════

@app.route("/api/zones/update", methods=["POST"])
def update_zone():
    """Update a zone's name and capacity."""
    data     = request.get_json()
    zone_id  = data.get("id")
    name     = data.get("name")
    capacity = data.get("capacity")

    zones_path = os.path.join("data", "zones.json")
    if not os.path.exists(zones_path):
        return jsonify({"error": "zones.json not found"}), 404

    with open(zones_path) as f:
        zones = json.load(f)

    updated = False
    for zone in zones:
        if zone["id"] == zone_id:
            if name:     zone["name"]     = name
            if capacity: zone["capacity"] = int(capacity)
            updated = True
            break

    if not updated:
        return jsonify({"error": f"Zone {zone_id} not found"}), 404

    with open(zones_path, "w") as f:
        json.dump(zones, f, indent=2)

    pipeline.reload_zones()
    return jsonify({"success": True, "zones": zones})


@app.route("/api/zones/delete", methods=["POST"])
def delete_zone():
    """Delete a zone by ID."""
    data    = request.get_json()
    zone_id = data.get("id")

    zones_path = os.path.join("data", "zones.json")
    if not os.path.exists(zones_path):
        return jsonify({"error": "zones.json not found"}), 404

    with open(zones_path) as f:
        zones = json.load(f)

    zones = [z for z in zones if z["id"] != zone_id]

    with open(zones_path, "w") as f:
        json.dump(zones, f, indent=2)

    pipeline.reload_zones()
    return jsonify({"success": True, "zones": zones})


# ══════════════════════════════════════════════
# FLOOR MAP UPLOAD
# ══════════════════════════════════════════════

@app.route("/api/upload_floormap", methods=["POST"])
def upload_floormap():
    """Upload and replace the floor map image."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_IMAGE_EXT:
        return jsonify({"error": f"Invalid file type. Allowed: {ALLOWED_IMAGE_EXT}"}), 400

    os.makedirs(os.path.dirname(FLOORMAP_PATH), exist_ok=True)
    file.save(FLOORMAP_PATH)
    logger.info(f"Floor map updated: {FLOORMAP_PATH}")
    return jsonify({"success": True, "message": "Floor map updated"})


# ══════════════════════════════════════════════
# ANALYTICS API
# ══════════════════════════════════════════════

@app.route("/api/analytics/zone_features")
def analytics_zone_features():
    """Return zone feature time series for charts."""
    if not os.path.exists(LOG_FEATURES_CSV):
        return jsonify({"error": "No feature log found. Run the pipeline first."}), 404

    try:
        df = pd.read_csv(LOG_FEATURES_CSV)

        # Limit to last 500 rows for performance
        df = df.tail(500)

        result = {}
        for zone_name, group in df.groupby("zone_name"):
            result[zone_name] = {
                "frame_ids"  : group["frame_id"].tolist(),
                "counts"     : group["count"].tolist(),
                "speeds"     : group["avg_speed"].tolist(),
                "densities"  : group["density"].tolist(),
                "statuses"   : group["status"].tolist(),
            }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analytics/global_features")
def analytics_global_features():
    """Return global feature time series."""
    global_csv = os.path.join(LOG_DIR, "global_features_log.csv")
    if not os.path.exists(global_csv):
        return jsonify({"error": "No global feature log found."}), 404

    try:
        df = pd.read_csv(global_csv).tail(500)
        return jsonify({
            "frame_ids"  : df["frame_id"].tolist(),
            "counts"     : df["global_count"].tolist(),
            "speeds"     : df["global_speed"].tolist(),
            "densities"  : df["global_density"].tolist(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analytics/status_distribution")
def analytics_status_distribution():
    """Return status distribution for pie chart."""
    if not os.path.exists(LOG_FEATURES_CSV):
        return jsonify({"error": "No feature log found."}), 404

    try:
        df     = pd.read_csv(LOG_FEATURES_CSV)
        counts = df["status"].value_counts().to_dict()
        return jsonify({
            "labels" : list(counts.keys()),
            "values" : list(counts.values()),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analytics/zone_comparison")
def analytics_zone_comparison():
    """Return zone comparison summary table."""
    if not os.path.exists(LOG_FEATURES_CSV):
        return jsonify({"error": "No feature log found."}), 404

    try:
        df = pd.read_csv(LOG_FEATURES_CSV)
        result = []
        for zone_name, group in df.groupby("zone_name"):
            result.append({
                "zone"         : zone_name,
                "avg_count"    : round(group["count"].mean(), 2),
                "max_count"    : int(group["count"].max()),
                "avg_speed"    : round(group["avg_speed"].mean(), 2),
                "avg_density"  : round(group["density"].mean(), 4),
                "safe_pct"     : round((group["status"] == "SAFE").mean() * 100, 1),
                "moderate_pct" : round((group["status"] == "MODERATE").mean() * 100, 1),
                "critical_pct" : round((group["status"] == "CRITICAL").mean() * 100, 1),
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════

if __name__ == "__main__":
    logger.info(f"Starting dashboard at http://{FLASK_HOST}:{FLASK_PORT}")
    app.run(
        host    = FLASK_HOST,
        port    = FLASK_PORT,
        debug   = FLASK_DEBUG,
        threaded= True,
    )