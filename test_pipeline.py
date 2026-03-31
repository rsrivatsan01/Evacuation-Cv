# ─────────────────────────────────────────────
# test_pipeline.py
# End-to-end test: Detection + Tracking +
# Zones + Features + Congestion + Path Planning
#
# Usage:
#   python test_pipeline.py
#   python test_pipeline.py --no-display
#   python test_pipeline.py --no-log
# ─────────────────────────────────────────────

import cv2
import time
import argparse
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SAMPLE_VIDEO_PATH, VIDEO_DIR, FLOORMAP_PATH
from src.detection    import PersonDetector
from src.tracking     import PersonTracker
from src.zones        import ZoneManager, STATUS_COLORS
from src.features     import FeatureExtractor
from src.congestion   import CongestionInference
from src.pathplanning import VenueGraph, PersonRouter
from utils.visualizer import (
    draw_person_routing,
    build_annotators, draw_detections, draw_stats_panel,
    draw_alert_banner, draw_path_on_frame, draw_path_on_floormap,
)
from utils.logger import get_logger

logger = get_logger(__name__)

FLOORMAP_PATH_LOCAL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "static", "images", "floormap.png"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      type=str,   default=SAMPLE_VIDEO_PATH)
    parser.add_argument("--output",     type=str,
                        default=os.path.join(VIDEO_DIR, "output_annotated.mp4"))
    parser.add_argument("--conf",       type=float, default=0.4)
    parser.add_argument("--skip",       type=int,   default=2)
    parser.add_argument("--source",     type=str,   default="center",
                        help="Source node for path planning")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-log",     action="store_true")
    parser.add_argument("--max-frames", type=int,   default=None)
    return parser.parse_args()


def draw_zones(frame: np.ndarray, zone_manager: ZoneManager,
               predictions: dict) -> np.ndarray:
    for zone in zone_manager.zones:
        x1, y1, x2, y2 = zone.bbox

        # Use LSTM prediction status if available
        pred_status = predictions.get(zone.id, {}).get("status",
                                                        zone.current_status)
        color = STATUS_COLORS.get(pred_status, (0, 255, 0))

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        conf    = predictions.get(zone.id, {}).get("confidence")
        conf_str = f" {conf:.0%}" if conf else ""
        label   = f"{zone.name} | {zone.current_count}/{zone.capacity} | {pred_status}{conf_str}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        
        # Ensure label doesn't render off-screen at the top
        label_y = max(y1, th + 8)
        
        cv2.rectangle(frame, (x1, label_y - th - 8), (x1 + tw + 6, label_y), color, -1)
        cv2.putText(frame, label, (x1 + 3, label_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


def run_pipeline(args):
    if not os.path.exists(args.video):
        logger.error(f"Video not found: {args.video}")
        sys.exit(1)

    # ── Open video ────────────────────────────
    cap          = cv2.VideoCapture(args.video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"Video: {os.path.basename(args.video)} | "
                f"{width}x{height} | {source_fps:.0f}fps | {total_frames} frames")

    # ── Initialize all components ─────────────
    logger.info("Initializing pipeline components...")
    detector     = PersonDetector(confidence=args.conf)
    tracker      = PersonTracker()
    zone_manager = ZoneManager()
    zone_manager.load()
    extractor    = FeatureExtractor(zone_manager, (height, width),
                                    log_csv=not args.no_log)
    predictor    = CongestionInference(zone_manager)
    predictor.load()
    graph        = VenueGraph()
    graph.load()
    router       = None   # initialized after first frame (needs frame_shape)

    if not graph.loaded:
        logger.warning("Venue graph not loaded — path planning disabled")

    box_ann, label_ann = build_annotators()

    # ── Load floor map ─────────────────────────
    floormap = None
    if os.path.exists(FLOORMAP_PATH_LOCAL):
        floormap = cv2.imread(FLOORMAP_PATH_LOCAL)
        logger.info(f"Floor map loaded: {FLOORMAP_PATH_LOCAL}")
    else:
        logger.warning("Floor map not found — path overlay disabled")

    # ── Output writer ─────────────────────────
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(
        args.output, fourcc, source_fps / args.skip, (width, height)
    )

    logger.info(f"Output: {args.output}")
    logger.info("Press 'q' to quit\n")

    # ── Processing loop ───────────────────────
    frame_idx    = 0
    processed    = 0
    fps_history  = []
    last_frame   = None
    predictions  = {}
    path_result  = None
    start_total  = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.max_frames and frame_idx >= args.max_frames:
            break

        frame_idx += 1

        if frame_idx % args.skip != 0:
            if last_frame is not None:
                out_writer.write(last_frame)
            continue

        t0 = time.time()

        # ── Init PersonRouter on first frame ──
        if router is None:
            router = PersonRouter(zone_manager, (height, width))
            logger.info("PersonRouter initialized")

        # ── Core pipeline ─────────────────────
        detections = detector.detect(frame)
        tracked    = tracker.update(detections)
        zone_manager.update_counts(tracked)
        features   = extractor.update(tracked, frame_idx)

        # ── Congestion prediction ─────────────
        predictions = predictor.predict(features)

        # ── Per-person routing ───────────────
        routing = router.assign(tracked, predictions) if router else {}

        # ── Path planning ─────────────────────
        if graph.loaded:
            graph.update_weights(predictions)
            path_result = graph.find_safest_path(args.source)

        fps = 1.0 / max(time.time() - t0, 1e-6)
        fps_history.append(fps)
        avg_fps = sum(fps_history[-30:]) / len(fps_history[-30:])

        # ── Annotate video frame ──────────────
        annotated = frame.copy()
        annotated = draw_zones(annotated, zone_manager, predictions)
        annotated = draw_detections(
            annotated, tracked, box_ann, label_ann
        )
        # Draw per-person exit direction on each bounding box
        if routing:
            annotated = draw_person_routing(annotated, tracked, routing)
        annotated = draw_stats_panel(
            annotated, tracker.active_count,
            avg_fps, frame_idx, tracker.total_count
        )

        # Draw path arrows on video (Removed global indicator as requested)
        # if path_result:
        #     annotated = draw_path_on_frame(
        #         annotated, path_result, zone_manager, predictions
        #     )

        if zone_manager.any_critical():
            critical = [z.name for z in zone_manager.zones
                        if z.current_status == "CRITICAL"]
            annotated = draw_alert_banner(
                annotated,
                f"EVACUATION ALERT — Congestion at: {', '.join(critical)}"
            )

        last_frame = annotated
        processed += 1
        out_writer.write(annotated)

        # ── Show floor map in separate window ──
        if not args.no_display and floormap is not None and graph.loaded:
            fm_display = draw_path_on_floormap(
                floormap, path_result, graph, predictions, zone_manager
            )
            cv2.imshow("Floor Map — Evacuation Path", fm_display)

        if not args.no_display:
            cv2.imshow("Evacuation System — Live", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if processed % 50 == 0:
            path_str = (
                " → ".join(path_result["path"]) if path_result else "N/A"
            )
            logger.info(
                f"Frame {frame_idx}/{total_frames} | "
                f"Persons: {tracker.active_count} | "
                f"FPS: {avg_fps:.1f} | "
                f"Path: {path_str}"
            )

    # ── Cleanup ───────────────────────────────
    cap.release()
    out_writer.release()
    extractor.close()
    if not args.no_display:
        cv2.destroyAllWindows()

    total_time = time.time() - start_total

    print("\n" + "=" * 55)
    print("  PIPELINE RESULTS — Phase 3–7")
    print("=" * 55)
    print(f"  Input          : {os.path.basename(args.video)}")
    print(f"  Frames total   : {frame_idx}")
    print(f"  Processed      : {processed}")
    print(f"  Avg FPS        : {processed/total_time:.2f}")
    print(f"  Unique persons : {tracker.total_count}")
    print(f"  Exit zones     : {len(zone_manager)}")
    print(f"  Graph nodes    : {graph.G.number_of_nodes() if graph.loaded else 'N/A'}")
    if path_result:
        print(f"  Last path      : {' → '.join(path_result['path'])}")
        print(f"  Path status    : {path_result['path_status']}")
    print(f"  Output         : {args.output}")
    print("=" * 55)


if __name__ == "__main__":
    run_pipeline(parse_args())