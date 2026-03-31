# ─────────────────────────────────────────────
# utils/visualizer.py
# Drawing utilities for annotating video frames
# and floor map with bounding boxes, track IDs,
# zone status, stats, alerts, and paths.
# ─────────────────────────────────────────────

import cv2
import numpy as np
import supervision as sv
from typing import List, Optional

# ── Annotation colors ─────────────────────────
COLOR_BOX      = sv.Color(r=0,   g=220, b=80)
COLOR_STATS_BG = (0, 0, 0)
COLOR_STATS_FG = (255, 255, 255)

# ── Path / status colors (BGR) ────────────────
STATUS_BGR = {
    "SAFE"     : (0,   200, 0),
    "MODERATE" : (0,   165, 255),
    "CRITICAL" : (0,   0,   220),
}


def build_annotators():
    """Build supervision annotators for bounding boxes and labels."""
    box_annotator = sv.BoxAnnotator(
        color     = COLOR_BOX,
        thickness = 2,
    )
    label_annotator = sv.LabelAnnotator(
        color          = COLOR_BOX,
        text_color     = sv.Color.WHITE,
        text_scale     = 0.4,
        text_thickness = 1,
        text_padding   = 3,
    )
    return box_annotator, label_annotator


def draw_detections(
    frame, detections, box_annotator, label_annotator
) -> np.ndarray:
    """Draw bounding boxes and track ID labels on frame."""
    if len(detections) == 0:
        return frame

    labels = []
    for i in range(len(detections)):
        track_id = (int(detections.tracker_id[i])
                    if detections.tracker_id is not None else i)
        conf     = (float(detections.confidence[i])
                    if detections.confidence is not None else 0.0)
        labels.append(f"#{track_id} {conf:.2f}")

    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(
        scene=frame, detections=detections, labels=labels
    )
    return frame


def draw_stats_panel(
    frame, person_count, fps, frame_num, total_unique=0
) -> np.ndarray:
    """Draw semi-transparent stats panel in top-left corner."""
    panel_w = 260
    panel_h = 110
    margin  = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (margin, margin),
                  (margin + panel_w, margin + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    font   = cv2.FONT_HERSHEY_SIMPLEX
    fs     = 0.55
    x      = margin + 10
    y      = margin + 22
    lh     = 24

    for line in [
        f"People (active) : {person_count}",
        f"People (total)  : {total_unique}",
        f"FPS             : {fps:.1f}",
        f"Frame           : {frame_num}",
    ]:
        cv2.putText(frame, line, (x, y), font, fs,
                    (255, 255, 255), 1, cv2.LINE_AA)
        y += lh

    return frame


def draw_alert_banner(frame: np.ndarray, message: str) -> np.ndarray:
    """Draw red alert banner at bottom of frame."""
    h, w    = frame.shape[:2]
    banner_h = 45

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - banner_h), (w, h), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, f"  {message}", (20, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA)
    return frame


# ══════════════════════════════════════════════
# PATH DRAWING — VIDEO FRAME
# ══════════════════════════════════════════════

def draw_path_on_frame(
    frame      : np.ndarray,
    path_result: Optional[dict],
    zone_manager,
    predictions: dict,
) -> np.ndarray:
    """
    Draw directional evacuation arrows on the live video frame.

    Shows:
    - Arrow pointing toward recommended exit
    - Recommended exit label
    - Path status color (green/orange/red)

    Args:
        frame       : BGR video frame
        path_result : Output from VenueGraph.find_safest_path()
        zone_manager: ZoneManager instance
        predictions : CongestionInference.predict() output
    """
    if path_result is None:
        return frame

    h, w     = frame.shape[:2]
    path     = path_result.get("path", [])
    status   = path_result.get("path_status", "SAFE")
    color    = STATUS_BGR.get(status, (0, 200, 0))
    target   = path_result.get("target", "")
    cost     = path_result.get("cost", 0)

    if not path or len(path) < 2:
        return frame

    # ── Direction arrow ───────────────────────
    # Determine which direction to point based on target
    if "left" in target.lower():
        arrow_start = (w // 2, h // 2)
        arrow_end   = (w // 4, h // 2)
        dir_label   = "← EXIT LEFT"
    elif "right" in target.lower():
        arrow_start = (w // 2, h // 2)
        arrow_end   = (3 * w // 4, h // 2)
        dir_label   = "EXIT RIGHT →"
    else:
        arrow_start = (w // 2, h // 2)
        arrow_end   = (w // 2, h // 4)
        dir_label   = "↑ EXIT"

    # Draw thick directional arrow
    cv2.arrowedLine(frame, arrow_start, arrow_end,
                    color, 6, tipLength=0.3)

    # ── Path info panel (bottom-left) ─────────
    panel_x, panel_y = 10, h - 120
    panel_w, panel_h = 320, 110

    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.rectangle(frame,
                  (panel_x, panel_y),
                  (panel_x + panel_w, panel_y + panel_h),
                  color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    lh   = 22
    x    = panel_x + 10
    y    = panel_y + 22

    cv2.putText(frame, "RECOMMENDED EVACUATION",
                (x, y), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    y += lh

    cv2.putText(frame, f"  Route  : {' -> '.join(path)}",
                (x, y), font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    y += lh

    cv2.putText(frame, f"  Target : {path_result.get('target', '')}",
                (x, y), font, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
    y += lh

    cv2.putText(frame, f"  Status : {status}",
                (x, y), font, 0.40, color, 1, cv2.LINE_AA)
    y += lh

    cv2.putText(frame, f"  Cost   : {cost:.1f}",
                (x, y), font, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

    # ── Direction label near arrow ─────────────
    (tw, th), _ = cv2.getTextSize(
        dir_label, font, 0.7, 2
    )
    lx = (arrow_start[0] + arrow_end[0]) // 2 - tw // 2
    ly = arrow_end[1] - 20

    cv2.putText(frame, dir_label, (lx, ly),
                font, 0.7, color, 2, cv2.LINE_AA)

    return frame


# ══════════════════════════════════════════════
# PATH DRAWING — FLOOR MAP
# ══════════════════════════════════════════════

def draw_path_on_floormap(
    floormap    : np.ndarray,
    path_result : Optional[dict],
    graph,
    predictions : dict,
    zone_manager,
) -> np.ndarray:
    """
    Draw the recommended evacuation path on the floor map image.

    Shows:
    - Highlighted path edges (colored by status)
    - Animated pulsing on recommended exit
    - Zone congestion status overlaid on zone nodes
    - All exits with their current status color

    Args:
        floormap    : Floor map PNG as numpy array
        path_result : Output from VenueGraph.find_safest_path()
        graph       : VenueGraph instance
        predictions : CongestionInference.predict() output
        zone_manager: ZoneManager instance
    """
    if floormap is None:
        return floormap

    canvas = floormap.copy()

    if not graph.loaded:
        return canvas

    # ── Draw all edges (dim) ──────────────────
    for u, v, data in graph.G.edges(data=True):
        pos_u = graph.get_node_pos(u)
        pos_v = graph.get_node_pos(v)
        if pos_u and pos_v:
            edge_status = data.get("status", "SAFE")
            color       = STATUS_BGR.get(edge_status, (150, 150, 150))
            # Dim non-path edges
            dim_color = tuple(int(c * 0.35) for c in color)
            cv2.line(canvas, tuple(pos_u), tuple(pos_v),
                     dim_color, 2, cv2.LINE_AA)

    # ── Highlight recommended path ────────────
    if path_result:
        path    = path_result.get("path", [])
        status  = path_result.get("path_status", "SAFE")
        color   = STATUS_BGR.get(status, (0, 200, 0))

        for i in range(len(path) - 1):
            pos_u = graph.get_node_pos(path[i])
            pos_v = graph.get_node_pos(path[i + 1])
            if pos_u and pos_v:
                # Thick highlighted path
                cv2.line(canvas, tuple(pos_u), tuple(pos_v),
                         color, 5, cv2.LINE_AA)
                # Directional arrow mid-segment
                mx = (pos_u[0] + pos_v[0]) // 2
                my = (pos_u[1] + pos_v[1]) // 2
                dx = pos_v[0] - pos_u[0]
                dy = pos_v[1] - pos_u[1]
                dist = max(np.sqrt(dx**2 + dy**2), 1)
                ax = int(mx + (dx / dist) * 12)
                ay = int(my + (dy / dist) * 12)
                cv2.arrowedLine(canvas,
                                (mx - int(dx/dist*8), my - int(dy/dist*8)),
                                (ax, ay),
                                color, 2, tipLength=0.5)

    # ── Draw all nodes ────────────────────────
    node_type_colors = {
        "exit"     : (0,   180, 0),
        "zone"     : (0,   140, 220),
        "corridor" : (180, 120, 40),
    }
    zone_nodes = graph.get_zone_nodes()

    for nid, ndata in graph.nodes_data.items():
        pos   = ndata.get("pos")
        ntype = ndata.get("type", "corridor")
        if not pos:
            continue

        # Determine color based on congestion for exit/zone nodes
        zone_id    = ndata.get("zone_id")
        node_color = node_type_colors.get(ntype, (150, 150, 150))

        if zone_id and zone_id in predictions:
            status     = predictions[zone_id].get("status", "SAFE")
            node_color = STATUS_BGR.get(status, node_color)

        # Highlight nodes on recommended path
        on_path = (path_result and nid in path_result.get("path", []))
        r       = 20 if (ntype == "exit" and on_path) else \
                  18 if ntype == "exit" else 14

        cv2.circle(canvas, tuple(pos), r, node_color, -1)
        cv2.circle(canvas, tuple(pos), r,
                   (255, 255, 255) if on_path else (80, 80, 80),
                   3 if on_path else 2)

        # Node label
        short_label = nid.replace("corridor_", "C").replace(
            "zone_", "Z").replace("exit_", "E").upper()[:4]
        (tw, th), _ = cv2.getTextSize(
            short_label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1
        )
        cv2.putText(canvas, short_label,
                    (pos[0] - tw//2, pos[1] + th//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (255, 255, 255), 1, cv2.LINE_AA)

        # Zone congestion label below node
        if zone_id and zone_id in predictions:
            status = predictions[zone_id].get("status", "SAFE")
            conf   = predictions[zone_id].get("confidence")
            sub    = status[:3]
            if conf:
                sub += f" {conf:.0%}"
            cv2.putText(canvas, sub,
                        (pos[0] - 18, pos[1] + r + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        node_color, 1, cv2.LINE_AA)

    # ── Path summary panel ────────────────────
    if path_result:
        path   = path_result.get("path", [])
        status = path_result.get("path_status", "SAFE")
        color  = STATUS_BGR.get(status, (0, 200, 0))
        target = path_result.get("target", "")
        cost   = path_result.get("cost", 0)
        hops   = path_result.get("n_hops", 0)

        # Panel at bottom
        ph, pw_map = canvas.shape[:2]
        py = ph - 55
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, py), (pw_map, ph), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        txt = (f"  SAFEST ROUTE: {' -> '.join(path)}"
               f"  |  Target: {target}"
               f"  |  Status: {status}"
               f"  |  Cost: {cost:.1f}"
               f"  |  Hops: {hops}")
        cv2.putText(canvas, txt, (10, py + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    color, 1, cv2.LINE_AA)

    return canvas


def draw_person_routing(
    frame   : np.ndarray,
    detections,
    routing : dict,
) -> np.ndarray:
    """
    Draw per-person exit direction labels above each bounding box.

    For each tracked person shows:
        #<track_id>  ← Exit A
    Color reflects the assigned exit's congestion status.

    Args:
        frame      : BGR numpy array
        detections : sv.Detections with tracker_id
        routing    : Dict from PersonRouter.assign()
                     {track_id: {"label": "← Exit A", "color": ...}}

    Returns:
        Annotated frame
    """
    if len(detections) == 0 or detections.tracker_id is None:
        return frame

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness  = 1

    for i, bbox in enumerate(detections.xyxy):
        track_id = int(detections.tracker_id[i])
        info     = routing.get(track_id)

        if info is None:
            continue

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        color           = info["color"]
        direction_label = info["label"]
        full_label      = f"#{track_id} {direction_label}"

        # Measure text size
        (tw, th), baseline = cv2.getTextSize(
            full_label, font, font_scale, thickness
        )

        # Position label above bounding box
        label_x  = max(x1, 0)
        label_y  = max(y1 - 6, th + 4)

        # Draw background rectangle
        cv2.rectangle(
            frame,
            (label_x, label_y - th - 4),
            (label_x + tw + 6, label_y + 2),
            color, -1
        )

        # Draw text
        cv2.putText(
            frame, full_label,
            (label_x + 3, label_y - 2),
            font, font_scale,
            (0, 0, 0), thickness, cv2.LINE_AA
        )

    return frame