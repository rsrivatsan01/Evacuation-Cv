# ─────────────────────────────────────────────
# src/pathplanning.py
# Graph-based evacuation path planner.
# Loads venue graph, updates edge weights from
# congestion predictions, runs Dijkstra to find
# safest evacuation route to nearest safe exit.
# ─────────────────────────────────────────────

import os
import sys
import json
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.zones import STATUS_SAFE, STATUS_MODERATE, STATUS_CRITICAL
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Graph file path ───────────────────────────
GRAPH_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "venue_graph.json"
)

# ── Congestion weight multipliers ─────────────
# Higher = more expensive = avoid this path
CONGESTION_WEIGHTS = {
    STATUS_SAFE     : 1.0,    # Preferred
    STATUS_MODERATE : 3.5,    # Avoid if possible
    STATUS_CRITICAL : 10.0,   # Strongly avoid
}

# ── Node type base costs ──────────────────────
NODE_TYPE_COST = {
    "exit"     : 0.5,    # Low cost — we want to reach exits
    "zone"     : 1.0,    # Normal
    "corridor" : 1.0,    # Normal
}

# ── Path status colors (BGR for OpenCV) ───────
PATH_COLORS = {
    STATUS_SAFE     : (0,   200, 0),    # Green
    STATUS_MODERATE : (0,   165, 255),  # Orange
    STATUS_CRITICAL : (0,   0,   220),  # Red
}


class VenueGraph:
    """
    Represents the venue as a weighted undirected graph.

    Nodes: exits, zones, corridors
    Edges: physical connections between nodes
    Weights: updated each frame based on congestion predictions

    Usage:
        graph = VenueGraph()
        graph.load()
        graph.update_weights(predictions)
        result = graph.find_safest_path("center", "exit")
    """

    def __init__(self, graph_path: str = GRAPH_PATH):
        self.graph_path = graph_path
        self.G          = nx.Graph()
        self.nodes_data : Dict[str, dict] = {}
        self.loaded     = False

    def load(self) -> bool:
        """
        Load venue graph from JSON file.
        Returns True if successful.
        """
        if not os.path.exists(self.graph_path):
            logger.error(f"Venue graph not found: {self.graph_path}")
            return False

        try:
            with open(self.graph_path) as f:
                data = json.load(f)

            # ── Add nodes ─────────────────────────
            for node in data["nodes"]:
                nid = node["id"]
                self.G.add_node(nid, **node)
                self.nodes_data[nid] = node

            # ── Add edges (bidirectional) ──────────
            for edge in data["edges"]:
                self.G.add_edge(
                    edge["from"], edge["to"],
                    base_weight = edge.get("base_weight", 1.0),
                    weight      = edge.get("base_weight", 1.0),
                )

            self.loaded = True
            logger.info(
                f"Venue graph loaded: {self.graph_path}\n"
                f"  Nodes : {self.G.number_of_nodes()} "
                f"({', '.join(self.nodes_data.keys())})\n"
                f"  Edges : {self.G.number_of_edges()}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load venue graph: {e}")
            return False

    def get_exit_nodes(self) -> List[str]:
        """Return list of all exit node IDs."""
        return [
            nid for nid, data in self.nodes_data.items()
            if data.get("type") == "exit"
        ]

    def get_zone_nodes(self) -> Dict[int, str]:
        """Return dict mapping zone_id → node_id for zone nodes."""
        result = {}
        for nid, data in self.nodes_data.items():
            if "zone_id" in data:
                result[data["zone_id"]] = nid
        return result

    def update_weights(self, predictions: Dict[int, dict]):
        """
        Update edge weights based on current congestion predictions.
        Called every frame after LSTM inference.

        Args:
            predictions : Dict from CongestionInference.predict()
                          {zone_id: {"status": "SAFE", ...}}
        """
        # Build zone_id → status mapping
        zone_status = {}
        for zone_id, pred in predictions.items():
            zone_status[zone_id] = pred.get("status", STATUS_SAFE)

        # Get zone node mapping
        zone_nodes = self.get_zone_nodes()

        # Update edge weights for all edges
        for u, v, data in self.G.edges(data=True):
            base_w = data["base_weight"]

            # Check if either endpoint is a zone/exit node
            u_data = self.nodes_data.get(u, {})
            v_data = self.nodes_data.get(v, {})

            # Find congestion multiplier for this edge
            multiplier = 1.0
            for node_data in [u_data, v_data]:
                zone_id = node_data.get("zone_id")
                if zone_id and zone_id in zone_status:
                    status     = zone_status[zone_id]
                    node_mult  = CONGESTION_WEIGHTS.get(status, 1.0)
                    multiplier = max(multiplier, node_mult)

            self.G[u][v]["weight"]      = base_w * multiplier
            self.G[u][v]["status"]      = zone_status.get(
                u_data.get("zone_id") or v_data.get("zone_id"), STATUS_SAFE
            )
            self.G[u][v]["multiplier"]  = multiplier

    def find_safest_path(
        self,
        source     : str,
        target     : str = None,
    ) -> Optional[dict]:
        """
        Find the safest path from source to target using Dijkstra.
        If target is None, finds the path to the nearest safe exit.

        Args:
            source : Starting node ID
            target : Target node ID (or None for nearest safe exit)

        Returns:
            Dict with path details or None if no path found.
        """
        if not self.loaded:
            return None

        if source not in self.G:
            logger.warning(f"Source node '{source}' not in graph")
            return None

        exit_nodes = self.get_exit_nodes()

        if target:
            # Find path to specific target
            targets = [target] if target in self.G else []
        else:
            # Find path to nearest safe exit (lowest weight)
            targets = exit_nodes

        if not targets:
            return None

        best_result = None
        best_cost   = float('inf')

        for t in targets:
            if t == source:
                continue
            try:
                path = nx.dijkstra_path(self.G, source, t, weight='weight')
                cost = nx.dijkstra_path_length(self.G, source, t, weight='weight')

                if cost < best_cost:
                    best_cost   = cost
                    best_result = {
                        "path"        : path,
                        "target"      : t,
                        "cost"        : round(cost, 3),
                        "n_hops"      : len(path) - 1,
                        "path_status" : self._path_status(path),
                        "node_positions": [
                            self.nodes_data[n].get("pos", [0, 0])
                            for n in path
                        ],
                    }
            except nx.NetworkXNoPath:
                continue
            except Exception as e:
                logger.warning(f"Path error {source}→{t}: {e}")

        if best_result:
            logger.debug(
                f"Path: {' → '.join(best_result['path'])} "
                f"(cost={best_result['cost']:.2f})"
            )

        return best_result

    def find_all_paths_from(self, source: str) -> List[dict]:
        """
        Find safest paths from source to ALL exits.
        Used to show multiple options on dashboard.
        """
        results = []
        for exit_node in self.get_exit_nodes():
            result = self.find_safest_path(source, exit_node)
            if result:
                results.append(result)
        results.sort(key=lambda r: r["cost"])
        return results

    def _path_status(self, path: List[str]) -> str:
        """
        Determine overall path status based on worst edge along the path.
        """
        worst = STATUS_SAFE
        priority = {STATUS_SAFE: 0, STATUS_MODERATE: 1, STATUS_CRITICAL: 2}

        for i in range(len(path) - 1):
            u, v   = path[i], path[i+1]
            status = self.G[u][v].get("status", STATUS_SAFE)
            if priority.get(status, 0) > priority.get(worst, 0):
                worst = status

        return worst

    def get_node_pos(self, node_id: str) -> Optional[List[int]]:
        """Get pixel position of a node on the floor map."""
        return self.nodes_data.get(node_id, {}).get("pos")

    @property
    def summary(self) -> dict:
        """Returns graph summary for dashboard."""
        zone_nodes = self.get_zone_nodes()
        return {
            "n_nodes"    : self.G.number_of_nodes(),
            "n_edges"    : self.G.number_of_edges(),
            "exit_nodes" : self.get_exit_nodes(),
            "zone_nodes" : zone_nodes,
            "loaded"     : self.loaded,
        }


class PersonRouter:
    """
    Assigns each tracked person an individual evacuation direction
    based on their position in the frame and current exit congestion.

    Logic:
        1. Compute pixel distance from person's feet to each exit zone center
        2. Apply congestion penalty to each exit
        3. Assign the exit with lowest (distance × congestion_multiplier)
        4. Derive direction label: "← Exit A" or "Exit B →"

    Usage:
        router  = PersonRouter(zone_manager, frame_shape)
        routing = router.assign(tracked_detections, predictions)
        # routing = {track_id: {"label": "← Exit A", "color": (0,200,0)}}
    """

    # Direction arrow symbols
    ARROW_LEFT  = "\u2190"   # ←
    ARROW_RIGHT = "\u2192"   # →
    ARROW_UP    = "\u2191"   # ↑
    ARROW_DOWN  = "\u2193"   # ↓

    def __init__(self, zone_manager, frame_shape: tuple):
        """
        Args:
            zone_manager : ZoneManager with loaded exit zones
            frame_shape  : (height, width) of video frame
        """
        self.zone_manager = zone_manager
        self.frame_h      = frame_shape[0]
        self.frame_w      = frame_shape[1]

        # Pre-compute exit zone centers and directions
        self.exit_info = self._build_exit_info()

    def _build_exit_info(self) -> List[dict]:
        """
        Build exit info list from zone manager.
        Determines direction label based on zone position in frame.
        """
        exits = []
        for zone in self.zone_manager.zones:
            cx = (zone.x1 + zone.x2) / 2
            cy = (zone.y1 + zone.y2) / 2

            # Determine primary direction based on zone position
            # relative to frame center
            fc_x = self.frame_w / 2
            fc_y = self.frame_h / 2

            dx = cx - fc_x
            dy = cy - fc_y

            # Pick dominant direction
            if abs(dx) >= abs(dy):
                arrow = self.ARROW_LEFT if dx < 0 else self.ARROW_RIGHT
            else:
                arrow = self.ARROW_UP if dy < 0 else self.ARROW_DOWN

            exits.append({
                "zone_id"  : zone.id,
                "zone_name": zone.name,
                "center_x" : cx,
                "center_y" : cy,
                "arrow"    : arrow,
                "status"   : zone.current_status,
            })

        return exits

    def _congestion_penalty(self, status: str) -> float:
        """
        Returns a distance penalty multiplier based on exit congestion.
        Higher = more expensive = route people away from this exit.
        """
        return CONGESTION_WEIGHTS.get(status, 1.0)

    def assign(
        self,
        detections  : "sv.Detections",
        predictions : Dict[int, dict],
    ) -> Dict[int, dict]:
        """
        Assign each tracked person their best exit direction.

        Args:
            detections  : sv.Detections with tracker_id populated
            predictions : {zone_id: {"status": ...}} from CongestionInference

        Returns:
            Dict mapping track_id → routing info dict:
            {
                "exit_name"  : "Exit A - North",
                "zone_id"    : 1,
                "arrow"      : "←",
                "label"      : "← Exit A",
                "color"      : (0, 200, 0),
                "cost"       : 245.3,
            }
        """
        import supervision as sv

        routing = {}

        if len(detections) == 0 or detections.tracker_id is None:
            return routing

        # Update exit statuses from predictions
        self._refresh_exit_statuses(predictions)

        for i, bbox in enumerate(detections.xyxy):
            track_id = int(detections.tracker_id[i])

            # Person's feet position (bottom-center of bbox)
            px = float((bbox[0] + bbox[2]) / 2)
            py = float(bbox[3])

            # Determine if person is already securely inside an exit zone
            best_exit = None
            best_score = 0.0

            for zone in self.zone_manager.zones:
                if track_id in zone.person_ids:
                    # Lock onto this exit
                    for exit_info in self.exit_info:
                        if exit_info["zone_id"] == zone.id:
                            best_exit = exit_info
                            break
                    break
            
            # If not inside a zone, calculate optimal routing
            if best_exit is None:
                best_score = float('inf')

                for exit_info in self.exit_info:
                    # Euclidean distance to exit center
                    dx       = px - exit_info["center_x"]
                    dy       = py - exit_info["center_y"]
                    distance = np.sqrt(dx**2 + dy**2)

                    # Apply additive congestion penalty
                    penalty = self._congestion_penalty(exit_info["status"])
                    # Subtract 1.0 so SAFE (1.0) adds 0 penalty.
                    # Multiply by 1000 to add a massive pixel distance barrier for congested exits.
                    additive_cost  = (penalty - 1.0) * 1000.0
                    score          = distance + additive_cost

                    if score < best_score:
                        best_score = score
                        best_exit  = exit_info

            if best_exit is None:
                continue

            # Build short exit label (first word after "Exit" or full name)
            name_parts = best_exit["zone_name"].split()
            if len(name_parts) >= 2:
                short_name = " ".join(name_parts[:2])  # e.g. "Exit A"
            else:
                short_name = best_exit["zone_name"]

            arrow = best_exit["arrow"]
            label = f"{arrow} {short_name}" if arrow in [self.ARROW_LEFT, self.ARROW_UP] \
                    else f"{short_name} {arrow}"

            # Color based on exit congestion status
            from src.zones import STATUS_COLORS
            color = STATUS_COLORS.get(best_exit["status"], (0, 200, 0))

            routing[track_id] = {
                "exit_name" : best_exit["zone_name"],
                "zone_id"   : best_exit["zone_id"],
                "arrow"     : arrow,
                "label"     : label,
                "color"     : color,
                "cost"      : round(best_score, 1),
                "status"    : best_exit["status"],
            }

        return routing

    def _refresh_exit_statuses(self, predictions: Dict[int, dict]):
        """Update exit statuses from latest predictions."""
        for exit_info in self.exit_info:
            zone_id = exit_info["zone_id"]
            if zone_id in predictions:
                exit_info["status"] = predictions[zone_id].get(
                    "status", exit_info["status"]
                )
            else:
                # Fall back to zone manager's rule-based status
                for zone in self.zone_manager.zones:
                    if zone.id == zone_id:
                        exit_info["status"] = zone.current_status
                        break