# ─────────────────────────────────────────────
# tools/define_graph.py
# Interactive venue graph builder tool.
# Click to place nodes, click between nodes
# to draw edges. Saves to data/venue_graph.json
#
# Controls:
#   Left click          : Place new node
#   Right click node    : Delete node
#   'e' then click node : Connect edge to last node
#   's'                 : Save graph
#   'r'                 : Remove last node
#   'q'                 : Quit
#
# Usage:
#   python tools/define_graph.py
#   python tools/define_graph.py --load   (edit existing)
# ─────────────────────────────────────────────

import cv2
import json
import argparse
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FLOORMAP_PATH
from src.pathplanning import GRAPH_PATH

FLOORMAP_PATH_LOCAL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "static", "images", "floormap.png"
)

NODE_TYPES  = ["corridor", "zone", "exit"]
NODE_COLORS = {
    "corridor" : (180, 120, 40),
    "zone"     : (0,   140, 220),
    "exit"     : (0,   180, 0),
}


class GraphBuilder:
    def __init__(self, base_image: np.ndarray, existing: dict = None):
        self.base      = base_image.copy()
        self.display   = base_image.copy()
        self.nodes     = existing.get("nodes", []) if existing else []
        self.edges     = existing.get("edges", []) if existing else []
        self.selected  = None   # node id being connected
        self.edge_mode = False

    def _find_node_at(self, x, y, radius=18):
        for node in self.nodes:
            px, py = node["pos"]
            if (px - x)**2 + (py - y)**2 <= radius**2:
                return node
        return None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked = self._find_node_at(x, y)

            if self.edge_mode and self.selected:
                # Connect edge
                if clicked and clicked["id"] != self.selected["id"]:
                    # Check edge doesn't already exist
                    exists = any(
                        (e["from"] == self.selected["id"] and e["to"] == clicked["id"]) or
                        (e["from"] == clicked["id"] and e["to"] == self.selected["id"])
                        for e in self.edges
                    )
                    if not exists:
                        self.edges.append({
                            "from"        : self.selected["id"],
                            "to"          : clicked["id"],
                            "base_weight" : 1.0,
                        })
                        print(f"  ✅ Edge: {self.selected['id']} ↔ {clicked['id']}")
                self.selected  = None
                self.edge_mode = False

            elif clicked:
                # Select node for edge
                self.selected  = clicked
                self.edge_mode = True
                print(f"  🔗 Selected: {clicked['id']} — now click another node to connect")

            else:
                # Place new node
                node_id = f"node_{len(self.nodes)+1}"
                print(f"\n  New node at ({x}, {y})")
                ntype = input(f"  Type [corridor/zone/exit] (default: corridor): ").strip()
                if ntype not in NODE_TYPES:
                    ntype = "corridor"

                label = input(f"  Label (default: {node_id}): ").strip()
                if not label:
                    label = node_id

                nid = label.lower().replace(" ", "_")

                node = {"id": nid, "label": label, "type": ntype, "pos": [x, y]}

                if ntype in ["zone", "exit"]:
                    try:
                        zone_id = int(input("  Zone ID (int): ").strip())
                        node["zone_id"] = zone_id
                    except ValueError:
                        pass

                self.nodes.append(node)
                print(f"  ✅ Added: {nid} ({ntype})")

            self._refresh()

        elif event == cv2.EVENT_RBUTTONDOWN:
            clicked = self._find_node_at(x, y)
            if clicked:
                self.nodes  = [n for n in self.nodes if n["id"] != clicked["id"]]
                self.edges  = [e for e in self.edges
                               if e["from"] != clicked["id"] and e["to"] != clicked["id"]]
                print(f"  🗑️  Removed: {clicked['id']}")
                self._refresh()

    def _refresh(self):
        self.display = self.base.copy()

        # Draw edges
        for edge in self.edges:
            n1 = next((n for n in self.nodes if n["id"] == edge["from"]), None)
            n2 = next((n for n in self.nodes if n["id"] == edge["to"]),   None)
            if n1 and n2:
                cv2.line(self.display,
                         tuple(n1["pos"]), tuple(n2["pos"]),
                         (100, 100, 200), 2)

        # Draw nodes
        for node in self.nodes:
            color = NODE_COLORS.get(node["type"], (150, 150, 150))
            r     = 18 if node["type"] == "exit" else 14
            is_sel = self.selected and node["id"] == self.selected["id"]

            cv2.circle(self.display, tuple(node["pos"]), r, color, -1)
            cv2.circle(self.display, tuple(node["pos"]), r,
                       (255, 255, 0) if is_sel else (255, 255, 255),
                       3 if is_sel else 2)

            short = node["id"][:4].upper()
            (tw, th), _ = cv2.getTextSize(
                short, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1
            )
            cv2.putText(self.display, short,
                        (node["pos"][0] - tw//2, node["pos"][1] + th//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 255, 255), 1, cv2.LINE_AA)

            # Label below
            cv2.putText(self.display, node["label"],
                        (node["pos"][0] - 30, node["pos"][1] + r + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30,
                        (50, 50, 50), 1, cv2.LINE_AA)

        # Instructions
        h = self.display.shape[0]
        for i, txt in enumerate([
            "Left-click empty: Add node",
            "Left-click node : Select for edge",
            "Right-click node: Delete node",
            "S: Save | R: Remove last | Q: Quit",
            f"Nodes: {len(self.nodes)} | Edges: {len(self.edges)}",
        ]):
            cv2.putText(self.display, txt,
                        (10, h - 100 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (50, 50, 200), 1, cv2.LINE_AA)

    def run(self):
        win = "Define Venue Graph"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 1000, 600)
        cv2.setMouseCallback(win, self.mouse_callback)
        self._refresh()

        print("\n" + "="*50)
        print("  VENUE GRAPH BUILDER")
        print("="*50)
        print("  Left-click empty space : Add node")
        print("  Left-click a node      : Select for edge connection")
        print("  Left-click another node: Draw edge between them")
        print("  Right-click a node     : Delete it")
        print("  S : Save  | R : Remove last | Q : Quit")
        print("="*50 + "\n")

        while True:
            cv2.imshow(win, self.display)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('s'):
                cv2.destroyAllWindows()
                return {"nodes": self.nodes, "edges": self.edges}

            elif key == ord('r'):
                if self.nodes:
                    removed = self.nodes.pop()
                    self.edges = [e for e in self.edges
                                  if e["from"] != removed["id"]
                                  and e["to"] != removed["id"]]
                    print(f"  ↩️  Removed: {removed['id']}")
                    self._refresh()

            elif key == ord('q'):
                cv2.destroyAllWindows()
                return {}

            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", action="store_true",
                        help="Load and edit existing graph")
    parser.add_argument("--floormap", type=str,
                        default=FLOORMAP_PATH_LOCAL)
    args = parser.parse_args()

    if not os.path.exists(args.floormap):
        print(f"❌ Floor map not found: {args.floormap}")
        sys.exit(1)

    floormap = cv2.imread(args.floormap)

    existing = {}
    if args.load and os.path.exists(GRAPH_PATH):
        with open(GRAPH_PATH) as f:
            existing = json.load(f)
        print(f"✅ Loaded existing graph: {len(existing['nodes'])} nodes")

    builder = GraphBuilder(floormap, existing)
    result  = builder.run()

    if not result or not result.get("nodes"):
        print("No graph saved.")
        return

    os.makedirs(os.path.dirname(GRAPH_PATH), exist_ok=True)
    with open(GRAPH_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n✅ Graph saved: {GRAPH_PATH}")
    print(f"   Nodes : {len(result['nodes'])}")
    print(f"   Edges : {len(result['edges'])}")
    for n in result["nodes"]:
        print(f"   [{n['type']:9}] {n['id']} @ {n['pos']}")


if __name__ == "__main__":
    main()