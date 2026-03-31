# ─────────────────────────────────────────────
# tests/test_phase6.py
# Phase 6 completion test.
# Verifies model files, loading, inference,
# and integration with the feature pipeline.
#
# Usage:
#   python tests/test_phase6.py
# ─────────────────────────────────────────────

import os
import sys
import json
import pickle
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_DIR, LSTM_WEIGHTS, LSTM_SEQUENCE_LENGTH

SCALER_PATH = os.path.join(MODEL_DIR, "lstm_scaler.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "lstm_config.json")

PASS = "✅"
FAIL = "❌"

results = []

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    results.append((status, name, detail))
    print(f"  {status}  {name}" + (f" — {detail}" if detail else ""))
    return condition


print("\n" + "="*55)
print("  PHASE 6 — COMPLETION TEST")
print("="*55)


# ── Test 1: File existence ─────────────────────
print("\n[1] Model Files")
check("lstm_congestion.pt exists",
      os.path.exists(LSTM_WEIGHTS),
      f"{os.path.getsize(LSTM_WEIGHTS)/1024/1024:.1f} MB"
      if os.path.exists(LSTM_WEIGHTS) else "MISSING")

check("lstm_scaler.pkl exists",
      os.path.exists(SCALER_PATH),
      f"{os.path.getsize(SCALER_PATH)/1024:.1f} KB"
      if os.path.exists(SCALER_PATH) else "MISSING")

check("lstm_config.json exists",
      os.path.exists(CONFIG_PATH),
      f"{os.path.getsize(CONFIG_PATH)/1024:.1f} KB"
      if os.path.exists(CONFIG_PATH) else "MISSING")

if not all(os.path.exists(p) for p in [LSTM_WEIGHTS, SCALER_PATH, CONFIG_PATH]):
    print("\n  ❌ Cannot continue — model files missing.")
    print("     Download from Drive and place in models/ folder.")
    sys.exit(1)


# ── Test 2: Config validity ────────────────────
print("\n[2] Config Validation")
with open(CONFIG_PATH) as f:
    config = json.load(f)

for key in ["seq_len", "n_features", "feature_cols", "status_map",
            "zone_names", "hidden_size", "n_layers", "dropout"]:
    check(f"config has '{key}'", key in config,
          str(config.get(key, "MISSING")))

check("n_features == 4",   config.get("n_features") == 4,
      f"got {config.get('n_features')}")
check("seq_len == 30",     config.get("seq_len") == 30,
      f"got {config.get('seq_len')}")
check("3 status classes",  len(config.get("status_map", {})) == 3,
      str(config.get("status_map")))


# ── Test 3: Scaler validity ────────────────────
print("\n[3] Scaler Validation")
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

check("Scaler is StandardScaler",
      hasattr(scaler, "transform"), type(scaler).__name__)
check("Scaler fitted on 3 features",
      hasattr(scaler, "mean_") and len(scaler.mean_) == 3,
      f"mean={scaler.mean_.round(3).tolist()}"
      if hasattr(scaler, "mean_") else "not fitted")

try:
    scaled = scaler.transform(np.array([[5.0, 2.5, 0.01]], dtype=np.float32))
    check("Scaler.transform() works", scaled.shape == (1, 3),
          f"output shape: {scaled.shape}")
except Exception as e:
    check("Scaler.transform() works", False, str(e))


# ── Test 4: Model loading ──────────────────────
print("\n[4] Model Loading")
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.Linear(d, 1)
    def forward(self, x):
        w = F.softmax(self.attn(x), dim=1)
        return (w * x).sum(dim=1), w.squeeze(-1)

class CongestionPredictor(nn.Module):
    def __init__(self, n_features, hidden_size, n_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, n_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        d = hidden_size * 2
        self.attention = SelfAttention(d)
        self.dropout   = nn.Dropout(dropout)
        self.fc1       = nn.Linear(d, 64)
        self.fc2       = nn.Linear(64, 3)
    def forward(self, x):
        out, _ = self.lstm(x)
        ctx, w = self.attention(out)
        return self.fc2(F.relu(self.fc1(self.dropout(ctx)))), w

try:
    model = CongestionPredictor(
        n_features  = config["n_features"],
        hidden_size = config["hidden_size"],
        n_layers    = config["n_layers"],
        dropout     = config["dropout"],
    )
    ckpt = torch.load(LSTM_WEIGHTS, map_location="cpu", weights_only=False)
    check("Checkpoint has model_state_dict", "model_state_dict" in ckpt)
    check("Checkpoint has best_val_loss", "best_val_loss" in ckpt,
          f"{ckpt.get('best_val_loss', 0):.4f}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    check("Model weights loaded", True, f"{params:,} parameters")
except Exception as e:
    check("Model loads successfully", False, str(e))
    sys.exit(1)


# ── Test 5: Inference ─────────────────────────
print("\n[5] Inference Test")
seq_len    = config["seq_len"]
n_features = config["n_features"]
statuses   = ["SAFE", "MODERATE", "CRITICAL"]

for label, val in [("zero input (SAFE)", 0.0), ("high input (CRITICAL)", 3.0)]:
    try:
        x      = torch.full((1, seq_len, n_features), val)
        with torch.no_grad():
            logits, _ = model(x)
            probs     = F.softmax(logits, dim=1).numpy()[0]
        check(f"Inference on {label}",
              abs(probs.sum() - 1.0) < 0.01,
              f"pred={statuses[probs.argmax()]} | "
              f"S:{probs[0]:.3f} M:{probs[1]:.3f} C:{probs[2]:.3f}")
    except Exception as e:
        check(f"Inference on {label}", False, str(e))


# ── Test 6: CongestionInference class ─────────
print("\n[6] CongestionInference Integration")
try:
    from src.zones import ZoneManager
    from src.congestion import CongestionInference

    zm = ZoneManager()
    zm.load()

    predictor = CongestionInference(zm)
    loaded    = predictor.load()
    check("CongestionInference.load() succeeds", loaded)

    if loaded:
        check("seq_len matches config",
              predictor.seq_len == config["seq_len"],
              str(predictor.seq_len))
        check("Buffers for all zones",
              len(predictor.buffers) == len(zm.zones),
              f"{len(predictor.buffers)} buffers")

        # Simulate warmup
        dummy = {"zones": [
            {"zone_id": z.id, "zone_name": z.name,
             "count": 3, "avg_speed": 2.5,
             "density": 0.02, "status": "SAFE"}
            for z in zm.zones
        ]}
        for _ in range(config["seq_len"]):
            predictor.predict(dummy)

        check("Predictor ready after warmup", predictor.is_ready)

        preds = predictor.predict(dummy)
        for zone_id, pred in preds.items():
            check(f"Zone {zone_id} returns valid status",
                  pred.get("status") in ["SAFE", "MODERATE", "CRITICAL"],
                  pred.get("status"))
            if pred.get("confidence") is not None:
                check(f"Zone {zone_id} confidence in [0,1]",
                      0.0 <= pred["confidence"] <= 1.0,
                      f"{pred['confidence']:.3f}")
except Exception as e:
    check("CongestionInference integration", False, str(e))


# ── Test 7: Model quality ──────────────────────
print("\n[7] Model Quality")
if "best_val_loss" in ckpt:
    v = ckpt["best_val_loss"]
    check("Val loss < 0.20 (good)",  v < 0.20, f"{v:.4f}")
    check("Val loss < 0.15 (great)", v < 0.15, f"{v:.4f}")

if "history" in ckpt:
    best_acc = max(ckpt["history"].get("val_acc", [0]))
    check("Val accuracy > 90%", best_acc > 90, f"{best_acc:.1f}%")
    check("Val accuracy > 95%", best_acc > 95, f"{best_acc:.1f}%")


# ── Summary ────────────────────────────────────
print("\n" + "="*55)
passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)
print(f"  RESULTS: {passed}/{len(results)} passed | {failed} failed")
if failed == 0:
    print("  🎉 Phase 6 complete — ready for Phase 7!")
elif failed <= 2:
    print("  ⚠️  Minor issues — check failed tests above")
else:
    print("  ❌ Phase 6 incomplete — fix issues before Phase 7")
print("="*55 + "\n")
sys.exit(0 if failed == 0 else 1)