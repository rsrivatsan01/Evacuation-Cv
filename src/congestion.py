# ─────────────────────────────────────────────
# src/congestion.py
# LSTM congestion predictor — inference only.
# Loads trained model + scaler and predicts
# congestion status per zone in real time.
# ─────────────────────────────────────────────

import os
import sys
import json
import pickle
import numpy as np
from collections import deque
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LSTM_WEIGHTS, LSTM_SEQUENCE_LENGTH, MODEL_DIR
from src.zones import ZoneManager, STATUS_SAFE, STATUS_MODERATE, STATUS_CRITICAL
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Model paths ───────────────────────────────
SCALER_PATH = os.path.join(MODEL_DIR, "lstm_scaler.pkl")
CONFIG_PATH = os.path.join(MODEL_DIR, "lstm_config.json")

# ── Status index → string mapping ─────────────
# Must match training notebook's STATUS_MAP:
# {'SAFE': 0, 'MODERATE': 1, 'CRITICAL': 2}
IDX_TO_STATUS = {0: STATUS_SAFE, 1: STATUS_MODERATE, 2: STATUS_CRITICAL}


# ── Model definition (must match training) ────
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        scores  = self.attn(lstm_out)
        weights = F.softmax(scores, dim=1)
        context = (weights * lstm_out).sum(dim=1)
        return context, weights.squeeze(-1)


class CongestionPredictor(nn.Module):
    def __init__(self, n_features=4, hidden_size=64, n_layers=2,
                 n_classes=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size    = n_features,
            hidden_size   = hidden_size,
            num_layers    = n_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if n_layers > 1 else 0.0,
        )
        lstm_out_dim   = hidden_size * 2
        self.attention = SelfAttention(lstm_out_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc1       = nn.Linear(lstm_out_dim, 64)
        self.fc2       = nn.Linear(64, n_classes)

    def forward(self, x):
        lstm_out, _     = self.lstm(x)
        context, attn_w = self.attention(lstm_out)
        out = self.dropout(context)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out, attn_w


class CongestionInference:
    """
    Real-time congestion predictor using the trained BiLSTM model.

    Maintains a per-zone ring buffer of the last SEQ_LEN feature
    vectors and runs inference every frame to predict status.

    Usage:
        predictor = CongestionInference(zone_manager)
        predictor.load()

        # Every frame, after feature extraction:
        predictions = predictor.predict(features_dict)
        # predictions = {zone_id: {"status": "SAFE", "probs": [...]}}
    """

    def __init__(self, zone_manager: ZoneManager):
        self.zone_manager = zone_manager
        self.model        = None
        self.scaler       = None
        self.config       = None
        self.device       = torch.device("cpu")
        self.loaded       = False

        # Use config seq_len after loading — fall back to config.py default
        self.seq_len      = LSTM_SEQUENCE_LENGTH

        # Per-zone feature ring buffer
        # zone_id → deque of feature vectors [count, avg_speed, density, zone_id_norm]
        self.buffers: Dict[int, deque] = {}
        self._init_buffers()

    def _init_buffers(self):
        """Initialize empty ring buffers for each zone."""
        for zone in self.zone_manager.zones:
            self.buffers[zone.id] = deque(maxlen=self.seq_len)

    def load(self) -> bool:
        """
        Load model weights, scaler and config from models/ directory.
        Returns True if successful, False if files not found.
        """
        required = {
            "model"  : LSTM_WEIGHTS,
            "scaler" : SCALER_PATH,
            "config" : CONFIG_PATH,
        }
        missing = [k for k, v in required.items() if not os.path.exists(v)]
        if missing:
            logger.warning(
                f"LSTM model files missing: {missing}\n"
                f"  Run the Phase 6 Colab notebook to train the model.\n"
                f"  Falling back to rule-based congestion status."
            )
            return False

        try:
            # ── Load config ───────────────────────
            with open(CONFIG_PATH) as f:
                self.config = json.load(f)

            # seq_len from training (may differ from config.py default)
            self.seq_len     = self.config.get("seq_len", LSTM_SEQUENCE_LENGTH)
            self.zone_to_idx = self.config.get("zone_to_idx", {})
            self.n_zones     = self.config.get("n_zones", 1)

            # JSON keys are always strings — convert idx_to_status keys to int
            raw_idx = self.config.get("idx_to_status", {})
            self.idx_to_status = {
                int(k): v for k, v in raw_idx.items()
            } if raw_idx else IDX_TO_STATUS

            # Re-init buffers with correct seq_len from training
            for zone in self.zone_manager.zones:
                self.buffers[zone.id] = deque(maxlen=self.seq_len)

            # ── Load scaler ───────────────────────
            with open(SCALER_PATH, "rb") as f:
                self.scaler = pickle.load(f)

            # ── Build + load model ────────────────
            self.model = CongestionPredictor(
                n_features  = self.config.get("n_features", 4),
                hidden_size = self.config.get("hidden_size", 64),
                n_layers    = self.config.get("n_layers", 2),
                n_classes   = 3,
                dropout     = self.config.get("dropout", 0.3),
            ).to(self.device)

            # weights_only=False needed for checkpoint dict with non-tensor data
            checkpoint = torch.load(
                LSTM_WEIGHTS,
                map_location = self.device,
                weights_only = False,
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.loaded = True

            logger.info(
                f"LSTM model loaded: {LSTM_WEIGHTS}\n"
                f"  Seq len    : {self.seq_len}\n"
                f"  Features   : {self.config.get('feature_cols')}\n"
                f"  Zones      : {self.config.get('zone_names')}\n"
                f"  Val loss   : {checkpoint.get('best_val_loss', 'N/A')}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
            self.loaded = False
            return False

    def _get_zone_id_norm(self, zone_name: str) -> float:
        """Normalize zone ID to 0-1 range (same as training)."""
        if not self.config:
            return 0.0
        idx = self.zone_to_idx.get(zone_name, 0)
        return idx / max(self.n_zones - 1, 1)

    def update_buffer(self, zone_id: int, zone_name: str, features: dict):
        """
        Add current frame features to zone's ring buffer.

        Args:
            zone_id   : Zone integer ID
            zone_name : Zone name string (must match training zone names)
            features  : Dict with keys: count, avg_speed, density
        """
        if zone_id not in self.buffers:
            self.buffers[zone_id] = deque(maxlen=self.seq_len)

        zone_id_norm = self._get_zone_id_norm(zone_name)

        # Normalize raw features [count, avg_speed, density] using scaler.
        # Use a DataFrame with column names to match how scaler was fitted
        # during training (fitted on a pandas DataFrame with named columns).
        # This avoids the sklearn "X does not have valid feature names" warning.
        import pandas as pd
        raw = pd.DataFrame([[
            features["count"],
            features["avg_speed"],
            features["density"],
        ]], columns=["count", "avg_speed", "density"])

        if self.scaler is not None:
            try:
                raw_scaled = self.scaler.transform(raw)[0]
            except Exception as e:
                logger.error(f"Scaler transform failed: {e}")
                raw_scaled = np.array([
                    features["count"],
                    features["avg_speed"],
                    features["density"],
                ], dtype=np.float32)
        else:
            raw_scaled = np.array([
                features["count"],
                features["avg_speed"],
                features["density"],
            ], dtype=np.float32)

        feature_vec = np.array([
            raw_scaled[0],   # count (scaled)
            raw_scaled[1],   # avg_speed (scaled)
            raw_scaled[2],   # density (scaled)
            zone_id_norm,    # zone_id (normalized 0-1)
        ], dtype=np.float32)

        self.buffers[zone_id].append(feature_vec)

    def predict_zone(self, zone_id: int) -> Optional[dict]:
        """
        Predict congestion status for a single zone.

        Returns None if buffer not yet full (warming up).
        Returns prediction dict once SEQ_LEN frames are buffered.
        """
        if not self.loaded:
            return None

        buffer = self.buffers.get(zone_id)
        if buffer is None or len(buffer) < self.seq_len:
            return None

        # Build input tensor (1, seq_len, n_features)
        seq    = np.array(list(buffer), dtype=np.float32)
        tensor = torch.tensor(seq).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, attn_w = self.model(tensor)
            probs          = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx       = int(probs.argmax())

        status = self.idx_to_status.get(pred_idx, STATUS_SAFE)

        return {
            "status"     : status,
            "probs"      : {
                STATUS_SAFE     : round(float(probs[0]), 4),
                STATUS_MODERATE : round(float(probs[1]), 4),
                STATUS_CRITICAL : round(float(probs[2]), 4),
            },
            "confidence" : round(float(probs[pred_idx]), 4),
            "pred_idx"   : pred_idx,
        }

    def predict(self, features: dict) -> Dict[int, dict]:
        """
        Run predictions for all zones using current frame features.

        Args:
            features : Output from FeatureExtractor.update()
                       Must have 'zones' key with per-zone feature dicts.

        Returns:
            Dict mapping zone_id → prediction dict.
            Falls back to rule-based status if model not loaded or
            buffer not yet full (first SEQ_LEN frames).
        """
        predictions = {}

        for zf in features.get("zones", []):
            zone_id   = zf["zone_id"]
            zone_name = zf["zone_name"]

            # Update ring buffer
            self.update_buffer(zone_id, zone_name, {
                "count"     : zf["count"],
                "avg_speed" : zf["avg_speed"],
                "density"   : zf["density"],
            })

            if self.loaded:
                pred = self.predict_zone(zone_id)
                if pred is not None:
                    # Enforce SAFE if the zone is empty to prevent
                    # the LSTM from confusing 0.0 speed as frozen traffic.
                    from src.zones import STATUS_SAFE, STATUS_MODERATE, STATUS_CRITICAL
                    if zf["count"] == 0:
                        pred["status"] = STATUS_SAFE
                        pred["probs"] = {STATUS_SAFE: 1.0, STATUS_MODERATE: 0.0, STATUS_CRITICAL: 0.0}
                        pred["confidence"] = 1.0
                        pred["pred_idx"] = 0
                    else:
                        zone_obj = next((z for z in self.zone_manager.zones if z.id == zone_id), None)
                        if zone_obj and zf["count"] > zone_obj.capacity:
                            # Apply a dynamic probability penalty for being over capacity
                            # For every person over capacity, shift 0.20 probability away from SAFE
                            overage = zf["count"] - zone_obj.capacity
                            penalty = overage * 0.20
                            
                            pred["probs"][STATUS_SAFE] = max(0.0, pred["probs"][STATUS_SAFE] - penalty)
                            pred["probs"][STATUS_MODERATE] = max(0.0, pred["probs"][STATUS_MODERATE] - (penalty * 0.5))
                            
                            # Rebalance the deducted probability into CRITICAL
                            current_sum = sum(pred["probs"].values())
                            pred["probs"][STATUS_CRITICAL] += max(0.0, 1.0 - current_sum)
                            
                            # Normalize
                            total = sum(pred["probs"].values())
                            for k in pred["probs"]:
                                pred["probs"][k] = round(pred["probs"][k] / total, 4)
                                
                            # Promote status if the penalty dethroned the original prediction
                            severity = {STATUS_SAFE: 0, STATUS_MODERATE: 1, STATUS_CRITICAL: 2}
                            best_status = max(pred["probs"], key=pred["probs"].get)
                            pred["status"] = best_status
                            pred["confidence"] = pred["probs"][best_status]
                            pred["pred_idx"] = severity[best_status]
                            
                    predictions[zone_id] = pred
                else:
                    # Buffer warming up — use rule-based
                    warming_frames = self.seq_len - len(self.buffers.get(zone_id, []))
                    predictions[zone_id] = {
                        "status"     : zf["status"],
                        "probs"      : None,
                        "confidence" : None,
                        "note"       : f"warming_up ({warming_frames} frames left)",
                    }
            else:
                # No model loaded — rule-based fallback
                predictions[zone_id] = {
                    "status"     : zf["status"],
                    "probs"      : None,
                    "confidence" : None,
                    "note"       : "rule_based",
                }

        return predictions

    @property
    def is_ready(self) -> bool:
        """True when model is loaded AND all zone buffers are full."""
        if not self.loaded:
            return False
        return all(
            len(buf) >= self.seq_len
            for buf in self.buffers.values()
        )

    @property
    def warmup_progress(self) -> dict:
        """Returns warmup progress per zone (0.0 – 1.0)."""
        return {
            zone_id: min(len(buf) / self.seq_len, 1.0)
            for zone_id, buf in self.buffers.items()
        }