import sys
import pandas as pd
import numpy as np
import torch
import joblib

sys.path.append('.')
from src.congestion import CongestionInference
from src.zones import ZoneManager, ExitZone

df = pd.read_csv('data/logs/features_log.csv')
zm = ZoneManager()
zm.zones = [ExitZone(1, "Exit A - North", [0,0,10,10], 50), ExitZone(2, "Exit B - South", [0,0,10,10], 50)]

predictor = CongestionInference(zm)
predictor.load()

out = ""
for zone_name in ["Exit A - North", "Exit B - South"]:
    zdf = df[df['zone_name'] == zone_name].tail(30)
    if len(zdf) < 30: continue
    
    zone_id = 1 if "A" in zone_name else 2
    for _, row in zdf.iterrows():
        predictor.update_buffer(zone_id, zone_name, {"count": row["count"], "avg_speed": row["avg_speed"], "density": row["density"]})
    
    pred = predictor.predict_zone(zone_id)
    out += f"\n--- {zone_name} ---\n{pred}\nLast feature_vec: {predictor.buffers[zone_id][-1]}\n"

with open('diag_out.txt', 'w') as f:
    f.write(out)
