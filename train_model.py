"""
train_model.py
----------------
Trains the flood-risk classifier for FloodShield (Raipur prototype) and
saves it to disk so the Streamlit app can load it instantly instead of
retraining on every run.

Same core approach as the original prototype notebook: since real
road-level historical flood sensor data isn't publicly available for
Raipur, we simulate realistic historical storms using the real
elevation/drainage estimates in raipur_data.py. The relationship
(more rain + lower elevation + worse drainage -> more likely to flood)
mirrors real flood physics; the exact numbers are illustrative.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

from raipur_data import roads_df

np.random.seed(42)

N_EVENTS = 200  # simulated historical storms

records = []
max_elev = roads_df.avg_elevation_m.max()
min_elev = roads_df.avg_elevation_m.min()

for event_id in range(N_EVENTS):
    rainfall_mm = np.round(np.random.uniform(10, 180), 1)
    for _, road in roads_df.iterrows():
        norm_elev = (road.avg_elevation_m - min_elev) / (max_elev - min_elev + 1e-9)
        flood_score = (
            (rainfall_mm / 180) * 0.5 +
            (1 - norm_elev) * 0.35 +
            (1 - road.drainage_quality) * 0.15
        )
        flood_score += np.random.normal(0, 0.07)
        flooded = 1 if flood_score > 0.55 else 0
        records.append({
            "event_id": event_id,
            "road_id": road.road_id,
            "rainfall_mm": rainfall_mm,
            "avg_elevation_m": road.avg_elevation_m,
            "drainage_quality": road.drainage_quality,
            "flooded": flooded,
        })

flood_history = pd.DataFrame(records)

features = ["rainfall_mm", "avg_elevation_m", "drainage_quality"]
target = "flooded"
X = flood_history[features]
y = flood_history[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Model accuracy on held-out test data: {acc*100:.1f}%")
print(classification_report(y_test, preds, target_names=["No Flood", "Flooded"]))

joblib.dump(model, "flood_model.joblib")
print("Saved model to flood_model.joblib")
