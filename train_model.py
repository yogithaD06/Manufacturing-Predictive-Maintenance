
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, roc_auc_score)

DATA = Path("data")
MODELS = Path("models")
MODELS.mkdir(exist_ok=True)

SENSORS = ["temperature_c", "vibration_mm_s", "pressure_bar", "rpm",
           "power_kw", "acoustic_db", "oil_quality_index"]


def build_features(df: pd.DataFrame, machines: pd.DataFrame) -> pd.DataFrame:
    """Rolling-window features per machine. Shared with the dashboard."""
    df = df.sort_values(["machine_id", "timestamp"]).copy()
    g = df.groupby("machine_id", group_keys=False)
    df[SENSORS] = g[SENSORS].transform(lambda s: s.ffill().bfill())
    for col in ["temperature_c", "vibration_mm_s", "power_kw", "acoustic_db", "oil_quality_index"]:
        for w, label in [(6, "1h"), (36, "6h"), (144, "24h")]:
            df[f"{col}_mean_{label}"] = g[col].transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"{col}_std_6h"] = g[col].transform(lambda s: s.rolling(36, min_periods=2).std())
        df[f"{col}_trend_24h"] = df[f"{col}_mean_1h"] - df[f"{col}_mean_24h"]
    df = df.merge(machines[["machine_id", "machine_type", "age_years", "rated_power_kw"]], on="machine_id")
    df["machine_type_code"] = df["machine_type"].astype("category").cat.codes
    df["hour"] = df["timestamp"].dt.hour
    return df


def feature_columns(df: pd.DataFrame) -> list:
    exclude = {"timestamp", "machine_id", "machine_status", "health_index", "failure_within_24h",
               "failure_within_7d", "rul_hours", "machine_type"}
    return [c for c in df.columns if c not in exclude]


def main():
    print("Loading data...")
    df = pd.read_parquet(DATA / "sensor_readings.parquet")
    machines = pd.read_csv(DATA / "machines.csv")

    print("Engineering features...")
    df = build_features(df, machines)
    df = df[df["machine_status"].isin(["Running", "Idle"])]  
    df = df.iloc[::3]                                         

    cutoff = df["timestamp"].min() + (df["timestamp"].max() - df["timestamp"].min()) * 0.75
    train, test = df[df.timestamp < cutoff], df[df.timestamp >= cutoff]
    feats = feature_columns(df)
    y_col = "failure_within_24h"
    print(f"Train rows: {len(train):,}  Test rows: {len(test):,}  Features: {len(feats)}")
    print(f"Positive rate - train: {train[y_col].mean():.3%}  test: {test[y_col].mean():.3%}")

    print("Training model...")
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                           class_weight="balanced", random_state=42)
    model.fit(train[feats], train[y_col])

    proba = model.predict_proba(test[feats])[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": round(float(roc_auc_score(test[y_col], proba)), 4),
        "pr_auc": round(float(average_precision_score(test[y_col], proba)), 4),
        "baseline_pr_auc": round(float(test[y_col].mean()), 4),
        "confusion_matrix": confusion_matrix(test[y_col], pred).tolist(),
        "train_rows": len(train), "test_rows": len(test), "features": feats,
    }
    print("\n=== Test results (last 25% of the timeline) ===")
    print(f"ROC-AUC: {metrics['roc_auc']}   PR-AUC: {metrics['pr_auc']} (random = {metrics['baseline_pr_auc']})")
    print(classification_report(test[y_col], pred, target_names=["No failure", "Failure <24h"], digits=3))

    joblib.dump(model, MODELS / "failure_24h_model.joblib")
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved models/failure_24h_model.joblib and models/metrics.json")


if __name__ == "__main__":
    main()
