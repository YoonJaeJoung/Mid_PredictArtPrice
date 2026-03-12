"""
train_cluster_models.py
-----------------------
Loads pre-computed embeddings and cluster assignments and trains
a separate XGBoost model for each cluster to predict log-price.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import json

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MODELS_DIR = SCRIPT_DIR / "models"
PARQUET_MULTI = SCRIPT_DIR / "auction_multimodal.parquet"
CLUSTERS_PATH = MODELS_DIR / "clusters_image_only.npy"

def _stack_embedding_col(series: pd.Series) -> np.ndarray:
    return np.stack(series.values).astype(np.float32)

def main():
    if not PARQUET_MULTI.exists() or not CLUSTERS_PATH.exists():
        print("Required files not found. Run train_auction_models.py first.")
        return

    print("Loading data...")
    df = pd.read_parquet(PARQUET_MULTI)
    df = df.dropna(subset=["Sold_Price_USD"]).copy()
    df["Sold_Price_USD"] = df["Sold_Price_USD"].astype(str).str.replace(",", "", regex=False).astype(float)
    df = df[df["Sold_Price_USD"] > 0].reset_index(drop=True)

    image_embs = _stack_embedding_col(df["image_embedding"])
    text_embs  = _stack_embedding_col(df["text_embedding"])
    prices     = df["Sold_Price_USD"].values.astype(np.float32)
    log_prices = np.log1p(prices)

    clusters = np.load(CLUSTERS_PATH)
    n_clusters = len(np.unique(clusters))

    print(f"Loaded {len(df)} rows across {n_clusters} clusters.")

    # Modes
    feature_sets = {
        "image_only": image_embs,
        "multimodal": np.hstack([image_embs, text_embs]),
    }
    
    for mode_key, features in feature_sets.items():
        print(f"\n========================================")
        print(f" TRAINING MODELS FOR MODE: {mode_key}")
        print(f"========================================")
        
        cluster_metrics = {}

        for c_id in range(n_clusters):
            print(f"\n--- Training XGBoost for Cluster {c_id} ({mode_key}) ---")
            mask = (clusters == c_id)
            c_features = features[mask]
            c_prices = log_prices[mask]

            if len(c_prices) < 10:
                print(f"Cluster {c_id} has too few samples ({len(c_prices)}). Skipping.")
                continue

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                c_features, c_prices, test_size=0.2, random_state=42
            )

            model = xgb.XGBRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                tree_method="hist",
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            y_pred = model.predict(X_test)

            mae_log = mean_absolute_error(y_test, y_pred)
            rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))
            
            y_pred_usd = np.expm1(y_pred)
            y_true_usd = np.expm1(y_test)
            mae_usd = mean_absolute_error(y_true_usd, y_pred_usd)

            print(f"Cluster {c_id} | Samples: {len(c_prices)} | MAE(log): {mae_log:.4f} | MAE(USD): ${mae_usd:,.0f}")

            # Save model
            model_path = MODELS_DIR / f"cluster_{mode_key}_model_{c_id}.pkl"
            joblib.dump(model, model_path)
            
            cluster_metrics[str(c_id)] = {
                "samples": int(len(c_prices)),
                "mae_log": float(mae_log),
                "rmse_log": float(rmse_log),
                "mae_usd": float(mae_usd)
            }

        # Save metrics
        metrics_path = MODELS_DIR / f"cluster_models_meta_{mode_key}.json"
        with open(metrics_path, "w") as f:
            json.dump(cluster_metrics, f, indent=2)

    print("\n✅ Cluster models (both modes) trained and saved successfully.")

if __name__ == "__main__":
    main()
