"""
benchmark_models.py
-------------------
Compares alternative regression models against the FNN baseline
for art price prediction using pre-computed embeddings.

Models: XGBoost, LightGBM, Ridge Regression, Random Forest
"""

import numpy as np
import pandas as pd
import joblib
import time
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
PARQUET_MULTI = SCRIPT_DIR / "auction_multimodal.parquet"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stack_embedding_col(series: pd.Series) -> np.ndarray:
    return np.stack(series.values).astype(np.float32)


def load_data():
    print("[1/3] Loading data …")
    df = pd.read_parquet(PARQUET_MULTI)
    df = df.dropna(subset=["Sold_Price_USD"]).copy()
    df["Sold_Price_USD"] = (
        df["Sold_Price_USD"].astype(str).str.replace(",", "", regex=False).astype(float)
    )
    df = df[df["Sold_Price_USD"] > 0].reset_index(drop=True)

    image_embs = _stack_embedding_col(df["image_embedding"])
    text_embs  = _stack_embedding_col(df["text_embedding"])
    prices     = df["Sold_Price_USD"].values.astype(np.float32)
    print(f"  Rows: {len(df):,}")
    return df, image_embs, text_embs, prices


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def get_models():
    """Return dict of model_name -> (model, fit_kwargs)."""
    return {
        "XGBoost": (
            xgb.XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                tree_method="hist",
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=20,
            ),
            lambda X_val, y_val: {"eval_set": [(X_val, y_val)], "verbose": False},
        ),
        "LightGBM": (
            lgb.LGBMRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ),
            lambda X_val, y_val: {
                "eval_set": [(X_val, y_val)],
                "callbacks": [lgb.early_stopping(20, verbose=False)],
            },
        ),
        "Ridge": (
            Ridge(alpha=1.0),
            lambda X_val, y_val: {},
        ),
        "RandomForest": (
            RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=10,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            ),
            lambda X_val, y_val: {},
        ),
    }


# ---------------------------------------------------------------------------
# Train + Evaluate
# ---------------------------------------------------------------------------

def train_and_evaluate(
    model_name, model, fit_kwargs_fn,
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    mode_label,
):
    print(f"\n── {model_name} ({mode_label}) ──")
    t0 = time.time()

    fit_kwargs = fit_kwargs_fn(X_val, y_val)
    model.fit(X_train, y_train, **fit_kwargs)

    elapsed = time.time() - t0

    y_pred = model.predict(X_test)

    # Log-space metrics
    mae_log  = mean_absolute_error(y_test, y_pred)
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred))

    # USD metrics
    y_pred_usd = np.expm1(y_pred)
    y_true_usd = np.expm1(y_test)
    mae_usd  = mean_absolute_error(y_true_usd, y_pred_usd)
    rmse_usd = np.sqrt(mean_squared_error(y_true_usd, y_pred_usd))

    print(f"  Time     : {elapsed:.1f}s")
    print(f"  MAE (log): {mae_log:.4f}")
    print(f"  RMSE(log): {rmse_log:.4f}")
    print(f"  MAE (USD): ${mae_usd:,.0f}")
    print(f"  RMSE(USD): ${rmse_usd:,.0f}")

    return {
        "model_name": model_name,
        "mode": mode_label,
        "mae_log": mae_log,
        "rmse_log": rmse_log,
        "mae_usd": mae_usd,
        "rmse_usd": rmse_usd,
        "time_s": elapsed,
        "y_pred": y_pred,
        "y_true": y_test,
        "model": model,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df, image_embs, text_embs, prices = load_data()

    print("\n[2/3] Splitting data (same seed as FNN) …")
    idx_train, idx_temp = train_test_split(np.arange(len(prices)), test_size=0.3, random_state=42)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42)
    print(f"  train={len(idx_train):,}  val={len(idx_val):,}  test={len(idx_test):,}")

    log_prices = np.log1p(prices)

    # Load existing FNN metrics for comparison
    fnn_results = []
    for fnn_mode in ["image_only", "multimodal"]:
        pred_path = MODELS_DIR / f"test_predictions_{fnn_mode}.npz"
        if pred_path.exists():
            data = np.load(pred_path)
            y_t, y_p = data["y_true"], data["y_pred"]
            fnn_results.append({
                "model_name": "FNN (baseline)",
                "mode": fnn_mode,
                "mae_log": mean_absolute_error(y_t, y_p),
                "rmse_log": np.sqrt(mean_squared_error(y_t, y_p)),
                "mae_usd": mean_absolute_error(np.expm1(y_t), np.expm1(y_p)),
                "rmse_usd": np.sqrt(mean_squared_error(np.expm1(y_t), np.expm1(y_p))),
                "time_s": 0,
            })

    # Prepare features for each mode
    modes = {
        "image_only": image_embs,
        "multimodal": np.hstack([image_embs, text_embs]),
    }

    print("\n[3/3] Training models …")
    all_results = list(fnn_results)
    best_per_mode = {}

    for mode_label, embs in modes.items():
        X_train = embs[idx_train]
        X_val   = embs[idx_val]
        X_test  = embs[idx_test]
        y_train = log_prices[idx_train]
        y_val   = log_prices[idx_val]
        y_test  = log_prices[idx_test]

        models = get_models()
        for model_name, (model, fit_kwargs_fn) in models.items():
            result = train_and_evaluate(
                model_name, model, fit_kwargs_fn,
                X_train, X_val, X_test,
                y_train, y_val, y_test,
                mode_label,
            )
            all_results.append({k: v for k, v in result.items() if k not in ("y_pred", "y_true", "model")})

            # Track best model per mode
            if mode_label not in best_per_mode or result["mae_log"] < best_per_mode[mode_label]["mae_log"]:
                best_per_mode[mode_label] = result

    # ---- Print comparison table ----
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Model':<20} {'Mode':<14} {'MAE(log)':<10} {'RMSE(log)':<10} {'MAE(USD)':<14} {'RMSE(USD)':<14} {'Time':<8}")
    print("-" * 80)
    for r in sorted(all_results, key=lambda x: (x["mode"], x["mae_log"])):
        print(
            f"{r['model_name']:<20} {r['mode']:<14} {r['mae_log']:<10.4f} {r['rmse_log']:<10.4f} "
            f"${r['mae_usd']:<13,.0f} ${r['rmse_usd']:<13,.0f} {r['time_s']:<8.1f}"
        )

    # ---- Save best models ----
    print("\n── Saving best models ──")
    for mode_label, result in best_per_mode.items():
        model_name = result["model_name"]
        model = result["model"]
        print(f"  Best for {mode_label}: {model_name} (MAE_log={result['mae_log']:.4f})")

        # Save model
        save_path = MODELS_DIR / f"best_{mode_label}.pkl"
        joblib.dump(model, save_path)
        print(f"  Saved → {save_path}")

        # Save predictions for Streamlit
        np.savez(
            MODELS_DIR / f"test_predictions_best_{mode_label}.npz",
            y_true=result["y_true"],
            y_pred=result["y_pred"],
        )

    # Save metadata about best models
    import json
    best_meta = {
        mode: {"model_name": r["model_name"], "mae_log": r["mae_log"], "rmse_log": r["rmse_log"],
               "mae_usd": r["mae_usd"], "rmse_usd": r["rmse_usd"]}
        for mode, r in best_per_mode.items()
    }
    with open(MODELS_DIR / "best_model_meta.json", "w") as f:
        json.dump(best_meta, f, indent=2)

    print("\n✅  Benchmark complete. Best models saved to", MODELS_DIR)


if __name__ == "__main__":
    main()
