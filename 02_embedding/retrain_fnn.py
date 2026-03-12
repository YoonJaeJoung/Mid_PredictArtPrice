"""
retrain_fnn.py
--------------
Re-trains only the FNN price predictors with weight_decay regularization
to reduce overfitting. Overwrites existing model files in models/.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
PARQUET_MULTI = SCRIPT_DIR / "auction_multimodal.parquet"

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
print(f"[INFO] Device: {DEVICE}")

# ---------------------------------------------------------------------------
# Model (same architecture as train_auction_models.py)
# ---------------------------------------------------------------------------
class AuctionPricePredictor(nn.Module):
    def __init__(self, mode='multimodal', image_dim=384, text_dim=768, dropout_rate=0.3):
        super().__init__()
        self.mode = mode
        if self.mode == 'multimodal':
            self.input_dim = image_dim + text_dim
        elif self.mode == 'image_only':
            self.input_dim = image_dim
        else:
            raise ValueError("Mode must be 'multimodal' or 'image_only'")

        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
        )

    def forward(self, image_emb, text_emb=None):
        if self.mode == 'multimodal':
            if text_emb is None:
                raise ValueError("text_emb required in 'multimodal' mode.")
            x = torch.cat((image_emb, text_emb), dim=1)
        else:
            x = image_emb
        return self.network(x)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _stack_embedding_col(series: pd.Series) -> np.ndarray:
    return np.stack(series.values).astype(np.float32)

# ---------------------------------------------------------------------------
# Training loop (with weight_decay + early stopping patience)
# ---------------------------------------------------------------------------
def train_fnn(
    mode: str,
    image_train, image_val, image_test,
    text_train, text_val, text_test,
    y_train, y_val, y_test,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    weight_decay: float = 1e-4,
    patience: int = 15,
):
    print(f"\n── Training FNN (mode={mode}, weight_decay={weight_decay}, patience={patience}) ──")
    model = AuctionPricePredictor(mode=mode).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.SmoothL1Loss()

    def _make_dataset(img, txt, y):
        tensors = [torch.tensor(img)]
        if mode == "multimodal":
            tensors.append(torch.tensor(txt))
        tensors.append(torch.tensor(y).unsqueeze(1))
        return TensorDataset(*tensors)

    ds_train = _make_dataset(image_train, text_train, y_train)
    ds_val   = _make_dataset(image_val,   text_val,   y_val)
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val   = DataLoader(ds_val,   batch_size=batch_size)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for batch in loader_train:
            if mode == "multimodal":
                img_b, txt_b, y_b = [b.to(DEVICE) for b in batch]
                preds = model(img_b, txt_b)
            else:
                img_b, y_b = [b.to(DEVICE) for b in batch]
                preds = model(img_b)
            loss = criterion(preds, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_b)
        train_loss /= len(ds_train)

        # --- val ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in loader_val:
                if mode == "multimodal":
                    img_b, txt_b, y_b = [b.to(DEVICE) for b in batch]
                    preds = model(img_b, txt_b)
                else:
                    img_b, y_b = [b.to(DEVICE) for b in batch]
                    preds = model(img_b)
                val_loss += criterion(preds, y_b).item() * len(y_b)
        val_loss /= len(ds_val)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        current_lr = optimizer.param_groups[0]['lr']
        if epoch % 5 == 0 or epoch == 1 or no_improve == 0:
            marker = " ✓ (best)" if no_improve == 0 else ""
            print(f"  Epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}  lr={current_lr:.1e}{marker}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Restore best
    model.load_state_dict(best_state)
    model.eval()

    # --- test ---
    ds_test = _make_dataset(image_test, text_test, y_test)
    loader_test = DataLoader(ds_test, batch_size=batch_size)
    preds_list, actuals_list = [], []
    with torch.no_grad():
        for batch in loader_test:
            if mode == "multimodal":
                img_b, txt_b, y_b = [b.to(DEVICE) for b in batch]
                p = model(img_b, txt_b)
            else:
                img_b, y_b = [b.to(DEVICE) for b in batch]
                p = model(img_b)
            preds_list.append(p.cpu().numpy())
            actuals_list.append(y_b.cpu().numpy())

    y_pred = np.concatenate(preds_list).flatten()
    y_true = np.concatenate(actuals_list).flatten()
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Test MAE (log): {mae:.4f}")
    print(f"  Test RMSE(log): {rmse:.4f}")

    # Convert back from log-space for interpretable metrics
    y_pred_usd = np.expm1(y_pred)
    y_true_usd = np.expm1(y_true)
    mae_usd  = mean_absolute_error(y_true_usd, y_pred_usd)
    rmse_usd = np.sqrt(mean_squared_error(y_true_usd, y_pred_usd))
    print(f"  Test MAE (USD): ${mae_usd:,.2f}")
    print(f"  Test RMSE(USD): ${rmse_usd:,.2f}")

    # Save model
    save_path = MODELS_DIR / f"fnn_{mode}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"  Saved → {save_path}")

    # Save test predictions
    pred_path = MODELS_DIR / f"test_predictions_{mode}.npz"
    np.savez(pred_path, y_true=y_true, y_pred=y_pred)

    return model, mae, rmse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("[1/2] Loading data …")
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

    # Split (same seed as original)
    print("\n[2/2] Training with regularization …")
    idx_train, idx_temp = train_test_split(np.arange(len(prices)), test_size=0.3, random_state=42)
    idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42)
    print(f"  train={len(idx_train):,}  val={len(idx_val):,}  test={len(idx_test):,}")

    log_prices = np.log1p(prices)
    img_train, img_val, img_test = image_embs[idx_train], image_embs[idx_val], image_embs[idx_test]
    txt_train, txt_val, txt_test = text_embs[idx_train],  text_embs[idx_val],  text_embs[idx_test]
    y_train, y_val, y_test       = log_prices[idx_train], log_prices[idx_val], log_prices[idx_test]

    train_fnn("image_only",
              img_train, img_val, img_test,
              txt_train, txt_val, txt_test,
              y_train, y_val, y_test)

    train_fnn("multimodal",
              img_train, img_val, img_test,
              txt_train, txt_val, txt_test,
              y_train, y_val, y_test)

    print("\n✅  Re-training complete. Models saved to", MODELS_DIR)


if __name__ == "__main__":
    main()
