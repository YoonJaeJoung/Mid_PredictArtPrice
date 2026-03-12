"""
train_auction_models.py
-----------------------
Loads processed Parquet files and:
  1. Trains FNN price predictors (image-only & multimodal)
  2. Fits PCA (2-D) and K-Means on embeddings
  3. Precomputes cosine-similarity index for top-5 search
  4. Saves all artefacts to  02_embedding/models/
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

PARQUET_IMAGE = SCRIPT_DIR / "auction_image_only.parquet"
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
# Model architecture (from prompt – do NOT modify)
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
    """Convert a column of list-of-float to a 2-D numpy array."""
    return np.stack(series.values).astype(np.float32)


def load_data():
    """Load parquet files and return merged dataframe + arrays."""
    print("[1/5] Loading parquet files …")
    df_mm = pd.read_parquet(PARQUET_MULTI)

    # Drop rows without a valid price
    df_mm = df_mm.dropna(subset=["Sold_Price_USD"]).copy()
    # Parse price (may be string with commas)
    df_mm["Sold_Price_USD"] = (
        df_mm["Sold_Price_USD"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    df_mm = df_mm[df_mm["Sold_Price_USD"] > 0].reset_index(drop=True)

    image_embs = _stack_embedding_col(df_mm["image_embedding"])
    text_embs  = _stack_embedding_col(df_mm["text_embedding"])
    prices     = df_mm["Sold_Price_USD"].values.astype(np.float32)

    print(f"       Rows with valid price: {len(df_mm):,}")
    return df_mm, image_embs, text_embs, prices


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_fnn(
    mode: str,
    image_train, image_val, image_test,
    text_train, text_val, text_test,
    y_train, y_val, y_test,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 256,
):
    """Train, evaluate, and save one AuctionPricePredictor."""
    print(f"\n── Training FNN (mode={mode}) ──")
    model = AuctionPricePredictor(mode=mode).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()  # Huber

    # Build tensors
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  train_loss={train_loss:.2f}  val_loss={val_loss:.2f}")

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
    print(f"  Test MAE : ${mae:,.2f}")
    print(f"  Test RMSE: ${rmse:,.2f}")

    # Save model
    save_path = MODELS_DIR / f"fnn_{mode}.pt"
    torch.save(model.state_dict(), save_path)
    print(f"  Saved → {save_path}")

    # Save test predictions for Streamlit
    pred_path = MODELS_DIR / f"test_predictions_{mode}.npz"
    np.savez(pred_path, y_true=y_true, y_pred=y_pred)

    return model, mae, rmse


# ---------------------------------------------------------------------------
# PCA + KMeans
# ---------------------------------------------------------------------------

def fit_pca_kmeans(
    df: pd.DataFrame,
    image_embs: np.ndarray,
    text_embs: np.ndarray,
    n_clusters: int = 10,
):
    """Fit PCA-2D and K-Means on both embedding sets, and save metadata."""
    print("\n[3/5] PCA + K-Means …")

    results = {}
    for label, embs in [("image_only", image_embs), ("multimodal", np.hstack([image_embs, text_embs]))]:
        pca = PCA(n_components=2, random_state=42)
        embs_2d = pca.fit_transform(embs)

        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = km.fit_predict(embs)

        # Save
        joblib.dump(pca, MODELS_DIR / f"pca_{label}.pkl")
        joblib.dump(km,  MODELS_DIR / f"kmeans_{label}.pkl")
        np.save(MODELS_DIR / f"pca_2d_{label}.npy", embs_2d)
        np.save(MODELS_DIR / f"clusters_{label}.npy", clusters)

        results[label] = {"pca_2d": embs_2d, "clusters": clusters}
        print(f"  {label}: PCA explained variance ratio = {pca.explained_variance_ratio_.sum():.4f}")

        # --- Sub-task: Calculate Cluster Stats and Representative Artworks ---
        cluster_meta = []
        prices = df["Sold_Price_USD"].values
        
        for c_id in range(n_clusters):
            mask = (clusters == c_id)
            c_prices = prices[mask]
            
            if len(c_prices) == 0:
                cluster_meta.append({
                    "cluster_id": c_id,
                    "count": 0,
                    "avg_price": 0,
                    "median_price": 0,
                    "min_price": 0,
                    "max_price": 0,
                    "std_price": 0,
                    "representative_artworks": []
                })
                continue
                
            # Find representative artworks (closest to centroid)
            centroid = km.cluster_centers_[c_id].reshape(1, -1)
            cluster_embs = embs[mask]
            from sklearn.metrics.pairwise import euclidean_distances
            dists = euclidean_distances(cluster_embs, centroid).flatten()
            
            # Get indices of top 10 closest
            closest_indices_in_mask = np.argsort(dists)[:10]
            # Map back to original dataframe indices
            original_indices = np.where(mask)[0][closest_indices_in_mask]
            
            reps = []
            for idx in original_indices:
                row = df.iloc[idx]
                reps.append({
                    "id": str(row["id"]),
                    "title": str(row["Artwork_Title"]),
                    "artist": str(row["Artist_Name"]),
                    "price": float(row["Sold_Price_USD"])
                })
                
            cluster_meta.append({
                "cluster_id": c_id,
                "count": int(len(c_prices)),
                "avg_price": float(np.mean(c_prices)),
                "median_price": float(np.median(c_prices)),
                "min_price": float(np.min(c_prices)),
                "max_price": float(np.max(c_prices)),
                "std_price": float(np.std(c_prices)),
                "representative_artworks": reps,
                "description": "" # Placeholder for manual LLM step
            })
            
        with open(MODELS_DIR / f"cluster_metadata_{label}.json", "w") as f:
            json.dump(cluster_meta, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Similarity index
# ---------------------------------------------------------------------------

def build_similarity_index(
    image_embs: np.ndarray,
    text_embs: np.ndarray,
    top_k: int = 5,
):
    """Pre-compute top-k cosine-similarity neighbours for each artwork."""
    from sklearn.metrics.pairwise import cosine_similarity

    print("\n[4/5] Building similarity index …")

    for label, embs in [("image_only", image_embs), ("multimodal", np.hstack([image_embs, text_embs]))]:
        n = len(embs)
        # Process in chunks to avoid memory issues
        chunk_size = 1000
        top_k_indices = np.zeros((n, top_k), dtype=np.int32)
        top_k_scores  = np.zeros((n, top_k), dtype=np.float32)

        for start in tqdm(range(0, n, chunk_size), desc=f"Similarity ({label})"):
            end = min(start + chunk_size, n)
            sims = cosine_similarity(embs[start:end], embs)  # (chunk, N)
            # Zero out self-similarity
            for i in range(end - start):
                sims[i, start + i] = -1
            # Top-k
            part_idx = np.argpartition(sims, -top_k, axis=1)[:, -top_k:]
            for i in range(end - start):
                sorted_local = part_idx[i][np.argsort(sims[i, part_idx[i]])[::-1]]
                top_k_indices[start + i] = sorted_local
                top_k_scores[start + i]  = sims[i, sorted_local]

        np.save(MODELS_DIR / f"sim_indices_{label}.npy", top_k_indices)
        np.save(MODELS_DIR / f"sim_scores_{label}.npy",  top_k_scores)
        print(f"  {label}: saved top-{top_k} similarity index")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df, image_embs, text_embs, prices = load_data()

    # ------ split ------
    print("\n[2/5] Train / val / test split …")
    idx_train, idx_temp = train_test_split(
        np.arange(len(prices)), test_size=0.3, random_state=42
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, random_state=42
    )
    print(f"  train={len(idx_train):,}  val={len(idx_val):,}  test={len(idx_test):,}")

    # Log-transform prices for more stable training
    log_prices = np.log1p(prices)

    img_train, img_val, img_test = image_embs[idx_train], image_embs[idx_val], image_embs[idx_test]
    txt_train, txt_val, txt_test = text_embs[idx_train],  text_embs[idx_val],  text_embs[idx_test]
    y_train, y_val, y_test       = log_prices[idx_train], log_prices[idx_val], log_prices[idx_test]

    # ------ FNN training ------
    train_fnn("image_only",
              img_train, img_val, img_test,
              txt_train, txt_val, txt_test,
              y_train, y_val, y_test)

    train_fnn("multimodal",
              img_train, img_val, img_test,
              txt_train, txt_val, txt_test,
              y_train, y_val, y_test)

    # ------ PCA + KMeans ------
    fit_pca_kmeans(df, image_embs, text_embs)

    # ------ Similarity ------
    build_similarity_index(image_embs, text_embs)

    # ------ Save metadata for Streamlit ------
    meta_cols = [c for c in df.columns if c not in ("image_embedding", "text_embedding")]
    df[meta_cols].to_parquet(MODELS_DIR / "metadata.parquet", index=False)

    # Save split indices
    np.savez(
        MODELS_DIR / "split_indices.npz",
        train=idx_train, val=idx_val, test=idx_test,
    )

    print("\n✅  Training complete. All artefacts saved to", MODELS_DIR)


if __name__ == "__main__":
    main()
