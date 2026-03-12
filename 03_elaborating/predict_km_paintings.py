import os
import json
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", "Palette images with Transparency", UserWarning)

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
CSV_PATH = SCRIPT_DIR / "km_paintings.csv"
MODEL_PATH = PROJECT_DIR / "02_embedding" / "models" / "best_image_only.pkl"
OUTPUT_CSV = SCRIPT_DIR / "km_price_predictions.csv"
KMEANS_MODEL_PATH = PROJECT_DIR / "02_embedding" / "models" / "kmeans_multimodal.pkl"

# Device
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
print(f"[INFO] Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# Replicated Model Architecture from 02_embedding
# ---------------------------------------------------------------------------
# (Model architecture removed since we are using scikit-learn compatible model)

# ---------------------------------------------------------------------------
# Helper logic
# ---------------------------------------------------------------------------
def _letterbox_to_square(img: Image.Image) -> Image.Image:
    img.thumbnail((256, 256))
    w, h = img.size
    side = max(w, h)
    new_img = Image.new("RGB", (side, side), (0, 0, 0))
    new_img.paste(img, ((side - w) // 2, (side - h) // 2))
    return new_img

def main():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found. Run dataset preparation script first.")
        sys.exit(1)
        
    df = pd.read_csv(CSV_PATH)
    print(f"[INFO] Loaded {len(df)} painting records.")
    
    # 1. Generate Image Embeddings
    print("[INFO] Loading DINOv2 model...")
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    dino_model = AutoModel.from_pretrained("facebook/dinov2-small").to(DEVICE).eval()
    
    emb_dim = 384
    embeddings = np.zeros((len(df), emb_dim), dtype=np.float32)
    valid_mask = np.zeros(len(df), dtype=bool)
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Image Embeddings"):
        abs_img_path = PROJECT_DIR / row['image_path']
        if not abs_img_path.exists():
            continue
            
        try:
            img = Image.open(abs_img_path)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGBA")
            img = img.convert("RGB")
            img = _letterbox_to_square(img)
            inputs = processor(images=img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                outputs = dino_model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings[i] = cls_emb
            valid_mask[i] = True
        except Exception:
            pass
            
    # 2. Predict
    print("[INFO] Loading best image_only non-linear model and similarity index...")
    global_model = None
    if MODEL_PATH.exists():
        global_model = joblib.load(MODEL_PATH)
    
    kmeans_model = None
    if KMEANS_MODEL_PATH.exists():
        kmeans_model = joblib.load(KMEANS_MODEL_PATH)
    
    cluster_models_mm = {}
    cluster_models_img = {}
    for c_id in range(10):
        mm_p = PROJECT_DIR / "02_embedding" / "models" / f"cluster_multimodal_model_{c_id}.pkl"
        img_p = PROJECT_DIR / "02_embedding" / "models" / f"cluster_image_only_model_{c_id}.pkl"
        if mm_p.exists():
            cluster_models_mm[c_id] = joblib.load(mm_p)
        if img_p.exists():
            cluster_models_img[c_id] = joblib.load(img_p)
    
    # kNN Data
    PRIMARY_EMBS_PATH = PROJECT_DIR / "02_embedding" / "auction_image_only.parquet"
    primary_embs, primary_prices = None, None
    if PRIMARY_EMBS_PATH.exists():
        df_primary = pd.read_parquet(PRIMARY_EMBS_PATH).dropna(subset=["Sold_Price_USD"])
        df_primary["Sold_Price_USD"] = pd.to_numeric(df_primary["Sold_Price_USD"].astype(str).str.replace(",", "", regex=False), errors='coerce')
        df_primary = df_primary[df_primary["Sold_Price_USD"] > 0].reset_index(drop=True)
        primary_embs = np.stack(df_primary["image_embedding"].values).astype(np.float32)
        primary_prices = df_primary["Sold_Price_USD"].values

    preds_xgb = []
    preds_knn = []
    preds_cluster_mm = []
    preds_cluster_img = []
    neighbors = []
    
    for i in range(len(df)):
        if valid_mask[i]:
            img_feat = embeddings[i:i+1]
            
            # Global
            p_xgb = np.expm1(global_model.predict(img_feat)[0]) if global_model else np.nan
            preds_xgb.append(p_xgb)
            
            # kNN
            if primary_embs is not None:
                from sklearn.metrics.pairwise import cosine_similarity
                sims = cosine_similarity(img_feat, primary_embs).flatten()
                top_5 = np.argsort(sims)[-5:][::-1]
                preds_knn.append(np.mean(primary_prices[top_5]))
                neighbors.append(top_5.tolist())
            else:
                preds_knn.append(np.nan); neighbors.append([])

            # Cluster
            if kmeans_model:
                mm_feat = np.hstack([img_feat, np.zeros((1, 768), dtype=np.float32)])
                c_id = kmeans_model.predict(mm_feat)[0]
                p_mm = np.expm1(cluster_models_mm[c_id].predict(mm_feat)[0]) if c_id in cluster_models_mm else np.nan
                p_img = np.expm1(cluster_models_img[c_id].predict(img_feat)[0]) if c_id in cluster_models_img else np.nan
                preds_cluster_mm.append(p_mm)
                preds_cluster_img.append(p_img)
            else:
                preds_cluster_mm.append(np.nan); preds_cluster_img.append(np.nan)
        else:
            preds_xgb.append(np.nan); preds_knn.append(np.nan)
            preds_cluster_mm.append(np.nan); preds_cluster_img.append(np.nan)
            neighbors.append([])
            
    df['predicted_price_xgb'] = preds_xgb
    df['predicted_price_knn'] = preds_knn
    df['predicted_price_cluster_mm'] = preds_cluster_mm
    df['predicted_price_cluster_img'] = preds_cluster_img
    df['predicted_price_cluster'] = preds_cluster_img # Default UI to image-only cluster model
    df['neighbor_indices'] = [json.dumps(x) for x in neighbors]
    
    results = df[['country', 'category', 'predicted_price_xgb', 'predicted_price_knn', 'predicted_price_cluster_mm', 'predicted_price_cluster_img', 'predicted_price_cluster', 'neighbor_indices']].dropna(subset=['predicted_price_xgb'])
    results.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Saved predictions to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
