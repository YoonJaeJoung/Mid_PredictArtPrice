"""
process_auction_data.py
-----------------------
Cleans auction data, generates text and image embeddings, and outputs
two Parquet files:
  - auction_image_only.parquet   (metadata + image embeddings)
  - auction_multimodal.parquet   (metadata + image + text embeddings)

Text model : nomic-ai/nomic-embed-text-v1.5  (sentence-transformers)
Image model: facebook/dinov2-small            (transformers)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", "Palette images with Transparency", UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_CSV = PROJECT_DIR / "01_dataPreparation" / "artworks_data_clean.csv"
IMAGE_DIR = PROJECT_DIR / "01_dataPreparation" / "image"
OUTPUT_DIR = SCRIPT_DIR  # outputs go into 02_embedding/
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
print(f"[INFO] Using device: {DEVICE}")

# ---------------------------------------------------------------------------
# 1.  Load & clean CSV
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    print("[1/4] Loading CSV …")
    df = pd.read_csv(DATA_CSV)
    print(f"       Total rows: {len(df):,}")

    # Resolve local image paths and flag existence
    df["local_image_path"] = df["id"].apply(
        lambda x: str(IMAGE_DIR / f"{x}.jpg")
    )
    df["_image_exists"] = df["local_image_path"].apply(os.path.isfile)
    found = df["_image_exists"].sum()
    print(f"       Images found on disk: {found:,} / {len(df):,}")

    return df


# ---------------------------------------------------------------------------
# 2.  Text embeddings
# ---------------------------------------------------------------------------

def generate_text_embeddings(df: pd.DataFrame, batch_size: int = 512) -> np.ndarray:
    """
    Embed ONLY: Artwork_Title, Artist_Name, Method, Year_Made.
    Returns a (N, 768) float32 array aligned to df index.
    Checkpoints to disk so restarts skip this step.
    """
    checkpoint_path = CHECKPOINT_DIR / "text_embeddings.npy"
    if checkpoint_path.exists():
        print("[2/4] Loading text embeddings from checkpoint …")
        embeddings = np.load(checkpoint_path)
        print(f"       Loaded shape: {embeddings.shape}")
        if len(embeddings) == len(df):
            return embeddings
        print("       ⚠ Row count mismatch, regenerating …")

    from sentence_transformers import SentenceTransformer

    print("[2/4] Generating text embeddings (nomic-embed-text-v1.5) …")
    model = SentenceTransformer(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        device=str(DEVICE),
    )

    def _format_row(row):
        title  = str(row.get("Artwork_Title", "") or "")
        artist = str(row.get("Artist_Name", "")  or "")
        method = str(row.get("Method", "")        or "")
        year   = str(row.get("Year_Made", "")     or "")
        # nomic requires a task prefix for embedding
        return f"search_document: Title: {title} | Artist: {artist} | Method: {method} | Year: {year}"

    texts = df.apply(_format_row, axis=1).tolist()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)
    print(f"       Text embedding shape: {embeddings.shape}")

    # Save checkpoint
    np.save(checkpoint_path, embeddings)
    print(f"       ✔ Checkpoint saved → {checkpoint_path}")

    del model
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    return embeddings


# ---------------------------------------------------------------------------
# 3.  Image embeddings
# ---------------------------------------------------------------------------

def _letterbox_to_square(img: Image.Image) -> Image.Image:
    """Pad image to a square by adding black bars (letterboxing)."""
    img.thumbnail((256, 256))
    w, h = img.size
    side = max(w, h)
    new_img = Image.new("RGB", (side, side), (0, 0, 0))
    new_img.paste(img, ((side - w) // 2, (side - h) // 2))
    return new_img


CHECKPOINT_EVERY = 1000  # save image embeddings every N rows


def generate_image_embeddings(
    df: pd.DataFrame,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate DINOv2-small embeddings for each image.
    Checkpoints every CHECKPOINT_EVERY rows so it can resume on restart.
    Returns (embeddings, valid_mask):
      - embeddings: (N, 384) float32 array (zeros for invalid rows)
      - valid_mask: (N,) bool  – True where image was processed successfully
    """
    from transformers import AutoImageProcessor, AutoModel

    n = len(df)
    emb_dim = 384  # dinov2-small CLS token dimension

    ckpt_emb_path  = CHECKPOINT_DIR / "image_embeddings.npy"
    ckpt_mask_path = CHECKPOINT_DIR / "image_valid_mask.npy"
    ckpt_pos_path  = CHECKPOINT_DIR / "image_resume_pos.npy"

    # Try to resume from checkpoint
    start_row = 0
    if ckpt_emb_path.exists() and ckpt_mask_path.exists() and ckpt_pos_path.exists():
        embeddings = np.load(ckpt_emb_path)
        valid_mask = np.load(ckpt_mask_path)
        start_row  = int(np.load(ckpt_pos_path))
        if len(embeddings) == n:
            if start_row >= n:
                print(f"[3/4] Image embeddings already complete (loaded from checkpoint)")
                print(f"       Valid image embeddings: {valid_mask.sum():,} / {n:,}")
                return embeddings, valid_mask
            print(f"[3/4] Resuming image embeddings from row {start_row:,} / {n:,} …")
        else:
            print("[3/4] Checkpoint row count mismatch, starting fresh …")
            start_row = 0
            embeddings = np.zeros((n, emb_dim), dtype=np.float32)
            valid_mask = np.zeros(n, dtype=bool)
    else:
        print("[3/4] Generating image embeddings (dinov2-small) …")
        embeddings = np.zeros((n, emb_dim), dtype=np.float32)
        valid_mask = np.zeros(n, dtype=bool)

    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(DEVICE).eval()

    paths = df["local_image_path"].tolist()
    exists = df["_image_exists"].tolist()

    processed_since_ckpt = 0
    total_batches = (n - start_row + batch_size - 1) // batch_size

    for start in tqdm(range(start_row, n, batch_size), desc="Image batches", total=total_batches):
        end = min(start + batch_size, n)
        batch_images = []
        batch_indices = []

        for i in range(start, end):
            if not exists[i]:
                continue
            try:
                img = Image.open(paths[i])
                # Resolve "palette images with transparency" warning by going through RGBA
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGBA")
                img = img.convert("RGB")
                img = _letterbox_to_square(img)
                batch_images.append(img)
                batch_indices.append(i)
            except (UnidentifiedImageError, OSError, Exception):
                continue

        if not batch_images:
            processed_since_ckpt += (end - start)
            continue

        inputs = processor(images=batch_images, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
        # CLS token is first token of last_hidden_state
        cls_embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        for j, idx in enumerate(batch_indices):
            embeddings[idx] = cls_embs[j]
            valid_mask[idx] = True

        processed_since_ckpt += (end - start)

        del inputs, outputs, cls_embs, batch_images
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Periodic checkpoint
        if processed_since_ckpt >= CHECKPOINT_EVERY:
            np.save(ckpt_emb_path,  embeddings)
            np.save(ckpt_mask_path, valid_mask)
            np.save(ckpt_pos_path,  np.array(end))
            processed_since_ckpt = 0

    # Final checkpoint (mark complete)
    np.save(ckpt_emb_path,  embeddings)
    np.save(ckpt_mask_path, valid_mask)
    np.save(ckpt_pos_path,  np.array(n))

    print(f"       Valid image embeddings: {valid_mask.sum():,} / {n:,}")
    return embeddings, valid_mask


# ---------------------------------------------------------------------------
# 4.  Save Parquet files
# ---------------------------------------------------------------------------

def save_outputs(
    df: pd.DataFrame,
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    valid_mask: np.ndarray,
):
    print("[4/4] Saving Parquet files …")

    # Keep only rows that have a valid image embedding
    df_valid = df[valid_mask].copy().reset_index(drop=True)
    img_emb_valid = image_embeddings[valid_mask]
    txt_emb_valid = text_embeddings[valid_mask]

    # Store embeddings as lists-of-floats (Parquet supports nested types)
    df_valid["image_embedding"] = list(img_emb_valid)
    df_valid["text_embedding"]  = list(txt_emb_valid)

    # Drop helper column
    df_valid.drop(columns=["_image_exists"], inplace=True, errors="ignore")

    # --- auction_image_only.parquet ---
    cols_image = [
        c for c in df_valid.columns if c != "text_embedding"
    ]
    path_img = OUTPUT_DIR / "auction_image_only.parquet"
    df_valid[cols_image].to_parquet(path_img, index=False)
    print(f"       Saved {path_img}  ({len(df_valid):,} rows)")

    # --- auction_multimodal.parquet ---
    path_mm = OUTPUT_DIR / "auction_multimodal.parquet"
    df_valid.to_parquet(path_mm, index=False)
    print(f"       Saved {path_mm}  ({len(df_valid):,} rows)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    df = load_data()

    text_embeddings  = generate_text_embeddings(df)
    
    del text_embeddings
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()

    image_embeddings, valid_mask = generate_image_embeddings(df)

    text_embeddings = np.load(CHECKPOINT_DIR / "text_embeddings.npy")
    save_outputs(df, image_embeddings, text_embeddings, valid_mask)
    print("\n✅  Processing complete.")


if __name__ == "__main__":
    main()
