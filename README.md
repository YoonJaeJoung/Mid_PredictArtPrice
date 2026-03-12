# Art Price Prediction & High-Value Image Generation

## Project Overview
This project is for the midterm project of the NYU course **Special Topics in Digital Media - Interrogating AI Tools for Creative Practice DM-UY 4913**. 

The inspiration for this project stemmed from the week 6 activity of making and using crowdsourced database content. I became curious about predicting the price of artworks from **Komar & Melamid — The People's Choice / Most Wanted Paintings (1994–1998)**.

### Goal
The core objective is to predict the auction prices of artworks and generate AI-based imagery that is statistically likely to evaluate at high prices. By analyzing historical sale data, the model identifies features and trends associated with premium values. The project specifically investigates how these predictive models evaluate the conceptual paintings by Komar & Melamid.

### Project Phases
1.  **Data Extraction**: Scraping and cleansing a large-scale dataset of fine art auction records.
2.  **Embedding & Price Prediction**: Developing ML models to map visual and textual features to sale prices.
3.  **Elaborating & Generative Insights**: Using model insights to generate "high-value" prompts and analyzing the K&M "Most Wanted" series.

### AI Tools Used
This project was developed with the assistance of several advanced AI tools and models:
- **Coding Assistance**: Antigravity's Gemini 3.1 Pro and Claude Opus 4.6.
- **Image Generation**: Gemini Nanobanana 2.
- **Text Generation**: Gemini 3 Flash (used for generating descriptive "vibes" for thematic clusters).


## Phase 1: Data Extraction

### 1. Initial Extraction
Data was sourced via the Artnet database (using an NYU-provided corporate plan). The initial search targeted artworks sold between **October 1, 2025, and March 2, 2026** across the following categories:
- Paintings
- Works on Paper
- Prints and Multiples
- Design

### 2. Refined Extraction
However, it was discovered that the "Design" category inadvertently included objects and furniture (e.g., chairs), which skewed the art prediction goals. Originally, `artworks_data.csv` was generated using string matching against the DOM texts (`auto_extract.py`). To correct the inclusion of furniture and objects, a separate, targeted search for only the "Design" category was performed to build an exclusion list. Furthermore, we discovered that some of these design objects, specifically items like the "Pair of 'Sol Y Luna' chairs", were not correctly capturing their "Method" and "Year_Made" due to the initial string matching logic missing those edge cases. 

To resolve this:
- Hybrid parsing logic was implemented, combining character-based string matching with specific DOM element scraping.
- Parallel extraction scripts were developed to handle both the main dataset (`batch_extract.py`) and the exclusion list (`batch_extract_exclude.py`).

### 3. Data Cleansing
- An exclusionary list of artworks (`artworks_exclude.csv`) matching non-target types (design furniture) was generated. 
- A merge script (`filter_data.py`) was introduced to compile the rewritten data, explicitly ignoring any artworks found under the exclusion file by matching a unique combination of Title, Artist, Auction, and Price.
- A final duplication check script (`remove_duplicates.py`) was applied, extracting only entirely novel metadata records and sorting them chronologically by `Sold_Date`.
- **`sanitize_csv.py`** was used to remove internal newlines and standardizing CSV formatting to ensure RFC 4180 compliance and resolve parser errors.
- The new cleansed result has been saved to `artworks_data_clean.csv`.

## Phase 2: Embedding & Price Prediction
In this phase, I developed a robust pipeline to transform raw auction metadata into a format suitable for machine learning. 
- **Embeddings**: I generated high-dimensional semantic mappings for text using `nomic-embed-text-v1.5` and visual features using `DINOv2-small`.
- **Model Training**: I trained several regression models, including Feed-forward Neural Networks (FNN) and XGBoost, to predict log-prices. The multimodal models consistently outperformed image-only models.
- **Clustering**: K-Means clustering was applied to group artworks by visual and semantic similarity, allowing me to derive price statistics and "vibes" for specific art niches.
- **Visualization**: A Streamlit application (`app.py`) was launched to provide an interactive interface for similarity searches, price evaluations, and cluster explorations.

## Phase 3: Elaborating & Future Features
The final phase focuses on leveraging the trained models for creative generation and deep analysis.
- **AI Art Generation (NanoBanana)**: I created a system that auto-generates structured DALL-E/Midjourney prompts tailored to specific price quartiles within any selected visual cluster, aiming to "engineer" high-value aesthetic characteristics.
- **Komar & Melamid Price Analysis**: I integrated the "Most Wanted" and "Least Wanted" paintings from the Dia Art Foundation project. By running these through my pipeline, I compared how a model trained on actual market data evaluates artworks designed based on public opinion polls.
- **Cluster-Specific Modeling**: To increase precision, I developed specialized XGBoost models trained exclusively on individual thematic clusters, allowing for expert-level prediction within specific genres.

## Phase 4: Future Features

### Backlog
- [ ] Clean and normalize metadata fields (e.g. converting currency to standard USD formats, parsing "Year" ranges into absolute integers).

## Directory Structure
- `01_dataPreparation/`
  - `dashboard/`: A web-based analytics dashboard visualizing the main dataset with `dashboard.html` and `serve_dashboard.py`.
  - `raw DOM/`: Contains the unedited saved HTML from Artnet. (not uploaded to github)
  - `deprecated/`: Houses intermediate files and scripts historically generated or no longer used (not uploaded to github; e.g. `artworks_data_rewritten.csv`, `artworks_exclude.csv`, `artworks_data.csv`, `artworks_data_fixed.csv`, `auto_extract.py`, `auto_extract_exclude.py`).
  - `artworks_data_clean.csv`: The final, up-to-date, mathematically unique and chronologically sorted dataset.
  - `image/`: Downloaded artwork images (~122 K files, named `{id}.jpg`).
  - `batch_extract.py` & `batch_extract_exclude.py`: Python scripts utilizing `BeautifulSoup4` for DOM traversal extraction.
  - `filter_data.py`: Prunes identified unwanted collections.
  - `remove_duplicates.py`: Resolves identical row data into unique models.
  - `sanitize_csv.py`: Cleans internal corruption and formatting issues in CSV files.

- `02_embedding/`
  - `prompt.md`: Specification for the embedding pipeline.
  - `process_auction_data.py`: Generates text and image embeddings.
    - **Text Embedding**: Utilizes `nomic-ai/nomic-embed-text-v1.5` for high-dimensional semantic mapping:
      1.  **Field Selection**: Combines `Artwork_Title`, `Artist_Name`, `Method`, and `Year_Made`.
      2.  **Task-Specific Formatting**: Prepends the `search_document:` prefix (required by Nomic v1.5) to ensure optimal embedding quality for retrieval tasks.
      3.  **Schema**: `search_document: Title: {t} | Artist: {a} | Method: {m} | Year: {y}`.
    - **Image Preprocessing**: To handle diverse aspect ratios without distortion or data loss:
      1.  **Memory-Safe Scaling**: High-resolution images are scaled to a maximum of 256px (`thumbnail`) to prevent RAM exhaustion.
      2.  **Letterboxing**: Images are padded with black bars to form a 1:1 square, preserving the original painting's proportions.
      3.  **Model Alignment**: The square output is processed at DINOv2's native 224x224 resolution, ensuring consistent feature extraction across landscape, portrait, and panoramic artworks.
    - Outputs `auction_image_only.parquet` and `auction_multimodal.parquet`.
  - `train_auction_models.py`: Full training pipeline — FNN price predictors (image-only & multimodal), PCA (2-D), K-Means clustering, and cosine-similarity index.
    - **Cluster Analysis**: Calculates price statistics (mean, median, min, max, std) for each cluster and identifies top 10 representative artworks. Saves results to `cluster_metadata_{label}.json`.
    - **Training Results (baseline)**: image_only val_loss=0.96, multimodal val_loss=0.85 (SmoothL1 on log-prices). Image-only showed overfitting (train/val gap widening after epoch 10).
  - `retrain_fnn.py`: FNN-only retraining script with regularization improvements:
    - **Weight Decay** (`1e-4`): L2 regularization to penalize large weights and reduce overfitting.
    - **LR Scheduling**: `ReduceLROnPlateau` (patience=5, factor=0.5) to automatically reduce learning rate when validation loss plateaus.
    - **Early Stopping** (patience=15): Halts training when no improvement is observed, preventing wasted epochs.
    - Reports test metrics in both log-space and USD for interpretability.
  - `benchmark_models.py`: Benchmarks XGBoost, LightGBM, Ridge, and RandomForest against FNN.
    - **Best Result**: XGBoost multimodal achieved MAE(log)=1.2441 (vs FNN 1.38), a 10% improvement in log-space error.
    - USD MAE remains ~$51K across all models, reflecting inherent art market price variance.
  - `train_cluster_models.py`: Trains isolated XGBoost regression models for each individual multimodal cluster to enable highly specialized price predictions for niche visual styles.
  - `clusterPrompt.md`: A template for manually generating cluster "vibe" summaries using an LLM.
  - `app.py`: Streamlit app with *six* interactive views:
    - **Visual Similarity Search**: Select an artwork → top-5 matches with images.
    - **Price Prediction Evaluation**: FNN vs XGBoost scatter (log-log scale) with delta metrics.
    - **Embedding Clusters**: PCA 2-D scatter with click-to-inspect artwork detail panel, price range filter slider, price distribution box plot with explanations.
    - **Cluster Price Prediction**: Inspect metrics on models trained specifically for each individual cluster, comparing errors vs the global models. Features an actual vs. predicted scatter plot on a log-log scale for better visibility of art price ranges.
    - **Komar & Melamid Predictions**: Displays price predictions for K&M's Most/Least Wanted paintings safely normalized to identical heights, and their 5 visually nearest neighbors (Title, Artist, Method, Price) from the auction dataset. Incorporates global algorithms and specific cluster model predictions.
    - **NanoBanana Prompts**: Auto-generates structured generative AI prompts for four targeted price quartiles within any selected visual cluster.
  - `models/`: Trained model weights, PCA/K-Means objects, precomputed similarity data (created at runtime).

- `03_elaborating/`
  - `prompt.md`: Specification for Komar & Melamid price prediction pipeline.
  - `images/`: The Most Wanted / Least Wanted painting images by country.
  - `prepare_km_dataset.py`: Prepares the structured dataset mapping images and countries.
  - `predict_km_paintings.py`: Generates DINOv2 embeddings for the K&M paintings to predict their auction prices using regression models and 5-Nearest Neighbors mapping to actual fine art.
  - `visualize_km_results.py`: Standalone scripts to visualize predictions and neighbor mappings.
  - `km_paintings.csv` & `km_price_predictions.csv`: Structured datasets and the algorithm's price predictions.

## Usage

### 01 — Data Preparation
```bash
source .venv/bin/activate
# Run batch extraction for main directory
python 01_dataPreparation/batch_extract.py
# Run batch extraction for excluded
python 01_dataPreparation/batch_extract_exclude.py
# Produce clean dataset
python 01_dataPreparation/filter_data.py
# Remove duplicates and output sorted final sheet
python 01_dataPreparation/remove_duplicates.py
# Sanitize CSVs if parser errors occur
python 01_dataPreparation/sanitize_csv.py

# Serve Data Analytics Dashboard (Visualizes artworks_data_clean.csv via web browser)
./.venv/bin/python 01_dataPreparation/dashboard/serve_dashboard.py
```

### 02 — Embedding Pipeline
```bash
# Step 1: Generate embeddings (takes time on 122 K images)
.venv/bin/python 02_embedding/process_auction_data.py

# Step 2: Train models, PCA, K-Means, similarity search
.venv/bin/python 02_embedding/train_auction_models.py

# Step 2b (optional): Retrain FNN only with weight decay + early stopping
.venv/bin/python 02_embedding/retrain_fnn.py

# Step 2c (optional): Benchmark alternative models (XGBoost, LightGBM, etc.)
.venv/bin/python 02_embedding/benchmark_models.py

# Step 2d (optional): Train cluster-specific XGBoost models for specialized prediction
.venv/bin/python 02_embedding/train_cluster_models.py

# Step 3: Launch the Streamlit explorer
.venv/bin/python -m streamlit run 02_embedding/app.py
```

### 03 — Elaborating Pipeline (Komar & Melamid Analysis)
```bash
# Step 1: Prepare the dataset mapping from saved images
.venv/bin/python 03_elaborating/prepare_km_dataset.py

# Step 2: Generate embeddings and predict price outliers
.venv/bin/python 03_elaborating/predict_km_paintings.py

# Step 3: Launch standalone result visualizer (alternatively viewable in Streamlit App)
.venv/bin/python 03_elaborating/visualize_km_results.py
```
