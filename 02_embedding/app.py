"""
app.py – Streamlit Visualization for Art Auction Embeddings
-----------------------------------------------------------
Features:
  1. Visual Similarity Search  (select artwork → show top-5 matches)
  2. Price Prediction Evaluation (scatter + MAE/RMSE, FNN vs Best model)
  3. Embedding Clusters          (Plotly PCA-2D scatter with click-to-inspect)

Toggle between "Image Only" and "Multimodal" embeddings at the top.
"""

import json
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR / "models"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Rainbow colors based on Plotly default palette reordered for intuition
RAINBOW_COLORS = [
    "#EF553B", # Red
    "#FFA15A", # Orange
    "#FECB52", # Yellow
    "#B6E880", # Lime
    "#00CC96", # Green
    "#19D3F3", # Cyan
    "#636EFA", # Blue
    "#AB63FA", # Purple
    "#FF6692", # Pink
    "#FF97FF"  # Magenta
]
CLUSTER_ORDER = [str(i) for i in range(10)]

# ---------------------------------------------------------------------------
# Load cached resources
# ---------------------------------------------------------------------------

@st.cache_data
def load_metadata() -> pd.DataFrame:
    return pd.read_parquet(MODELS_DIR / "metadata.parquet")


@st.cache_data
def load_predictions(mode: str):
    data = np.load(MODELS_DIR / f"test_predictions_{mode}.npz")
    return data["y_true"], data["y_pred"]


@st.cache_data
def load_best_predictions(mode: str):
    path = MODELS_DIR / f"test_predictions_best_{mode}.npz"
    if path.exists():
        data = np.load(path)
        return data["y_true"], data["y_pred"]
    return None, None


@st.cache_data
def load_best_model_meta():
    path = MODELS_DIR / "best_model_meta.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def load_pca(mode: str):
    return np.load(MODELS_DIR / f"pca_2d_{mode}.npy")


@st.cache_data
def load_clusters(mode: str):
    return np.load(MODELS_DIR / f"clusters_{mode}.npy")


@st.cache_data
def load_similarity(mode: str):
    indices = np.load(MODELS_DIR / f"sim_indices_{mode}.npy")
    scores  = np.load(MODELS_DIR / f"sim_scores_{mode}.npy")
    return indices, scores


@st.cache_data
def load_cluster_metadata(mode: str):
    path = MODELS_DIR / f"cluster_metadata_{mode}.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None

@st.cache_data
def load_cluster_predictions(mode: str):
    path = MODELS_DIR / f"test_predictions_cluster_all_{mode}.npz"
    if path.exists():
        data = np.load(path)
        return {k: data[k] for k in data.files}
    return None

# ---------------------------------------------------------------------------

# App
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Art Auction Explorer",
        page_icon="🎨",
        layout="wide",
    )
    st.title("🎨 Art Auction Embedding Explorer")

    # ---- mode toggle ----
    mode_label = st.radio(
        "Embedding Mode",
        ["Image Only", "Multimodal (Image + Text)"],
        horizontal=True,
    )
    mode = "image_only" if mode_label.startswith("Image") else "multimodal"

    df = load_metadata()

    # ---- tabs ----
    tab1, tab2, tab3, tab_cluster, tab4, tab_nano = st.tabs([
        "🔍 Visual Similarity Search",
        "💰 Price Prediction Evaluation",
        "🗺️ Embedding Clusters",
        "📊 Cluster Price Prediction",
        "🖼️ Komar & Melamid Predictions",
        "🍌 NanoBanana Prompts"
    ])

    # =====================================================================
    # TAB 1: Visual Similarity Search
    # =====================================================================
    with tab1:
        st.header("Visual Similarity Search")
        st.write("Select an artwork to find the **top 5 most visually similar** pieces.")

        sim_indices, sim_scores = load_similarity(mode)

        # Build selection label
        labels = (
            df["id"].astype(str) + " – " +
            df["Artwork_Title"].fillna("Untitled").str[:60] + " (" +
            df["Artist_Name"].fillna("Unknown") + ")"
        ).tolist()

        selected_idx = st.selectbox(
            "Choose an artwork",
            range(len(labels)),
            format_func=lambda i: labels[i],
        )

        if selected_idx is not None:
            col_src, col_matches = st.columns([1, 3])

            with col_src:
                st.subheader("Selected Artwork")
                img_path = df.iloc[selected_idx]["local_image_path"]
                img_url = df.iloc[selected_idx].get("Image_URL", "")

                if Path(img_path).exists():
                    st.image(Image.open(img_path), use_container_width=True)
                elif img_url:
                    st.image(img_url, use_container_width=True)
                
                actual_price = df.iloc[selected_idx]['Sold_Price_USD']
                st.caption(
                    f"**{df.iloc[selected_idx]['Artwork_Title']}**  \n"
                    f"{df.iloc[selected_idx]['Artist_Name']}  \n"
                    f"${actual_price:,.0f}"
                )

            with col_matches:
                st.subheader("Top 5 Similar Artworks")
                cols = st.columns(5)
                
                neighbor_prices = []
                for rank, col in enumerate(cols):
                    nb_idx = int(sim_indices[selected_idx, rank])
                    score  = sim_scores[selected_idx, rank]
                    nb_row = df.iloc[nb_idx]
                    nb_price = nb_row['Sold_Price_USD']
                    neighbor_prices.append(nb_price)
                    
                    with col:
                        nb_img = nb_row["local_image_path"]
                        nb_url = nb_row.get("Image_URL", "")
                        if Path(nb_img).exists():
                            st.image(Image.open(nb_img), use_container_width=True)
                        elif nb_url:
                            st.image(nb_url, use_container_width=True)
                        st.caption(
                            f"**{nb_row['Artwork_Title'][:40]}**  \n"
                            f"{nb_row['Artist_Name']}  \n"
                            f"${nb_price:,.0f}  \n"
                            f"Similarity: {score:.3f}"
                        )
                        
            # --- 5-NN Price Comparison ---
            st.markdown("---")
            st.subheader("5-NN Price Prediction Analysis")
            avg_neighbor_price = np.mean(neighbor_prices)
            error_val = abs(actual_price - avg_neighbor_price)
            error_pct = (error_val / actual_price) * 100 if actual_price > 0 else 0
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Actual Price", f"${actual_price:,.0f}")
            c2.metric("5-NN Avg Price", f"${avg_neighbor_price:,.0f}")
            c3.metric("Error", f"${error_val:,.0f}", f"{error_pct:.1f}%", delta_color="inverse")
            
            # Global 5-NN Error Calculation
            with st.expander("Calculate Global 5-NN Error (All Artworks)"):
                if st.button("Compute Global Error for Current Mode"):
                    with st.spinner("Calculating..."):
                        all_errors = []
                        all_prices = df["Sold_Price_USD"].values
                        for i in range(len(df)):
                            nb_indices = sim_indices[i]
                            nb_prices = all_prices[nb_indices]
                            pred = np.mean(nb_prices)
                            all_errors.append(abs(all_prices[i] - pred))
                        
                        global_mae = np.mean(all_errors)
                        st.success(f"Global 5-NN Mean Absolute Error (MAE): **${global_mae:,.0f}**")

    # =====================================================================
    # TAB 2: Price Prediction Evaluation
    # =====================================================================
    with tab2:
        st.header("Price Prediction Evaluation")

        # --- FNN predictions ---
        y_true_log, y_pred_log = load_predictions(mode)
        y_true = np.expm1(y_true_log)
        y_pred = np.expm1(y_pred_log)
        mae  = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # --- Best model predictions ---
        best_meta = load_best_model_meta()
        y_true_best_log, y_pred_best_log = load_best_predictions(mode)
        has_best = y_true_best_log is not None

        if has_best:
            y_true_best = np.expm1(y_true_best_log)
            y_pred_best = np.expm1(y_pred_best_log)
            mae_best  = np.mean(np.abs(y_true_best - y_pred_best))
            rmse_best = np.sqrt(np.mean((y_true_best - y_pred_best) ** 2))
            best_name = best_meta.get(mode, {}).get("model_name", "Best Model") if best_meta else "Best Model"

        # Metrics row
        if has_best:
            st.subheader("Model Comparison")
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("FNN MAE", f"${mae:,.0f}")
            col_b.metric("FNN RMSE", f"${rmse:,.0f}")
            col_c.metric(f"{best_name} MAE", f"${mae_best:,.0f}",
                         delta=f"{((mae_best - mae) / mae * 100):+.1f}%",
                         delta_color="inverse")
            col_d.metric(f"{best_name} RMSE", f"${rmse_best:,.0f}",
                         delta=f"{((rmse_best - rmse) / rmse * 100):+.1f}%",
                         delta_color="inverse")
        else:
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("MAE", f"${mae:,.0f}")
            col_m2.metric("RMSE", f"${rmse:,.0f}")

        # Scatter plot
        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=y_true, y=y_pred,
            mode="markers",
            marker=dict(size=3, opacity=0.4, color="#636EFA"),
            name="FNN",
        ))
        if has_best:
            fig.add_trace(go.Scattergl(
                x=y_true_best, y=y_pred_best,
                mode="markers",
                marker=dict(size=3, opacity=0.4, color="#EF553B"),
                name=best_name,
            ))
        # Perfect prediction line (log-spaced for log axes)
        all_true = np.concatenate([y_true, y_true_best]) if has_best else y_true
        all_pred = np.concatenate([y_pred, y_pred_best]) if has_best else y_pred
        min_val = max(1, min(all_true.min(), all_pred.min()))
        max_val = max(all_true.max(), all_pred.max())
        line_pts = np.logspace(np.log10(min_val), np.log10(max_val), 100)
        fig.add_trace(go.Scatter(
            x=line_pts, y=line_pts,
            mode="lines",
            line=dict(dash="dash", color="gray", width=1),
            name="Perfect Prediction",
        ))
        fig.update_layout(
            title="Actual vs Predicted Price (Test Set) — Log Scale",
            xaxis_title="Actual Price (USD)",
            yaxis_title="Predicted Price (USD)",
            xaxis_type="log",
            yaxis_type="log",
            height=600,
        )
        st.plotly_chart(fig, use_container_width=True)

        # How to read this graph
        with st.expander("📖 How to read this graph"):
            st.markdown("""
- **X-axis** = Actual sale price, **Y-axis** = Model's predicted price
- **Gray dashed line** = perfect prediction (actual = predicted)
- **Dots above the line** → model **overestimated** the price
- **Dots below the line** → model **underestimated** the price
- **Tighter cluster around the line** = better predictions
- **MAE** = average dollar error; **RMSE** = penalizes large errors more heavily
- A large gap between RMSE and MAE indicates some extreme outlier predictions (common for high-value artworks)
            """)

    # =====================================================================
    # TAB 3: Embedding Clusters
    # =====================================================================
    with tab3:
        st.header("Embedding Clusters (PCA 2-D)")

        pca_2d   = load_pca(mode)
        clusters = load_clusters(mode)

        cluster_df = pd.DataFrame({
            "PC1": pca_2d[:, 0],
            "PC2": pca_2d[:, 1],
            "Cluster": clusters.astype(str),
            "Title": df["Artwork_Title"].fillna("Untitled"),
            "Artist": df["Artist_Name"].fillna("Unknown"),
            "Price": df["Sold_Price_USD"],
            "idx": range(len(df)),
        })

        # Load metadata for hover and summary
        meta_list = load_cluster_metadata(mode)
        cluster_map = {str(m["cluster_id"]): m for m in meta_list} if meta_list else {}

        if cluster_map:
            cluster_df["Avg_Cluster_Price"] = cluster_df["Cluster"].map(
                lambda x: cluster_map.get(x, {}).get("avg_price", 0)
            )

        # --- Price range filter ---
        price_min = int(cluster_df["Price"].min())
        price_max = int(cluster_df["Price"].max())
        price_range = st.slider(
            "Filter by price range (USD)",
            min_value=price_min,
            max_value=min(price_max, 1_000_000),  # cap slider at $1M for usability
            value=(price_min, min(price_max, 1_000_000)),
            step=100,
            format="$%d",
            key="price_filter",
        )
        show_above = st.checkbox(f"Include artworks above ${min(price_max, 1_000_000):,}", value=True)

        mask = (cluster_df["Price"] >= price_range[0]) & (cluster_df["Price"] <= price_range[1])
        if show_above and price_range[1] >= min(price_max, 1_000_000):
            mask = mask | (cluster_df["Price"] > min(price_max, 1_000_000))
        filtered_df = cluster_df[mask].copy()
        st.caption(f"Showing {len(filtered_df):,} / {len(cluster_df):,} artworks")

        # --- Layout: scatter on left, artwork detail on right ---
        col_chart, col_detail = st.columns([3, 1])

        with col_chart:
            fig = px.scatter(
                filtered_df,
                x="PC1", y="PC2",
                color="Cluster",
                category_orders={"Cluster": CLUSTER_ORDER},
                color_discrete_sequence=RAINBOW_COLORS,
                custom_data=["idx", "Title", "Artist", "Price", "Cluster"],
                hover_data={
                    "PC1": False, "PC2": False, "idx": False,
                    "Title": True, "Artist": True, "Price": ":,.0f",
                    "Cluster": True,
                },
                title=f"Artwork Embeddings – PCA 2-D ({len(filtered_df):,} artworks)",
                height=600,
                opacity=0.5,
            )
            fig.update_traces(marker=dict(size=4))

            # Use Streamlit's on_select for click interaction
            event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="cluster_scatter")

        with col_detail:
            st.subheader("Artwork Detail")

            # Extract selected point from event
            selected_point_idx = None
            if event and event.selection and event.selection.points:
                point = event.selection.points[0]
                custom = point.get("customdata", [])
                if custom and len(custom) >= 1:
                    selected_point_idx = int(custom[0])

            if selected_point_idx is not None and 0 <= selected_point_idx < len(df):
                row = df.iloc[selected_point_idx]
                img_path = row.get("local_image_path", "")
                img_url = row.get("Image_URL", "")
                if img_path and Path(img_path).exists():
                    st.image(Image.open(img_path), use_container_width=True)
                elif img_url:
                    st.image(img_url, use_container_width=True)
                else:
                    st.info("No image available")

                st.markdown(f"**{row.get('Artwork_Title', 'Untitled')}**")
                st.markdown(f"*{row.get('Artist_Name', 'Unknown')}*")
                st.markdown(f"💰 **${row.get('Sold_Price_USD', 0):,.0f}**")

                c_id = str(clusters[selected_point_idx])
                c_meta = cluster_map.get(c_id, {})
                st.markdown(f"🏷️ Cluster **{c_id}**")
                if c_meta.get("description"):
                    st.caption(c_meta["description"])

                # Additional metadata
                for col_name in ["Method", "Year_Made", "Auction_House", "Sold_Date"]:
                    if col_name in row and pd.notna(row[col_name]):
                        st.text(f"{col_name}: {row[col_name]}")
            else:
                st.info("👆 Click a dot in the scatter plot to inspect that artwork")

        # ---- Price Distribution (separate row) ----
        st.subheader("Price Distribution by Cluster")
        fig_box = px.box(
            cluster_df,
            x="Cluster", y="Price",
            color="Cluster",
            category_orders={"Cluster": CLUSTER_ORDER},
            color_discrete_sequence=RAINBOW_COLORS,
            log_y=True,
            title="Sold Price Distribution (Log Scale)",
            points="outliers"
        )
        st.plotly_chart(fig_box, use_container_width=True)

        with st.expander("📖 How to read this chart"):
            st.markdown("""
- Each **box** shows the price distribution for one cluster
- **Box edges** = 25th and 75th percentile (middle 50% of prices)
- **Line inside box** = **median** price (50th percentile)
- **Whiskers** = extend to 1.5× the interquartile range
- **Dots beyond whiskers** = outliers (unusually high or low prices)
- **Y-axis is log scale** — each gridline is 10× the previous one ($10 → $100 → $1K → $10K…)
- A **tall box** means wide price variance; a **short box** means prices are tightly grouped
- Compare **median lines** across clusters to see which visual styles command higher prices
            """)

        # ---- Cluster Summary Table (separate row) ----
        st.subheader("Cluster Summary")
        if meta_list:
            summary_data = []
            for m in meta_list:
                summary_data.append({
                    "ID": m["cluster_id"],
                    "Count": m["count"],
                    "Avg Price": f"${m['avg_price']:,.0f}",
                    "Med Price": f"${m['median_price']:,.0f}",
                    "Description": m.get("description") or "*(Run clusterPrompt.md to generate)*"
                })
            st.table(pd.DataFrame(summary_data))
        else:
            st.info("No cluster metadata found. Please run training script first.")

        # ---- Representative Artworks ----
        if meta_list:
            st.markdown("---")
            st.subheader("View Representative Artworks per Cluster")
            c_select = st.selectbox("Select Cluster to inspect", [str(m["cluster_id"]) for m in meta_list])
            c_meta = cluster_map.get(c_select)
            if c_meta:
                st.write(f"**Description:** {c_meta.get('description') or 'None'}")
                reps = c_meta.get("representative_artworks", [])
                
                st.markdown("##### Model-Generated Concept Image")
                # Load the generated image from our stable models directory based on mode
                gen_img_path = MODELS_DIR / f"cluster_images/{mode}/cluster_{c_select}.png"
                
                if gen_img_path.exists():
                    st.image(Image.open(gen_img_path), width=400, caption=f"AI Generated Concept for Cluster {c_select}")
                else:
                    st.info("AI concept image not generated for this cluster yet.")
                
                st.markdown("##### Top Representative Artworks (closest to centroid)")
                # Display actual images for the representatives
                cols = st.columns(5)
                
                # Filter reps to only those with valid local images
                valid_reps = []
                for rep in reps:
                    matching_rows = df[df["id"].astype(str) == str(rep["id"])]
                    if not matching_rows.empty:
                        img_path = matching_rows.iloc[0]["local_image_path"]
                        img_url = matching_rows.iloc[0].get("Image_URL", "")
                        if Path(img_path).exists():
                            rep["local_image_path"] = img_path
                            valid_reps.append(rep)
                        elif img_url:
                            rep["local_image_path"] = None
                            rep["Image_URL"] = img_url
                            valid_reps.append(rep)
                        if len(valid_reps) >= 10:
                            break
                                
                for idx, rep in enumerate(valid_reps):
                    col_idx = idx % 5
                    img_path = rep.get("local_image_path")
                    img_url = rep.get("Image_URL", "")
                    with cols[col_idx]:
                        if img_path and Path(img_path).exists():
                            st.image(Image.open(img_path), use_container_width=True)
                        elif img_url:
                            st.image(img_url, use_container_width=True)
                        st.caption(
                            f"**{rep['title'][:30]}**\n\n"
                            f"*{rep['artist']}*\n\n"
                            f"${rep['price']:,.0f}"
                        )
                    # Add a new row of columns after 5 items
                    if col_idx == 4 and idx < len(valid_reps) - 1:
                        cols = st.columns(5)

    # =====================================================================
    # TAB CLUSTER: Cluster Price Prediction
    # =====================================================================
    with tab_cluster:
        st.header("Cluster Price Prediction")
        st.write(f"Evaluating the performance of price prediction models trained *specifically* on each cluster using the **{mode}** mode.")

        cluster_meta_path = MODELS_DIR / f"cluster_models_meta_{mode}.json"
        
        if cluster_meta_path.exists():
            with open(cluster_meta_path, "r") as f:
                cluster_metrics = json.load(f)
                
            selected_c_id = st.selectbox(
                f"Select a Cluster to view its {mode} price model performance",
                list(cluster_metrics.keys())
            )
            
            c_metrics = cluster_metrics[selected_c_id]
            st.write(f"**Cluster {selected_c_id} {mode} Model** trained on {c_metrics['samples']} samples.")
            
            col_a, col_b = st.columns(2)
            col_a.metric("Cluster Model MAE", f"${c_metrics['mae_usd']:,.0f}")
            col_b.metric("Cluster Model RMSE (log-space)", f"{c_metrics['rmse_log']:.4f}")
            
            # Show a generic comparison vs Global FNN
            y_true_log, y_pred_log = load_predictions(mode)
            y_true = np.expm1(y_true_log)
            y_pred = np.expm1(y_pred_log)
            global_mae = np.mean(np.abs(y_true - y_pred))
            
            st.caption(f"For context, the Global {mode} FNN model MAE on the entire test set is **${global_mae:,.0f}**")
            diff = c_metrics['mae_usd'] - global_mae
            if diff < 0:
                st.success(f"Cluster model outperforms global average by **${abs(diff):,.0f}**")
            else:
                st.warning(f"Cluster model error is higher than global average by **${diff:,.0f}**")
                
            cluster_preds = load_cluster_predictions(mode)
            if cluster_preds and f"y_true_{selected_c_id}" in cluster_preds:
                y_true_c = cluster_preds[f"y_true_{selected_c_id}"]
                y_pred_c = cluster_preds[f"y_pred_{selected_c_id}"]
                
                y_true_usd = np.expm1(y_true_c)
                y_pred_usd = np.expm1(y_pred_c)
                
                fig = px.scatter(
                    x=y_true_usd,
                    y=y_pred_usd,
                    labels={'x': 'Actual Sold Price (USD)', 'y': 'Predicted Price (USD)'},
                    title=f"Cluster {selected_c_id} {mode.capitalize()} - Actual vs Predicted (Log Scale)",
                    opacity=0.6,
                    color_discrete_sequence=['#AB63FA']
                )
                
                if len(y_true_usd) > 0:
                    # Use log-spaced points for the diagonal line
                    min_val = max(1.0, min(y_true_usd.min(), y_pred_usd.min()))
                    max_val = max(y_true_usd.max(), y_pred_usd.max())
                    line_pts = np.logspace(np.log10(min_val), np.log10(max_val), 100)
                    
                    fig.add_trace(
                        go.Scatter(x=line_pts, y=line_pts,
                                   mode='lines',
                                   line=dict(color='gray', dash='dash'),
                                   showlegend=False,
                                   name='Perfect Prediction')
                    )
                
                fig.update_layout(
                    xaxis_type="log",
                    yaxis_type="log",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning(f"Cluster model metrics for '{mode}' not found. Please run `train_cluster_models.py` first.")

    # =====================================================================
    # TAB 4: Komar & Melamid Predictions
    # =====================================================================
    with tab4:
        st.header("Komar & Melamid Predictions")
        st.write("Evaluating the 'Most Wanted' and 'Least Wanted' paintings from the Komar & Melamid Web Project against our art market pricing models.")
        
        km_csv = Path(SCRIPT_DIR.parent / "03_elaborating" / "km_price_predictions.csv")
        km_meta_csv = Path(SCRIPT_DIR.parent / "03_elaborating" / "km_paintings.csv")
        
        if km_csv.exists() and km_meta_csv.exists():
            km_df = pd.read_csv(km_csv)
            km_meta = pd.read_csv(km_meta_csv)
            
            # Merge or just use the meta for images
            # km_df has country, category, predicted_price_xgb, predicted_price_knn
            # km_meta has country, category, image_path
            
            # Selection for Country
            countries = km_df['country'].unique()
            selected_country = st.selectbox("Select Country to view paintings", countries)
            
            # Use columns dynamically based on what's available
            has_xgb = 'predicted_price_xgb' in km_df.columns
            has_knn = 'predicted_price_knn' in km_df.columns
            has_cluster = 'predicted_price_cluster' in km_df.columns

            if selected_country:
                # First fetch image paths to calculate height match
                mw_meta = km_meta[(km_meta['country'] == selected_country) & (km_meta['category'] == 'most_wanted')]
                lw_meta = km_meta[(km_meta['country'] == selected_country) & (km_meta['category'] == 'least_wanted')]
                
                mw_img = None
                lw_img = None
                mw_height = 0
                lw_height = 0
                
                if not mw_meta.empty:
                    p = SCRIPT_DIR.parent / mw_meta.iloc[0]['image_path']
                    if p.exists():
                        mw_img = Image.open(p)
                        mw_height = mw_img.height
                if not lw_meta.empty:
                    p = SCRIPT_DIR.parent / lw_meta.iloc[0]['image_path']
                    if p.exists():
                        lw_img = Image.open(p)
                        lw_height = lw_img.height
                        
                target_height = None
                if mw_img and lw_img:
                    target_height = min(mw_height, lw_height)
                    
                col_mw, col_lw = st.columns(2)
                
                # Most Wanted
                with col_mw:
                    st.subheader("Most Wanted")
                    if mw_img:
                        if target_height and mw_img.height > target_height:
                            aspect_ratio = mw_img.width / mw_img.height
                            new_width = int(target_height * aspect_ratio)
                            mw_img = mw_img.resize((new_width, target_height))
                        st.image(mw_img, use_container_width=False, caption="Most Wanted")
                    else:
                        st.warning("Image not found.")
                    
                    mw_pred = km_df[(km_df['country'] == selected_country) & (km_df['category'] == 'most_wanted')]
                    if not mw_pred.empty:
                        mp = mw_pred.iloc[0]
                        if has_xgb:
                            st.write(f"**XGBoost Prediction:** ${mp['predicted_price_xgb']:,.0f}")
                        if has_knn:
                            st.write(f"**5-NN Prediction:** ${mp['predicted_price_knn']:,.0f}")
                        if has_cluster:
                            st.write(f"**Cluster-Specific Prediction:** ${mp['predicted_price_cluster']:,.0f}")
                        
                        # Display Neighbors
                        if 'neighbor_indices' in mp and pd.notna(mp['neighbor_indices']):
                            st.markdown("##### 5 Nearest Neighbors (from Auction Dataset)")
                            indices = json.loads(mp['neighbor_indices'])
                            n_cols = st.columns(len(indices))
                            for i, idx in enumerate(indices):
                                with n_cols[i]:
                                    n_row = df.iloc[idx]
                                    n_img = n_row["local_image_path"]
                                    n_url = n_row.get("Image_URL", "")
                                    if Path(n_img).exists():
                                        st.image(Image.open(n_img), use_container_width=True)
                                    elif n_url:
                                        st.image(n_url, use_container_width=True)
                                    st.caption(
                                        f"**{n_row['Artwork_Title'][:40]}**\n\n"
                                        f"*{n_row['Artist_Name']}*\n\n"
                                        f"_{n_row['Method']}_\n\n"
                                        f"${n_row['Sold_Price_USD']:,.0f}"
                                    )

                # Least Wanted
                with col_lw:
                    st.subheader("Least Wanted")
                    if lw_img:
                        if target_height and lw_img.height > target_height:
                            aspect_ratio = lw_img.width / lw_img.height
                            new_width = int(target_height * aspect_ratio)
                            lw_img = lw_img.resize((new_width, target_height))
                        st.image(lw_img, use_container_width=False, caption="Least Wanted")
                    else:
                        st.warning("Image not found.")
                            
                    lw_pred = km_df[(km_df['country'] == selected_country) & (km_df['category'] == 'least_wanted')]
                    if not lw_pred.empty:
                        lp = lw_pred.iloc[0]
                        if has_xgb:
                            st.write(f"**XGBoost Prediction:** ${lp['predicted_price_xgb']:,.0f}")
                        if has_knn:
                            st.write(f"**5-NN Prediction:** ${lp['predicted_price_knn']:,.0f}")
                        if has_cluster:
                            st.write(f"**Cluster-Specific Prediction:** ${lp['predicted_price_cluster']:,.0f}")
                            
                        # Display Neighbors
                        if 'neighbor_indices' in lp and pd.notna(lp['neighbor_indices']):
                            st.markdown("##### 5 Nearest Neighbors (from Auction Dataset)")
                            indices = json.loads(lp['neighbor_indices'])
                            n_cols = st.columns(len(indices))
                            for i, idx in enumerate(indices):
                                with n_cols[i]:
                                    n_row = df.iloc[idx]
                                    n_img = n_row["local_image_path"]
                                    n_url = n_row.get("Image_URL", "")
                                    if Path(n_img).exists():
                                        st.image(Image.open(n_img), use_container_width=True)
                                    elif n_url:
                                        st.image(n_url, use_container_width=True)
                                    st.caption(
                                        f"**{n_row['Artwork_Title'][:40]}**\n\n"
                                        f"*{n_row['Artist_Name']}*\n\n"
                                        f"_{n_row['Method']}_\n\n"
                                        f"${n_row['Sold_Price_USD']:,.0f}"
                                    )

            st.markdown("---")
            # --- High Level Summary ---
            st.subheader("High-Level Summary (Across All Countries)")
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("**Most Wanted Rankings**")
                # Calculate average
                if has_xgb:
                    avg_most_xgb = km_df[km_df['category'] == 'most_wanted']['predicted_price_xgb'].mean()
                    st.metric("Avg Predicted Price (XGB)", f"${avg_most_xgb:,.0f}")
                if has_knn:
                    avg_most_knn = km_df[km_df['category'] == 'most_wanted']['predicted_price_knn'].mean()
                    st.metric("Avg Predicted Price (5-NN)", f"${avg_most_knn:,.0f}")
                if has_cluster:
                    avg_most_cluster = km_df[km_df['category'] == 'most_wanted']['predicted_price_cluster'].mean()
                    st.metric("Avg Predicted (Cluster)", f"${avg_most_cluster:,.0f}")
            
            with c2:
                st.markdown("**Least Wanted Rankings**")
                # Calculate average
                if has_xgb:
                    avg_least_xgb = km_df[km_df['category'] == 'least_wanted']['predicted_price_xgb'].mean()
                    st.metric("Avg Predicted Price (XGB)", f"${avg_least_xgb:,.0f}")
                if has_knn:
                    avg_least_knn = km_df[km_df['category'] == 'least_wanted']['predicted_price_knn'].mean()
                    st.metric("Avg Predicted Price (5-NN)", f"${avg_least_knn:,.0f}")
                if has_cluster:
                    avg_least_cluster = km_df[km_df['category'] == 'least_wanted']['predicted_price_cluster'].mean()
                    st.metric("Avg Predicted (Cluster)", f"${avg_least_cluster:,.0f}")
            
            # --- Per Country Interactive Chart ---
            st.subheader("Price Predictions by Country")
            
            # Transform wide to long for plotly clustered bar chart
            melted_data = []
            for _, row in km_df.iterrows():
                if has_xgb and pd.notna(row['predicted_price_xgb']):
                    melted_data.append({
                        'Country': row['country'],
                        'Category': f"{row['category'].replace('_', ' ').title()}",
                        'Price': row['predicted_price_xgb'],
                        'Model': 'XGBoost (Non-linear)'
                    })
                if has_knn and pd.notna(row['predicted_price_knn']):
                    melted_data.append({
                        'Country': row['country'],
                        'Category': f"{row['category'].replace('_', ' ').title()}",
                        'Price': row['predicted_price_knn'],
                        'Model': '5-NN (Similarity based)'
                    })
                if has_cluster and pd.notna(row['predicted_price_cluster']):
                    melted_data.append({
                        'Country': row['country'],
                        'Category': f"{row['category'].replace('_', ' ').title()}",
                        'Price': row['predicted_price_cluster'],
                        'Model': 'Cluster-Specific Model'
                    })
            
            if melted_data:
                plot_df = pd.DataFrame(melted_data)
                
                # Let user choose which model to plot
                model_options = plot_df['Model'].unique()
                selected_model = st.selectbox("Select Model for Comparison", model_options)
                
                filtered_plot_df = plot_df[plot_df['Model'] == selected_model].sort_values(by="Price", ascending=False)
                
                fig = px.bar(
                    filtered_plot_df, 
                    x='Country', 
                    y='Price', 
                    color='Category',
                    barmode='group',
                    category_orders={"Category": ["Most Wanted", "Least Wanted"]},
                    title=f"Most vs Least Wanted Prices ({selected_model})",
                    color_discrete_map={"Most Wanted": "#1f77b4", "Least Wanted": "#ff7f0e"}
                )
                fig.update_layout(yaxis_title="Predicted Price ($)", xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
            # --- Raw Data ---
            with st.expander("View Raw Data"):
                st.dataframe(
                    km_df.style.format({
                        "predicted_price_xgb": "${:,.0f}",
                        "predicted_price_knn": "${:,.0f}",
                        "predicted_price_image_only": "${:,.0f}",
                    })
                )
        else:
            st.warning("Komar & Melamid prediction results not found. Please run `predict_km_paintings.py` in the `03_elaborating` directory first.")

    # =====================================================================
    # TAB NANO: NanoBanana Prompts
    # =====================================================================
    with tab_nano:
        st.header("NanoBanana Prompts Generation")
        st.write("Generate descriptive prompts for 4 price quartiles inside each cluster to feed into the NanoBanana UI.")
        
        meta_list = load_cluster_metadata(mode)
        pca_2d = load_pca(mode)
        clusters = load_clusters(mode)
        
        cluster_df = pd.DataFrame({
            "Cluster": clusters.astype(str),
            "Price": df["Sold_Price_USD"],
            "Title": df["Artwork_Title"].fillna("Untitled"),
            "Artist": df["Artist_Name"].fillna("Unknown"),
            "Method": df["Method"].fillna("Unknown"),
        })

        if meta_list and not cluster_df.empty:
            c_select_nano = st.selectbox(
                "Select a Cluster to generate prompts", 
                [str(m["cluster_id"]) for m in meta_list],
                key="nano_cluster_select"
            )
            
            c_prices = cluster_df[cluster_df["Cluster"] == c_select_nano]["Price"].dropna()
            
            nano_data_path = MODELS_DIR / f"cluster_nano_artworks_{mode}.json"
            nano_curr = {}
            if nano_data_path.exists():
                with open(nano_data_path, "r") as f:
                    nano_all = json.load(f)
                    nano_curr = nano_all.get(str(c_select_nano), {})
            
            if nano_curr:
                st.subheader(f"Prompts for Cluster {c_select_nano}")
                st.markdown(f"**Base Description:** {nano_curr.get('description', '')}")
                
                tiers = nano_curr.get("tiers", {})
                quartiles = nano_curr.get("quartiles", [0, 0, 0])
                
                tier_configs = [
                    ("Tier 1: Economy", f"Under ${quartiles[0]:,.0f}", "q1"),
                    ("Tier 2: Mid-Market", f"${quartiles[0]:,.0f} - ${quartiles[1]:,.0f}", "q2"),
                    ("Tier 3: Premium", f"${quartiles[1]:,.0f} - ${quartiles[2]:,.0f}", "q3"),
                    ("Tier 4: Masterpiece", f"Over ${quartiles[2]:,.0f}", "q4"),
                ]
                
                for t_name, t_price, t_key in tier_configs:
                    t_data = tiers.get(t_key, {})
                    st.markdown(f"### {t_name} ({t_price})")
                    
                    col_left, col_right = st.columns([2, 1])
                    with col_left:
                        st.markdown("##### LLM Prompt with Visual Context")
                        st.code(t_data.get("prompt", "No prompt generated."), language="text")
                        
                        st.markdown("##### 10 Artworks Used For This Prompt")
                        artworks = t_data.get("artworks", [])
                        if artworks:
                            cols_aw = st.columns(10)
                            for idx, aw in enumerate(artworks[:10]):
                                with cols_aw[idx]:
                                    img_p = Path(aw["image"])
                                    if img_p.exists():
                                        img = Image.open(img_p)
                                        w, h = img.size
                                        sq_size = min(w, h)
                                        left = (w - sq_size)/2
                                        top = (h - sq_size)/2
                                        img = img.crop((left, top, left+sq_size, top+sq_size))
                                        img = img.resize((100, 100))
                                        st.image(img, use_container_width=True)
                                    else:
                                        st.write("No img")
                        else:
                            st.info("No artworks found for this tier.")
                    
                    with col_right:
                        st.markdown("##### AI-Generated Artwork")
                        gen_img_path = MODELS_DIR / f"nano_images/{mode}/cluster_{c_select_nano}_{t_key}.png"
                        if gen_img_path.exists():
                            st.image(Image.open(gen_img_path), caption="NanoBanana AI Result", use_container_width=True)
                        else:
                            st.info(f"AI image not generated yet. (Path: {gen_img_path.name})")
                    st.markdown("---")
            else:
                st.warning("NanoBanana prompt data missing. Please run prompt generation.")
        else:
            st.info("No cluster metadata found. Please run training scripts to generate cluster data.")

if __name__ == "__main__":
    main()
