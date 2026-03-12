import pandas as pd
import plotly.express as px
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "km_price_predictions.csv"

def main():
    if not CSV_PATH.exists():
        print(f"Error: {CSV_PATH} not found. Run predictions script first.")
        return

    df = pd.read_csv(CSV_PATH)

    # Format country names if necessary
    df["category_label"] = df["category"].apply(lambda x: "Most Wanted" if x == "most_wanted" else "Least Wanted")

    # Bar chart
    fig = px.bar(
        df, 
        x="country", 
        y="predicted_price_image_only", 
        color="category_label",
        barmode="group",
        title="Predicted Auction Prices: Most Wanted vs Least Wanted Paintings",
        labels={
            "country": "Country",
            "predicted_price_image_only": "Predicted Price (USD)",
            "category_label": "Category"
        },
        color_discrete_map={
            "Most Wanted": "#2E86C1", 
            "Least Wanted": "#E74C3C"
        }
    )

    fig.update_layout(
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
        xaxis_tickangle=-45,
        template="plotly_white",
        legend_title_text=""
    )

    # Calculate and display the overall trend annotation
    most_avg = df[df["category"] == "most_wanted"]["predicted_price_image_only"].mean()
    least_avg = df[df["category"] == "least_wanted"]["predicted_price_image_only"].mean()

    diff_percent = ((most_avg - least_avg) / least_avg) * 100 if least_avg else 0
    trend_text = f"Avg Most Wanted: ${most_avg:,.0f}<br>Avg Least Wanted: ${least_avg:,.0f}<br>Difference: {diff_percent:+.1f}%"

    fig.add_annotation(
        text=trend_text,
        xref="paper", yref="paper",
        x=0.98, y=0.95,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=12)
    )

    # Save to HTML and show
    output_html = SCRIPT_DIR / "km_price_visualization.html"
    fig.write_html(str(output_html))
    print(f"[INFO] Saved interactive visualization to {output_html}")
    
    # Try to open it in the default browser natively if not running headless
    import webbrowser
    webbrowser.open(output_html.absolute().as_uri())

if __name__ == "__main__":
    main()
