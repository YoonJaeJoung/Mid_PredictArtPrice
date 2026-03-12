import pandas as pd
import os

def add_id_column(csv_path):
    print(f"Reading {csv_path}...")
    # Read the CSV. 
    # Based on the head output, it seems to have a header.
    df = pd.read_csv(csv_path)
    
    # Check if 'id' already exists
    if 'id' in df.columns:
        print("'id' column already exists. Skipping.")
        return

    # Add 'id' column at the beginning (index 0)
    # Using 1-based indexing for IDs
    df.insert(0, 'id', range(1, len(df) + 1))
    
    print(f"Added {len(df)} IDs. Saving back to {csv_path}...")
    df.to_csv(csv_path, index=False)
    print("Done.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(current_dir, "artworks_data_clean.csv")
    
    if os.path.exists(csv_file):
        add_id_column(csv_file)
    else:
        print(f"Error: {csv_file} not found.")
