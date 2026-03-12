import csv
import os
from datetime import datetime
import os

INPUT_CSV = "artworks_data.csv"
OUTPUT_CSV = "artworks_data_clean.csv"

def get_row_key(row):
    return (
        ' '.join(row.get('Image_URL', '').split()),
        ' '.join(row.get('Artwork_Title', '').split()),
        ' '.join(row.get('Method', '').split()),
        ' '.join(row.get('Artist_Name', '').split()),
        ' '.join(row.get('Year_Made', '').split()),
        ' '.join(row.get('Sold_Date', '').split()),
        ' '.join(row.get('Sold_Price_USD', '').split()),
        ' '.join(row.get('Auction_Name', '').split())
    )

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found.")
        return

    seen_keys = set()
    cleaned_rows = []
    duplicate_count = 0
    total_count = 0

    print(f"Reading {INPUT_CSV} to remove duplicates (ignoring Source_File)...")

    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            total_count += 1
            key = get_row_key(row)
            
            if key in seen_keys:
                duplicate_count += 1
            else:
                seen_keys.add(key)
                cleaned_rows.append(row)

    print(f"\nTotal rows processed: {total_count}")
    print(f"Duplicates found and removed: {duplicate_count}")
    print(f"Unique rows remaining: {len(cleaned_rows)}")
    
    # Sort the rows by Sold_Date sequentially
    print("Sorting rows sequentially by Sold_Date...")
    def parse_date(date_str):
        try:
            return datetime.strptime(date_str, "%d %B %Y")
        except ValueError:
            # Fallback for unexpected formats (sort to end/beginning depending)
            return datetime.min
            
    cleaned_rows.sort(key=lambda x: parse_date(x.get('Sold_Date', '')))

    # Write cleaned data to a new CSV file
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    print(f"\nCleaned dataset saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
