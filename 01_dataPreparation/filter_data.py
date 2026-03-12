import csv
import os

REWRITTEN_CSV = "deprecated/artworks_data_rewritten.csv"
EXCLUDE_CSV = "deprecated/artworks_exclude.csv"
FINAL_CSV = "artworks_data.csv"

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
    exclude_keys = set()
    try:
        with open(EXCLUDE_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                exclude_keys.add(get_row_key(row))
    except FileNotFoundError:
        print(f"{EXCLUDE_CSV} not found. Proceeding without exclusions.")

    print(f"Loaded {len(exclude_keys)} unique artwork signatures to exclude from {EXCLUDE_CSV}.")

    if not os.path.exists(REWRITTEN_CSV):
        print(f"{REWRITTEN_CSV} not found.")
        return

    written = 0
    excluded_count = 0
    with open(REWRITTEN_CSV, "r", encoding="utf-8") as in_csv, \
         open(FINAL_CSV, "w", encoding="utf-8", newline="") as out_csv:
        reader = csv.DictReader(in_csv)
        writer = csv.DictWriter(out_csv, fieldnames=reader.fieldnames)
        writer.writeheader()
        
        for row in reader:
            if get_row_key(row) not in exclude_keys:
                writer.writerow(row)
                written += 1
            else:
                excluded_count += 1

    print(f"Excluded {excluded_count} artworks based on matching criteria.")
    print(f"Saved {written} filtered artworks to {FINAL_CSV}.")

if __name__ == "__main__":
    main()
