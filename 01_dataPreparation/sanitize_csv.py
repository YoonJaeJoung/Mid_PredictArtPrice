import csv
import os

def sanitize_csv(file_path):
    print(f"Sanitizing {file_path}...")
    temp_file = file_path + ".tmp"
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as fin:
            # We use the csv module which handles quoted newlines correctly
            reader = csv.reader(fin)
            header = next(reader)
            
            with open(temp_file, 'w', encoding='utf-8', newline='') as fout:
                # Use QUOTE_MINIMAL but we will manually clean fields
                writer = csv.writer(fout, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(header)
                
                count = 0
                for row in reader:
                    # Clean each field: remove newlines/tabs and strip whitespace
                    cleaned_row = []
                    for field in row:
                        if field:
                            # Replace any newline/carriage return/tab with a space
                            cleaned_field = field.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
                            # Replace multiple spaces with one
                            cleaned_field = ' '.join(cleaned_field.split())
                            cleaned_row.append(cleaned_field)
                        else:
                            cleaned_row.append("")
                    
                    writer.writerow(cleaned_row)
                    count += 1
        
        # Replace original with fixed version
        os.replace(temp_file, file_path)
        print(f"Successfully sanitized {count} rows in {file_path}")
        
    except Exception as e:
        print(f"Error sanitizing {file_path}: {e}")
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    files_to_fix = [
        "01_dataPreparation/artworks_data.csv",
        "01_dataPreparation/artworks_data_clean.csv"
    ]
    for csv_file in files_to_fix:
        if os.path.exists(csv_file):
            sanitize_csv(csv_file)
        else:
            print(f"File not found: {csv_file}")
