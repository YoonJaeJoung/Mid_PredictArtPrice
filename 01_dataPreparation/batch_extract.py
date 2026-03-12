import os
import csv
from bs4 import BeautifulSoup
import re
from datetime import datetime

INPUT_DIR = "raw DOM"
CSV_FILE = "artworks_data_rewritten.csv"

def parse_dom(dom_html, file_path):
    soup = BeautifulSoup(dom_html, 'html.parser')
    
    h2_tags = soup.find_all('h2')
    items = []
    for h2 in h2_tags:
        if h2.text.strip() and h2.parent and h2.parent.name == 'div':
            # The artwork container is usually 4 levels up
            container = h2.parent.parent.parent.parent
            if container and container not in items:
                items.append(container)

    results = []
    
    for item in items:
        # Image URL
        img = item.find('img')
        img_url = img.get('src') if img else "No image found"
        
        texts = [t for t in item.strings if t.strip()]
        
        artist = "Unknown Artist"
        title = "Unknown Title"
        year_made = "Unknown Year"
        method = "Unknown Method"
        sold_date = "Unknown Date"
        auction = "Unknown Auction"
        sold_price = "N/A"
        
        # Artist
        h2 = item.find('h2')
        if h2: artist = h2.get_text(strip=True)
        
        # Title
        h3 = item.find('h3')
        if h3:
             title_text = h3.get_text(separator="|", strip=True)
             parts = title_text.split('|')
             title = parts[0]
        
        # Auction Name
        auction_div = item.find('div', class_=re.compile(r'min-w-full leading-normal'))
        if auction_div:
            auction = auction_div.get_text(strip=True)
                  
        for i, text in enumerate(texts):
             text = text.strip()
             if not text: continue
             
             lower_text = text.lower()
             if "on canvas" in lower_text or "on board" in lower_text or "gouache" in lower_text or "oil" in lower_text or "acrylic" in lower_text or "bronze" in lower_text or "print" in lower_text or "lithograph" in lower_text or "collage" in lower_text:
                  if method == "Unknown Method": method = text
                  
             if re.match(r'^(18|19|20)\d{2}$', text) and len(text) == 4:
                  if year_made == "Unknown Year": year_made = text
                  
             if re.match(r'^\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(18|19|20)\d{2}$', text, re.I):
                  sold_date = text
                  
             # Price
             if text == "USD" and i > 0:
                  potential_price = texts[i-1].replace(',', '').strip()
                  if potential_price.isdigit():
                       sold_price = f"{texts[i-1]}"
                       
        # Fallback for Method and Year_Made if they are still unknown
        if method == "Unknown Method" or year_made == "Unknown Year":
            details_divs = item.find_all('div', class_=re.compile(r'pb-1'))
            for div in details_divs:
                div_class = " ".join(div.get("class", []))
                div_text = div.get_text(strip=True)
                if not div_text: continue
                
                # Method fallback
                if method == "Unknown Method" and "hidden" in div_class and "sm:inline-block" in div_class and "print:inline-block" in div_class:
                    if "USD" not in div_text and "est." not in div_text:
                        method = div_text
                
                # Year fallback
                elif year_made == "Unknown Year" and "hidden" in div_class and "sm:block" in div_class and "print:block" in div_class:
                    if "USD" not in div_text and "est." not in div_text:
                        year_made = div_text
                       
        if sold_price == "N/A":
             # Fallback
             for i, text in enumerate(texts):
                  if text.replace(',', '').isdigit() and i + 1 < len(texts) and texts[i+1] == "USD":
                       sold_price = f"{text}"
                       break
        
        if artist != "Unknown Artist":
            results.append({
                'Image_URL': img_url,
                'Artwork_Title': title,
                'Method': method,
                'Artist_Name': artist,
                'Year_Made': year_made,
                'Sold_Date': sold_date,
                'Sold_Price_USD': sold_price,
                'Auction_Name': auction,
                'Source_File': os.path.basename(file_path)
            })
            
    return results

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        return

    fieldnames = ['Image_URL', 'Artwork_Title', 'Method', 'Artist_Name', 'Year_Made', 'Sold_Date', 'Sold_Price_USD', 'Auction_Name', 'Source_File']
    
    # Initialize the CSV file with headers
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    html_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.html')]
    total_files = len(html_files)
    
    batch_size = 100
    current_batch = []
    processed_count = 0

    print("=" * 64)
    print(" Starting batch extraction to rewrite CSV from raw DOM files.")
    print("=" * 64)
    print(f"Total files to process: {total_files}\n")

    for filename in html_files:
        file_path = os.path.join(INPUT_DIR, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                dom_html = f.read()
            
            parsed_data = parse_dom(dom_html, file_path)
            current_batch.extend(parsed_data)
            processed_count += 1
            print(f"[{processed_count}/{total_files}] Processing: {filename}")
            
            if processed_count % batch_size == 0:
                with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    writer.writerows(current_batch)
                print(f"--- Flushed {len(current_batch)} artworks to CSV (Processed {processed_count} files) ---")
                current_batch = []
                
        except Exception as e:
            print(f"Error reading/parsing {filename}: {e}")

    # Flush remaining
    if current_batch:
        with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writerows(current_batch)
        print(f"--- Flushed final {len(current_batch)} artworks to CSV (Processed {processed_count} files) ---")

    print(f"\nSuccessfully processed {processed_count} files.")
    print(f"Data appended to '{CSV_FILE}'.")

if __name__ == "__main__":
    main()
