import os
import re
import csv
import time
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup

BASE_URL = "https://awp.diaart.org/km/"

COUNTRY_MAP = {
    'usa': ('us', 'United States'),
    'fra': ('fr', 'France'),
    'tur': ('tr', 'Turkey'),
    'ice': ('is', 'Iceland'),
    'rus': ('ru', 'Russia'),
    'den': ('dk', 'Denmark'),
    'chi': ('cn', 'China'),
    'ken': ('ke', 'Kenya'),
    'fin': ('fi', 'Finland'),
    'por': ('pt', 'Portugal'),
    'hol': ('nl', 'Netherlands'),
    'ukr': ('ua', 'Ukraine'),
    'ita': ('it', 'Italy'),
    'ger': ('de', 'Germany'),
    'web': ('web', 'The Web')
}

def get_image_url(page_url, category):
    try:
        resp = requests.get(page_url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Find all images
        for img in soup.find_all('img'):
            src = img.get('src', '').lower()
            if 'sm' not in src and 'icon' not in src and src.endswith('.jpg'):
                return urljoin(page_url, img.get('src'))
        
        # Fallback to regex
        matches = re.findall(r'<IMG[^>]+SRC=[\'"]?([^\'">]+\.jpg)[\'"]?', resp.text, re.IGNORECASE)
        for src in matches:
            if 'sm' not in src.lower():
                return urljoin(page_url, src)
        
        # If all else fails, just take the first jpg
        for src in matches:
            return urljoin(page_url, src)

    except Exception as e:
        print(f"Failed to fetch {page_url}: {e}")
    return None

def download_image(url, save_path):
    if os.path.exists(save_path):
        return True
    try:
        resp = requests.get(url, stream=True, timeout=10)
        resp.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(base_dir, 'images')
    most_dir = os.path.join(images_dir, 'most_wanted')
    least_dir = os.path.join(images_dir, 'least_wanted')
    
    os.makedirs(most_dir, exist_ok=True)
    os.makedirs(least_dir, exist_ok=True)
    
    csv_rows = []
    
    for prefix, (code, name) in COUNTRY_MAP.items():
        for category in ['most', 'least']:
            cat_name = f"{category}_wanted"
            page_url = f"{BASE_URL}{prefix}/{category}.html"
            
            print(f"Processing {name} ({cat_name})...")
            img_url = get_image_url(page_url, category)
            
            if not img_url:
                print(f"  -> Could not find image URL for {page_url}")
                continue
                
            print(f"  -> Found image: {img_url}")
            
            save_dir = most_dir if category == 'most' else least_dir
            filename = f"{code}.jpg"
            save_path = os.path.join(save_dir, filename)
            rel_path = f"03_elaborating/images/{cat_name}/{filename}"
            
            if download_image(img_url, save_path):
                csv_rows.append({
                    'country': name,
                    'country_code': code,
                    'category': cat_name,
                    'image_path': rel_path
                })
            time.sleep(1) # Be polite
            
    csv_path = os.path.join(base_dir, 'km_paintings.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['country', 'country_code', 'category', 'image_path'])
        writer.writeheader()
        writer.writerows(csv_rows)
        
    print(f"Dataset preparation complete. Metadata saved to {csv_path}")

if __name__ == '__main__':
    main()
