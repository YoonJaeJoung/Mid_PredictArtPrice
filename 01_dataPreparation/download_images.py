import pandas as pd
import os
import asyncio
from tqdm import tqdm
from playwright.async_api import async_playwright

async def download_image(semaphore, context, url, save_path):
    async with semaphore:
        if os.path.exists(save_path):
            return True
        
        modified_url = url.replace("width=212,height=282", "width=300")
        
        try:
            # context.request is MUCH faster than context.new_page + page.goto
            response = await context.request.get(modified_url, timeout=30000)
            
            if response.status == 200:
                body = await response.body()
                with open(save_path, 'wb') as f:
                    f.write(body)
                return True
            else:
                return False
        except Exception:
            return False

async def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(current_dir, "artworks_data_clean.csv")
    image_dir = os.path.join(current_dir, "image")
    
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        
    df = pd.read_csv(csv_file)
    image_url_col = 'Image URL' if 'Image URL' in df.columns else df.columns[1]
    id_col = 'id'
    
    print("Checking for existing images to skip...")
    existing_files = set(os.listdir(image_dir))
    
    pending_tasks_data = []
    for _, row in df.iterrows():
        img_id = str(row[id_col])
        filename = f"{img_id}.jpg"
        if filename not in existing_files:
            url = row[image_url_col]
            save_path = os.path.join(image_dir, filename)
            pending_tasks_data.append((url, save_path))
    
    total_skipped = len(df) - len(pending_tasks_data)
    print(f"Skipping {total_skipped} images. Remaining: {len(pending_tasks_data)}")
    
    if not pending_tasks_data:
        print("All images already downloaded.")
        return

    # ULTRA SPEED: Using 200 concurrent API requests
    CONCURRENCY_LIMIT = 200
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    async with async_playwright() as p:
        # Launching browser only to get the context/request capability
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )
        
        tasks = [download_image(semaphore, context, url, path) for url, path in pending_tasks_data]
        
        success_count = 0
        with tqdm(total=len(tasks), desc="Downloading", unit="img", smoothing=0.05) as pbar:
            for task in asyncio.as_completed(tasks):
                result = await task
                if result:
                    success_count += 1
                pbar.update(1)
                
        await browser.close()
        
    print(f"\nDownload complete. Success: {success_count} new images.")

if __name__ == "__main__":
    asyncio.run(main())
