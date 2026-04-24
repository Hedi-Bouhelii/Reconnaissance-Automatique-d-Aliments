import requests
import os
from tqdm import tqdm

def download_food101():
    """Download Food-101 dataset"""
    url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    zip_path = './data/food-101.tar.gz'
    
    # Create data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    
    print("Downloading Food-101 dataset...")
    print(f"URL: {url}")
    print(f"Destination: {zip_path}")
    
    # Download with progress bar
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as f, tqdm(
        desc="Downloading",
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    
    print(f"Download completed: {zip_path}")
    return zip_path

if __name__ == "__main__":
    download_food101()
