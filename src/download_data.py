import os
import requests
import zipfile
import pandas as pd

def main():
    os.makedirs('data', exist_ok=True)
    dataset_path = 'data/MetroPT3(AirCompressor).csv'
    
    if os.path.exists(dataset_path):
        print(f"Dataset already exists at {dataset_path}")
        return

    zip_url = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
    zip_path = "data/metropt3.zip"
    
    print("Downloading MetroPT-3 Dataset from UCI...")
    response = requests.get(zip_url, stream=True)
    if response.status_code == 200:
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete. Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('data/')
        os.remove(zip_path)
        print(f"Extracted dataset. Ensure the CSV is named properly at {dataset_path}")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

if __name__ == '__main__':
    main()
