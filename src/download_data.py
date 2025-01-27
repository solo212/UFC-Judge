import os
from kaggle.api.kaggle_api_extended import KaggleApi


DATA_DIR = "../data" 
DATASET_NAME = "rajeevw/ufcdata"  

def download_dataset():

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")
    

    api = KaggleApi()
    api.authenticate()

  
    print("Downloading dataset...")
    api.dataset_download_files(DATASET_NAME, path=DATA_DIR, unzip=True)
    print(f"Dataset downloaded and extracted to: {DATA_DIR}")

if __name__ == "__main__":
    download_dataset()
