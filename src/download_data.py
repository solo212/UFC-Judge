import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Define the directory where the dataset will be stored
DATA_DIR = "../data"  # Adjust the path if needed
DATASET_NAME = "rajeevw/ufcdata"  # Replace with the Kaggle dataset ID

def download_dataset():
    # Ensure the target directory exists
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created directory: {DATA_DIR}")
    
    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download the dataset to the specified directory
    print("Downloading dataset...")
    api.dataset_download_files(DATASET_NAME, path=DATA_DIR, unzip=True)
    print(f"Dataset downloaded and extracted to: {DATA_DIR}")

if __name__ == "__main__":
    download_dataset()
