import os
import gdown
import zipfile
import shutil

def fetch_and_prepare_data():
    """Fetch data from Google Drive and prepare it for training"""
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Google Drive file ID from the link
    file_id = '1dhNlD9-r1d50i5fTym-KJoDPLvkJyvKW'
    
    # Download path
    zip_path = 'data/main_data.zip'
    
    # Download the file
    print("Downloading data from Google Drive...")
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, zip_path, quiet=False)
    
    # Extract the zip file
    print("Extracting data...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('data')
    
    # Remove the zip file
    os.remove(zip_path)
    print("Data preparation completed!")

if __name__ == "__main__":
    fetch_and_prepare_data() 