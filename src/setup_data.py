import os
import shutil
import kagglehub

def setup_raw_data():
    # Target directory based on our repo structure
    raw_data_dir = "./data/raw"
    os.makedirs(raw_data_dir, exist_ok=True)
    
    print("Downloading competition dataset via kagglehub...")
    # Note: Make sure your Kaggle API token is set up on your machine!
    # kagglehub will prompt you to log in if it's not configured.
    cache_path = kagglehub.competition_download('mindshift-analytics-haul-mark-challenge')
    
    print(f"Downloaded to cache: {cache_path}")
    print(f"Copying files to {raw_data_dir}...")
    
    # Copy files from the hidden kagglehub cache to your project folder
    for item in os.listdir(cache_path):
        src = os.path.join(cache_path, item)
        dst = os.path.join(raw_data_dir, item)
        
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
            
    print("✅ Success! Competition files are now in your local data/raw/ folder.")

if __name__ == "__main__":
    setup_raw_data()