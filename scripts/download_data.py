import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

GOOGLE_DRIVE_FILE_ID = "YOUR_FILE_ID_HERE"
OUTPUT_ZIP = project_root / "data" / "training_data.zip"
EXTRACT_TO = project_root / "data"

def download_from_google_drive(file_id: str, output_path: Path):
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
    
    print(f"Downloading dataset from Google Drive...")
    print(f"File ID: {file_id}")
    
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, str(output_path), quiet=False)
    
    print(f"Downloaded to: {output_path}")

def extract_zip(zip_path: Path, extract_to: Path):
    import zipfile
    
    print(f"Extracting {zip_path.name}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print(f"Extracted to: {extract_to}")
    
    zip_path.unlink()
    print("Removed ZIP file")

def verify_dataset():
    data_dir = project_root / "data" / "training_images"
    
    if not data_dir.exists():
        print("Error: training_images directory not found!")
        return False
    
    categories = ["ChanDung", "TinhVat", "TheThao", "PhongCanh", "DongVat"]
    total_images = 0
    
    print("\nDataset verification:")
    for category in categories:
        category_dir = data_dir / category
        if category_dir.exists():
            count = len(list(category_dir.glob("*")))
            total_images += count
            print(f"  {category}: {count} images")
        else:
            print(f"  {category}: MISSING")
    
    print(f"\nTotal: {total_images} images")
    return total_images > 0

def main():
    if GOOGLE_DRIVE_FILE_ID == "YOUR_FILE_ID_HERE":
        print("Error: Please update GOOGLE_DRIVE_FILE_ID in this script")
        print("\nHow to get File ID:")
        print("1. Upload data ZIP to Google Drive")
        print("2. Right click -> Share -> Get link")
        print("3. Copy the file ID from URL:")
        print("   https://drive.google.com/file/d/FILE_ID_HERE/view")
        print("4. Update GOOGLE_DRIVE_FILE_ID in this script")
        return
    
    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    
    if OUTPUT_ZIP.exists():
        print(f"ZIP already exists: {OUTPUT_ZIP}")
        response = input("Re-download? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download")
        else:
            download_from_google_drive(GOOGLE_DRIVE_FILE_ID, OUTPUT_ZIP)
            extract_zip(OUTPUT_ZIP, EXTRACT_TO)
    else:
        download_from_google_drive(GOOGLE_DRIVE_FILE_ID, OUTPUT_ZIP)
        extract_zip(OUTPUT_ZIP, EXTRACT_TO)
    
    verify_dataset()
    print("\nDataset ready!")

if __name__ == "__main__":
    main()
