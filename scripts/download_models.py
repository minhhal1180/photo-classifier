import urllib.request
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config

def download_file(url: str, destination: Path):
    print(f"Downloading {destination.name}...")
    
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        sys.stdout.write(f"\r  Progress: {percent:.1f}%")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, destination, reporthook)
    print(f"Downloaded: {destination}")

def download_yolo_face_model():
    config = get_config()
    model_path = config.paths.yolo_face_model
    
    if model_path.exists():
        print(f"YOLOv8 model exists: {model_path}")
        return
    
    print("YOLOv8 will auto-download on first use")

def main():
    config = get_config()
    config.paths.models_dir.mkdir(parents=True, exist_ok=True)
    download_yolo_face_model()
    print("Model setup completed")

if __name__ == '__main__':
    main()
