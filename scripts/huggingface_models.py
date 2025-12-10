from huggingface_hub import HfApi, create_repo
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config

def upload_models_to_huggingface(repo_id: str, token: str):
    config = get_config()
    api = HfApi()
    
    try:
        create_repo(repo_id=repo_id, token=token, repo_type="model", exist_ok=True)
        print(f"Repository created: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Repository might already exist: {e}")
    
    models_to_upload = [
        ("random_forest_classifier.pkl", "Random Forest classifier"),
        ("yolov8n-face.pt", "YOLOv8 face detection model"),
        ("feature_scaler.pkl", "Feature scaler"),
    ]
    
    for filename, description in models_to_upload:
        model_path = config.paths.models_dir / filename
        
        if not model_path.exists():
            print(f"Skipping {filename} (not found)")
            continue
        
        print(f"Uploading {description}...")
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=filename,
            repo_id=repo_id,
            token=token
        )
        print(f"  Uploaded: {filename}")
    
    print(f"\nAll models uploaded to: https://huggingface.co/{repo_id}")

def download_models_from_huggingface(repo_id: str):
    from huggingface_hub import hf_hub_download
    config = get_config()
    
    models_to_download = [
        "random_forest_classifier.pkl",
        "yolov8n-face.pt",
        "feature_scaler.pkl",
    ]
    
    config.paths.models_dir.mkdir(parents=True, exist_ok=True)
    
    for filename in models_to_download:
        try:
            print(f"Downloading {filename}...")
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=str(config.paths.models_dir)
            )
            print(f"  Downloaded: {downloaded_path}")
        except Exception as e:
            print(f"  Failed to download {filename}: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload/Download models to/from HuggingFace Hub")
    parser.add_argument("action", choices=["upload", "download"], help="Action to perform")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID (username/repo-name)")
    parser.add_argument("--token", help="HuggingFace token (for upload)")
    
    args = parser.parse_args()
    
    if args.action == "upload":
        if not args.token:
            print("Error: --token required for upload")
            print("Get token from: https://huggingface.co/settings/tokens")
            sys.exit(1)
        upload_models_to_huggingface(args.repo_id, args.token)
    else:
        download_models_from_huggingface(args.repo_id)
