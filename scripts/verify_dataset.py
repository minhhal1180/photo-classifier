import sys
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def verify_dataset():
    data_dir = project_root / "data" / "training_images"
    
    if not data_dir.exists():
        print("Error: data/training_images not found!")
        print("Please download dataset first: python scripts/download_data.py")
        return False
    
    categories = ["ChanDung", "TinhVat", "TheThao", "PhongCanh", "DongVat"]
    
    print("=" * 60)
    print("DATASET VERIFICATION")
    print("=" * 60)
    
    total_images = 0
    total_size = 0
    file_types = defaultdict(int)
    
    for category in categories:
        category_dir = data_dir / category
        
        if not category_dir.exists():
            print(f"\n{category}: MISSING")
            continue
        
        files = list(category_dir.glob("*"))
        count = len(files)
        total_images += count
        
        category_size = sum(f.stat().st_size for f in files if f.is_file())
        total_size += category_size
        
        for f in files:
            if f.is_file():
                file_types[f.suffix.lower()] += 1
        
        print(f"\n{category}:")
        print(f"  Images: {count}")
        print(f"  Size: {category_size / (1024**3):.2f} GB")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total images: {total_images}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    
    print("\nFile types:")
    for ext, count in sorted(file_types.items()):
        print(f"  {ext}: {count} files")
    
    metadata_csv = project_root / "data" / "metadata" / "exif_data_complete.csv"
    if metadata_csv.exists():
        print(f"\nMetadata: Found")
        import pandas as pd
        try:
            df = pd.read_csv(metadata_csv)
            print(f"  Rows: {len(df)}")
        except:
            print(f"  Error reading CSV")
    else:
        print(f"\nMetadata: NOT FOUND")
    
    if total_images == 0:
        print("\nStatus: FAILED - No images found")
        return False
    elif total_images < 500:
        print(f"\nStatus: WARNING - Only {total_images} images (recommend 500+)")
        return True
    else:
        print(f"\nStatus: OK")
        return True

if __name__ == "__main__":
    success = verify_dataset()
    sys.exit(0 if success else 1)
