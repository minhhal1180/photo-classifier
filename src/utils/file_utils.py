import shutil
from pathlib import Path
from typing import List, Optional

def ensure_unique_path(path: Path) -> Path:
    candidate = path
    counter = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        counter += 1
    return candidate

def create_directory_structure(base_dir: Path, categories: list, time_buckets: list, create_by_person: bool = False):
    base_dir.mkdir(parents=True, exist_ok=True)
    
    for category in categories:
        category_name = category.get('display_name', category['name'])
        category_path = base_dir / category_name
        
        # Tạo thư mục time buckets
        for bucket in time_buckets:
            (category_path / bucket).mkdir(parents=True, exist_ok=True)
        
        # Tạo thư mục ByPerson cho chân dung
        if create_by_person and category['name'] == 'ChanDung':
            (category_path / 'ByPerson').mkdir(parents=True, exist_ok=True)

def get_supported_image_files(directory: Path, extensions: List[str]) -> List[Path]:
    image_files = []
    
    if not directory.exists():
        return image_files
    
    for ext in extensions:
        # Tìm cả lowercase và uppercase
        image_files.extend(directory.rglob(f"*{ext.lower()}"))
        image_files.extend(directory.rglob(f"*{ext.upper()}"))
    
    return sorted(set(image_files))

def move_file(source: Path, destination: Path, overwrite: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    if not overwrite and destination.exists():
        destination = ensure_unique_path(destination)
    
    shutil.move(str(source), str(destination))
    return destination

def copy_file(source: Path, destination: Path, overwrite: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    if not overwrite and destination.exists():
        destination = ensure_unique_path(destination)
    
    shutil.copy2(str(source), str(destination))
    return destination

def get_file_size_mb(file_path: Path) -> float:
    if not file_path.exists():
        return 0.0
    
    return file_path.stat().st_size / (1024 * 1024)

def clean_directory(directory: Path, keep_extensions: Optional[List[str]] = None):
    if not directory.exists():
        return
    
    for item in directory.iterdir():
        if item.is_file():
            if keep_extensions is None or item.suffix.lower() in keep_extensions:
                continue
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
