import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config
from src.utils.logger import setup_logger

def create_directory_structure():
    config = get_config()
    
    directories = [
        config.paths.training_exports,
        config.paths.inference_images,
        config.paths.inference_exports,
        config.paths.known_faces,
        config.paths.models_dir,
        config.paths.metadata_dir,
        project_root / 'logs',
        project_root / 'notebooks',
        project_root / 'tests',
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def create_readme_files():
    config = get_config()
    
    readmes = {
        config.paths.known_faces / 'README.md': ,
        
        config.paths.inference_images / 'README.md': ,
        
        config.paths.models_dir / 'README.md': 
    }
    
    for path, content in readmes.items():
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)

def main():
    print("Setup starting...")
    create_directory_structure()
    create_readme_files()
    print("Setup completed")

if __name__ == '__main__':
    main()
