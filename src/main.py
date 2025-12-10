import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.workflows.training_workflow import TrainingWorkflow
from src.workflows.inference_workflow import InferenceWorkflow
from src.utils.logger import setup_logger

def train_command(args):
    
    logger = setup_logger("main")
    logger.info("=" * 60)
    logger.info("PHOTO CLASSIFIER - TRAINING MODE")
    logger.info("=" * 60)
    
    workflow = TrainingWorkflow(config_path=args.config)
    
    results = workflow.run(
        extract_exif=not args.skip_exif,
        detect_faces=args.face_detect,
        save_model=not args.no_save
    )
    
    logger.info("\nTraining completed successfully!")
    logger.info(f"Train Accuracy: {results['train_accuracy']:.4f}")
    logger.info(f"Test Accuracy: {results['test_accuracy']:.4f}")

def classify_command(args):
    
    logger = setup_logger("main")
    logger.info("=" * 60)
    logger.info("PHOTO CLASSIFIER - INFERENCE MODE")
    logger.info("=" * 60)
    
    workflow = InferenceWorkflow(config_path=args.config)
    
    results = workflow.run(
        extract_exif=not args.skip_exif,
        detect_faces=args.face_detect,
        recognize_faces=args.recognize,
        organize_by_category=args.organize_category,
        organize_by_person=args.organize_person,
        copy_files=not args.move
    )
    
    logger.info("\nClassification completed successfully!")
    logger.info(f"Total images processed: {results['total_images']}")
    logger.info(f"Results saved to: {results['output_dir']}")

def info_command(args):
    
    config = Config(config_path=args.config)
    
    print("\n" + "=" * 60)
    print("PHOTO CLASSIFIER - CONFIGURATION INFO")
    print("=" * 60)
    
    print("\nPaths:")
    print(f"  Training Images: {config.paths.training_images}")
    print(f"  Inference Images: {config.paths.inference_images}")
    print(f"  Models Directory: {config.paths.models_dir}")
    print(f"  Known Faces: {config.paths.known_faces}")
    
    print("\nModel:")
    print(f"  Type: {config.model.type}")
    print(f"  Trees: {config.model.random_forest['n_estimators']}")
    print(f"  Max Depth: {config.model.random_forest['max_depth']}")
    
    print("\nFace Detection:")
    print(f"  Enabled: {config.face_detection.enabled}")
    if config.face_detection.enabled:
        print(f"  Device: {config.face_detection.device}")
        print(f"  Model: {config.face_detection.model_name}")
        print(f"  Confidence: {config.face_detection.confidence_threshold}")
    
    print("\nFace Recognition:")
    print(f"  Enabled: {config.face_recognition.enabled}")
    if config.face_recognition.enabled:
        print(f"  Model: {config.face_recognition.model_name}")
        print(f"  Distance Metric: {config.face_recognition.distance_metric}")
    
    print("\n" + "=" * 60)

def main():
    
    parser = argparse.ArgumentParser(
        description="Photo Classifier - AI-powered photo categorization system"
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file (default: config.yaml)'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the classifier model')
    train_parser.add_argument(
        '--skip-exif',
        action='store_true',
        help='Skip EXIF extraction (use existing data)'
    )
    train_parser.add_argument(
        '--no-face-detect',
        dest='face_detect',
        action='store_false',
        help='Disable face detection during training'
    )
    train_parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save the trained model'
    )
    train_parser.set_defaults(face_detect=True)
    
    # Classify command
    classify_parser = subparsers.add_parser('classify', help='Classify new photos')
    classify_parser.add_argument(
        '--skip-exif',
        action='store_true',
        help='Skip EXIF extraction (use existing data)'
    )
    classify_parser.add_argument(
        '--no-face-detect',
        dest='face_detect',
        action='store_false',
        help='Disable face detection'
    )
    classify_parser.add_argument(
        '--recognize',
        action='store_true',
        help='Enable face recognition'
    )
    classify_parser.add_argument(
        '--no-organize-category',
        dest='organize_category',
        action='store_false',
        help='Do not organize by category'
    )
    classify_parser.add_argument(
        '--organize-person',
        action='store_true',
        help='Organize by person (requires --recognize)'
    )
    classify_parser.add_argument(
        '--move',
        action='store_true',
        help='Move files instead of copying'
    )
    classify_parser.set_defaults(
        face_detect=True,
        organize_category=True
    )
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Display configuration info')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'classify':
        classify_command(args)
    elif args.command == 'info':
        info_command(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
