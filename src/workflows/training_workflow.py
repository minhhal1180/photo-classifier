import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
from tqdm import tqdm

from ..config import Config
from ..data_processing.exif_extractor import ExifExtractor
from ..data_processing.feature_engineering import FeatureEngineer
from ..face_detection.yolo_face_detector import YOLOFaceDetector
from ..models.trainer import ModelTrainer
from ..utils.logger import setup_logger

class TrainingWorkflow:
    
    def __init__(self, config_path: str = "config.yaml"):
        
        self.config = Config(config_path)
        self.logger = setup_logger("training_workflow")
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.config.paths.training_exports / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Training output directory: {self.output_dir}")
    
    def run(
        self,
        extract_exif: bool = True,
        detect_faces: bool = True,
        save_model: bool = True
    ) -> dict:
        
        self.logger.info("=" * 60)
        self.logger.info("STARTING TRAINING WORKFLOW")
        self.logger.info("=" * 60)
        
        # Step 1: Extract EXIF
        if extract_exif:
            self.logger.info("\nStep 1: Extracting EXIF data...")
            exif_csv_path = self._extract_exif()
        else:
            # Use existing EXIF file
            exif_csv_path = self._find_latest_exif_file()
            self.logger.info(f"Using existing EXIF file: {exif_csv_path}")
        
        # Step 2: Load and engineer features
        self.logger.info("\nStep 2: Engineering features...")
        df = self._engineer_features(exif_csv_path)
        
        # Step 3: Extract category labels from folder structure
        self.logger.info("\nStep 3: Extracting category labels...")
        df = self._extract_category_labels(df)
        
        # Step 4: Face detection (optional)
        if detect_faces and self.config.face_detection.enabled:
            self.logger.info("\nStep 4: Detecting faces...")
            df = self._detect_faces(df)
        else:
            self.logger.info("\nStep 4: Skipping face detection")
            df['HasFace'] = 0
            df['NumFaces'] = 0
            df['FaceAreaRatio'] = 0.0
        
        # Step 5: Prepare features
        self.logger.info("\nStep 5: Preparing features...")
        feature_columns = self._get_feature_columns()
        
        # Step 6: Train model
        self.logger.info("\nStep 6: Training model...")
        training_results = self._train_model(df, feature_columns)
        
        # Step 7: Save model
        if save_model:
            self.logger.info("\nStep 7: Saving model...")
            self._save_model()
        
        # Step 8: Generate reports
        self.logger.info("\nStep 8: Generating reports...")
        self._generate_reports()
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("TRAINING WORKFLOW COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Output directory: {self.output_dir}")
        
        return training_results
    
    def _extract_exif(self) -> Path:
        
        extractor = ExifExtractor()
        
        exif_csv = extractor.extract_from_folder(
            input_folder=self.config.paths.training_images,
            output_folder=self.config.paths.metadata_dir,
            recursive=True
        )
        
        if exif_csv is None:
            raise RuntimeError("EXIF extraction failed")
        
        self.logger.info(f"EXIF data saved: {exif_csv}")
        
        # Create a copy as "latest"
        latest_csv = self.config.paths.metadata_dir / "training_exif_latest.csv"
        import shutil
        shutil.copy(exif_csv, latest_csv)
        
        return exif_csv
    
    def _find_latest_exif_file(self) -> Path:
        
        latest_csv = self.config.paths.metadata_dir / "training_exif_latest.csv"
        
        if latest_csv.exists():
            return latest_csv
        
        # Find most recent file
        csv_files = list(self.config.paths.metadata_dir.glob("exif_data_complete_*.csv"))
        if not csv_files:
            raise FileNotFoundError("No EXIF data found. Run with extract_exif=True")
        
        csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return csv_files[0]
    
    def _engineer_features(self, exif_csv_path: Path) -> pd.DataFrame:
        
        # Load EXIF data
        df = pd.read_csv(exif_csv_path, encoding='utf-8-sig')
        self.logger.info(f"Loaded {len(df)} images")
        
        # Engineer features
        engineer = FeatureEngineer()
        df = engineer.engineer_features(df)
        
        self.logger.info(f"Features engineered: {len(engineer.feature_columns)} features")
        
        return df
    
    def _extract_category_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.copy()
        
        # Extract category from SourceFile path
        def get_category_from_path(source_file: str) -> str:
            path = Path(source_file)
            # Get parent folder name
            parent_folder = path.parent.name
            
            # Map to standard category names
            category_mapping = {
                'ChanDung': 'ChanDung',
                'chandung': 'ChanDung',
                'portrait': 'ChanDung',
                'Portrait': 'ChanDung',
                
                'TinhVat': 'TinhVat',
                'tinhvat': 'TinhVat',
                'stilllife': 'TinhVat',
                'StillLife': 'TinhVat',
                
                'TheThao': 'TheThao',
                'thethao': 'TheThao',
                'sports': 'TheThao',
                'Sports': 'TheThao',
                
                'PhongCanh': 'PhongCanh',
                'phongcanh': 'PhongCanh',
                'landscape': 'PhongCanh',
                'Landscape': 'PhongCanh',
                
                'DongVat': 'DongVat',
                'dongvat': 'DongVat',
                'wildlife': 'DongVat',
                'Wildlife': 'DongVat',
            }
            
            return category_mapping.get(parent_folder, 'Unknown')
        
        df['Category'] = df['SourceFile'].apply(get_category_from_path)
        
        # Remove unknown categories
        df = df[df['Category'] != 'Unknown']
        
        self.logger.info(f"Category distribution:")
        self.logger.info(df['Category'].value_counts())
        
        return df
    
    def _detect_faces(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # Initialize face detector
        detector = YOLOFaceDetector(
            model_path=str(self.config.paths.yolo_face_model),
            device=self.config.face_detection.device,
            confidence_threshold=self.config.face_detection.confidence_threshold,
            iou_threshold=self.config.face_detection.iou_threshold,
            min_face_size_ratio=self.config.face_detection.min_face_size_ratio,
            max_faces=self.config.face_detection.max_faces
        )
        
        # Detect faces for each image
        face_features_list = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Detecting faces"):
            image_path = row['SourceFile']
            
            try:
                face_features = detector.extract_face_features(image_path)
                face_features_list.append(face_features)
            except Exception as e:
                self.logger.warning(f"Error detecting faces in {image_path}: {e}")
                face_features_list.append({
                    'HasFace': 0,
                    'NumFaces': 0,
                    'FaceAreaRatio': 0.0
                })
        
        # Add to dataframe
        face_df = pd.DataFrame(face_features_list)
        df = pd.concat([df.reset_index(drop=True), face_df], axis=1)
        
        self.logger.info(f"Faces detected in {df['HasFace'].sum()} / {len(df)} images")
        
        return df
    
    def _get_feature_columns(self) -> list:
        
        # EXIF features
        feature_columns = self.config._config['features']['exif_features'].copy()
        
        # Add face features if enabled
        if self.config.face_detection.enabled:
            feature_columns.extend(self.config._config['features']['vision_features'])
        
        return feature_columns
    
    def _train_model(self, df: pd.DataFrame, feature_columns: list) -> dict:
        
        # Initialize trainer
        trainer = ModelTrainer(
            model_config=self.config._config['model'],
            feature_columns=feature_columns,
            use_scaler=self.config.model.use_scaler
        )
        
        # Train
        training_results = trainer.train(
            df=df,
            target_column='Category',
            test_size=self.config.model.test_size,
            random_state=self.config.model.random_state,
            stratify=self.config.model.stratify
        )
        
        # Store trainer for later use
        self.trainer = trainer
        
        return training_results
    
    def _save_model(self):
        
        self.trainer.save_model(
            model_path=str(self.config.paths.rf_model),
            scaler_path=str(self.config.paths.scaler) if self.config.model.use_scaler else None
        )
    
    def _generate_reports(self):
        
        # Save training report
        report_path = self.output_dir / "classification_report.txt"
        self.trainer.save_training_report(report_path)
        
        # Plot confusion matrix
        cm_path = self.output_dir / "confusion_matrix.png"
        self.trainer.plot_confusion_matrix(cm_path)
        
        # Plot feature importance
        fi_path = self.output_dir / "feature_importance.png"
        self.trainer.plot_feature_importance(fi_path)
        
        self.logger.info(f"Reports saved to: {self.output_dir}")
