import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional
from tqdm import tqdm

from ..config import Config
from ..data_processing.exif_extractor import ExifExtractor
from ..data_processing.feature_engineering import FeatureEngineer
from ..face_detection.yolo_face_detector import YOLOFaceDetector
from ..face_detection.face_recognizer import FaceRecognizer
from ..models.predictor import CategoryPredictor
from ..classification.category_classifier import CategoryClassifier
from ..utils.logger import setup_logger

class InferenceWorkflow:
    
    def __init__(self, config_path: str = "config.yaml"):
        
        self.config = Config(config_path)
        self.logger = setup_logger("inference_workflow")
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.config.paths.inference_exports / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Inference output directory: {self.output_dir}")
    
    def run(
        self,
        extract_exif: bool = True,
        detect_faces: bool = True,
        recognize_faces: bool = True,
        organize_by_category: bool = True,
        organize_by_person: bool = False,
        copy_files: bool = True
    ) -> dict:
        
        self.logger.info("=" * 60)
        self.logger.info("STARTING INFERENCE WORKFLOW")
        self.logger.info("=" * 60)
        
        # Step 1: Extract EXIF
        if extract_exif:
            self.logger.info("\nStep 1: Extracting EXIF data...")
            exif_csv_path = self._extract_exif()
        else:
            exif_csv_path = self._find_latest_exif_file()
            self.logger.info(f"Using existing EXIF file: {exif_csv_path}")
        
        # Step 2: Load and engineer features
        self.logger.info("\nStep 2: Engineering features...")
        df = self._engineer_features(exif_csv_path)
        
        # Step 3: Face detection (optional)
        if detect_faces and self.config.face_detection.enabled:
            self.logger.info("\nStep 3: Detecting faces...")
            df = self._detect_faces(df)
        else:
            self.logger.info("\nStep 3: Skipping face detection")
            df['HasFace'] = 0
            df['NumFaces'] = 0
            df['FaceAreaRatio'] = 0.0
            df['face_boxes'] = None
        
        # Step 4: Face recognition (optional)
        if recognize_faces and detect_faces and self.config.face_recognition.enabled:
            self.logger.info("\nStep 4: Recognizing faces...")
            df = self._recognize_faces(df)
        else:
            self.logger.info("\nStep 4: Skipping face recognition")
            df['person_name'] = None
        
        # Step 5: Predict categories
        self.logger.info("\nStep 5: Predicting categories...")
        df = self._predict_categories(df)
        
        # Step 6: Apply business rules
        self.logger.info("\nStep 6: Applying business rules...")
        df = self._apply_business_rules(df)
        
        # Step 7: Save results
        self.logger.info("\nStep 7: Saving results...")
        results_csv = self.output_dir / "classification_results.csv"
        df.to_csv(results_csv, index=False, encoding='utf-8-sig')
        self.logger.info(f"Results saved: {results_csv}")
        
        # Step 8: Organize files
        if organize_by_category:
            self.logger.info("\nStep 8a: Organizing by category...")
            category_counts = self._organize_by_category(df, copy_files)
            self.logger.info(f"Category counts: {category_counts}")
        
        if organize_by_person and recognize_faces:
            self.logger.info("\nStep 8b: Organizing by person...")
            person_counts = self._organize_by_person(df, copy_files)
            self.logger.info(f"Person counts: {person_counts}")
        
        # Step 9: Generate summary report
        self.logger.info("\nStep 9: Generating summary report...")
        self._generate_summary_report(df)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("INFERENCE WORKFLOW COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Output directory: {self.output_dir}")
        
        return {
            'total_images': len(df),
            'output_dir': str(self.output_dir),
            'results_csv': str(results_csv)
        }
    
    def _extract_exif(self) -> Path:
        
        extractor = ExifExtractor()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = self.config.paths.metadata_dir
        
        exif_csv = extractor.extract_from_folder(
            input_folder=self.config.paths.inference_images,
            output_folder=output_folder,
            recursive=False
        )
        
        if exif_csv is None:
            raise RuntimeError("EXIF extraction failed")
        
        self.logger.info(f"EXIF data saved: {exif_csv}")
        return exif_csv
    
    def _find_latest_exif_file(self) -> Path:
        
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
        face_results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Detecting faces"):
            image_path = row['SourceFile']
            
            try:
                result = detector.detect_faces(image_path, return_crops=False)
                face_results.append({
                    'HasFace': 1 if result['has_face'] else 0,
                    'NumFaces': result['num_faces'],
                    'FaceAreaRatio': result['face_area_ratio'],
                    'face_boxes': result['boxes']
                })
            except Exception as e:
                self.logger.warning(f"Error detecting faces in {image_path}: {e}")
                face_results.append({
                    'HasFace': 0,
                    'NumFaces': 0,
                    'FaceAreaRatio': 0.0,
                    'face_boxes': []
                })
        
        # Add to dataframe
        face_df = pd.DataFrame(face_results)
        df = pd.concat([df.reset_index(drop=True), face_df], axis=1)
        
        self.logger.info(f"Faces detected in {df['HasFace'].sum()} / {len(df)} images")
        
        return df
    
    def _recognize_faces(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # Initialize face recognizer
        recognizer = FaceRecognizer(
            known_faces_dir=str(self.config.paths.known_faces),
            model_name=self.config.face_recognition.model_name,
            detector_backend=self.config.face_recognition.detector_backend,
            distance_metric=self.config.face_recognition.distance_metric,
            distance_threshold=self.config.face_recognition.distance_threshold
        )
        
        person_names = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Recognizing faces"):
            if row['HasFace'] == 0 or not row['face_boxes']:
                person_names.append(None)
                continue
            
            image_path = row['SourceFile']
            face_boxes = row['face_boxes']
            
            try:
                names = recognizer.recognize_faces_in_image(image_path, face_boxes)
                # Take first recognized person (for simplicity)
                recognized_name = next((name for name in names if name), None)
                person_names.append(recognized_name)
            except Exception as e:
                self.logger.warning(f"Error recognizing faces in {image_path}: {e}")
                person_names.append(None)
        
        df['person_name'] = person_names
        
        recognized_count = df['person_name'].notna().sum()
        self.logger.info(f"People recognized in {recognized_count} / {len(df)} images")
        
        return df
    
    def _predict_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        
        # Get feature columns
        feature_columns = self.config._config['features']['exif_features'].copy()
        
        if self.config.face_detection.enabled:
            feature_columns.extend(self.config._config['features']['vision_features'])
        
        # Initialize predictor
        predictor = CategoryPredictor(
            model_path=str(self.config.paths.rf_model),
            feature_columns=feature_columns,
            scaler_path=str(self.config.paths.scaler) if self.config.model.use_scaler else None
        )
        
        # Predict
        predictions_df = predictor.predict_with_confidence(df)
        df = pd.concat([df.reset_index(drop=True), predictions_df], axis=1)
        
        self.logger.info("Category predictions:")
        self.logger.info(df['predicted_category'].value_counts())
        
        return df
    
    def _apply_business_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        
        df = CategoryClassifier.apply_business_rules(
            df=df,
            category_column='predicted_category',
            confidence_column='confidence'
        )
        
        self.logger.info("Refined categories:")
        self.logger.info(df['refined_category'].value_counts())
        
        return df
    
    def _organize_by_category(self, df: pd.DataFrame, copy_files: bool) -> dict:
        
        category_dir = self.output_dir / "by_category"
        
        counts = CategoryClassifier.organize_by_category(
            df=df,
            source_dir=self.config.paths.inference_images,
            output_dir=category_dir,
            category_column='refined_category',
            copy_files=copy_files
        )
        
        return counts
    
    def _organize_by_person(self, df: pd.DataFrame, copy_files: bool) -> dict:
        
        person_dir = self.output_dir / "by_person"
        
        counts = CategoryClassifier.organize_by_person(
            df=df,
            source_dir=self.config.paths.inference_images,
            output_dir=person_dir,
            person_column='person_name',
            copy_files=copy_files
        )
        
        return counts
    
    def _generate_summary_report(self, df: pd.DataFrame):
        
        report_path = self.output_dir / "summary_report.txt"
        
        CategoryClassifier.generate_summary_report(
            df=df,
            output_path=report_path,
            category_column='refined_category'
        )
