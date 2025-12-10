import pandas as pd
from pathlib import Path
from typing import Dict, List
from datetime import datetime

class CategoryClassifier:
    
    CATEGORIES = {
        'ChanDung': 'Portrait',
        'TinhVat': 'Still Life',
        'TheThao': 'Sports',
        'PhongCanh': 'Landscape',
        'DongVat': 'Wildlife'
    }
    
    TIME_PERIODS = {
        'Sang': (5, 12),   # 5:00 - 12:00
        'Chieu': (12, 18),  # 12:00 - 18:00
        'Toi': (18, 24)     # 18:00 - 24:00 (and 0:00 - 5:00)
    }
    
    def __init__(self):
        
        pass
    
    @staticmethod
    def get_time_period(hour: int) -> str:
        
        if hour < 0 or hour > 23:
            return 'Unknown'
        
        if 5 <= hour < 12:
            return 'Sang'
        elif 12 <= hour < 18:
            return 'Chieu'
        else:
            return 'Toi'
    
    @staticmethod
    def refine_category_with_faces(
        predicted_category: str,
        has_face: bool,
        num_faces: int,
        face_area_ratio: float,
        confidence: float
    ) -> str:
        
        # High confidence predictions - trust the model
        if confidence > 0.8:
            return predicted_category
        
        # Face-based refinement for Portrait/StillLife confusion
        if predicted_category in ['ChanDung', 'TinhVat']:
            # Strong face presence → Portrait
            if has_face and face_area_ratio > 0.15:
                return 'ChanDung'
            
            # No faces but predicted Portrait with low confidence → Still Life
            if predicted_category == 'ChanDung' and not has_face and confidence < 0.6:
                return 'TinhVat'
            
            # No faces and predicted Still Life → confirm
            if predicted_category == 'TinhVat' and not has_face:
                return 'TinhVat'
        
        # For other categories, trust the model
        return predicted_category
    
    @staticmethod
    def apply_business_rules(
        df: pd.DataFrame,
        category_column: str = 'predicted_category',
        confidence_column: str = 'confidence'
    ) -> pd.DataFrame:
        
        df = df.copy()
        
        # Add time period if Hour exists
        if 'Hour' in df.columns:
            df['TimePeriod'] = df['Hour'].apply(CategoryClassifier.get_time_period)
        
        # Refine categories using face features
        if all(col in df.columns for col in ['HasFace', 'NumFaces', 'FaceAreaRatio']):
            df['refined_category'] = df.apply(
                lambda row: CategoryClassifier.refine_category_with_faces(
                    predicted_category=row[category_column],
                    has_face=bool(row['HasFace']),
                    num_faces=int(row['NumFaces']),
                    face_area_ratio=float(row['FaceAreaRatio']),
                    confidence=float(row[confidence_column])
                ),
                axis=1
            )
        else:
            df['refined_category'] = df[category_column]
        
        return df
    
    @staticmethod
    def organize_by_category(
        df: pd.DataFrame,
        source_dir: Path,
        output_dir: Path,
        category_column: str = 'refined_category',
        copy_files: bool = True
    ) -> Dict[str, int]:
        
        import shutil
        
        source_dir = Path(source_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        category_counts = {}
        
        for category in CategoryClassifier.CATEGORIES.keys():
            category_dir = output_dir / category
            category_dir.mkdir(exist_ok=True)
            
            # Filter images for this category
            category_df = df[df[category_column] == category]
            category_counts[category] = len(category_df)
            
            # Copy/move files
            for _, row in category_df.iterrows():
                filename = row.get('FileName', '')
                if not filename:
                    continue
                
                source_file = source_dir / filename
                dest_file = category_dir / filename
                
                if source_file.exists():
                    try:
                        if copy_files:
                            shutil.copy2(source_file, dest_file)
                        else:
                            shutil.move(str(source_file), str(dest_file))
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
        
        return category_counts
    
    @staticmethod
    def organize_by_person(
        df: pd.DataFrame,
        source_dir: Path,
        output_dir: Path,
        person_column: str = 'person_name',
        copy_files: bool = True
    ) -> Dict[str, int]:
        
        import shutil
        
        source_dir = Path(source_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter rows with recognized people
        recognized_df = df[df[person_column].notna() & (df[person_column] != '')]
        
        person_counts = {}
        
        for person_name in recognized_df[person_column].unique():
            person_dir = output_dir / person_name
            person_dir.mkdir(exist_ok=True)
            
            # Filter images for this person
            person_df = recognized_df[recognized_df[person_column] == person_name]
            person_counts[person_name] = len(person_df)
            
            # Copy/move files
            for _, row in person_df.iterrows():
                filename = row.get('FileName', '')
                if not filename:
                    continue
                
                source_file = source_dir / filename
                dest_file = person_dir / filename
                
                if source_file.exists():
                    try:
                        if copy_files:
                            shutil.copy2(source_file, dest_file)
                        else:
                            shutil.move(str(source_file), str(dest_file))
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
        
        return person_counts
    
    @staticmethod
    def generate_summary_report(
        df: pd.DataFrame,
        output_path: Path,
        category_column: str = 'refined_category'
    ):
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("PHOTO CLASSIFICATION SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Images: {len(df)}\n\n")
            
            # Category distribution
            f.write("=" * 60 + "\n")
            f.write("CATEGORY DISTRIBUTION\n")
            f.write("=" * 60 + "\n")
            category_counts = df[category_column].value_counts()
            for category, count in category_counts.items():
                percentage = (count / len(df)) * 100
                f.write(f"{category:15s}: {count:5d} ({percentage:5.1f}%)\n")
            f.write("\n")
            
            # Time period distribution (if available)
            if 'TimePeriod' in df.columns:
                f.write("=" * 60 + "\n")
                f.write("TIME PERIOD DISTRIBUTION\n")
                f.write("=" * 60 + "\n")
                time_counts = df['TimePeriod'].value_counts()
                for period, count in time_counts.items():
                    percentage = (count / len(df)) * 100
                    f.write(f"{period:15s}: {count:5d} ({percentage:5.1f}%)\n")
                f.write("\n")
            
            # Face detection stats (if available)
            if 'HasFace' in df.columns:
                f.write("=" * 60 + "\n")
                f.write("FACE DETECTION STATISTICS\n")
                f.write("=" * 60 + "\n")
                faces_detected = df['HasFace'].sum()
                percentage = (faces_detected / len(df)) * 100
                f.write(f"Images with Faces: {faces_detected} ({percentage:.1f}%)\n")
                f.write(f"Average Faces per Image: {df['NumFaces'].mean():.2f}\n")
                f.write("\n")
            
            # Person recognition stats (if available)
            if 'person_name' in df.columns:
                recognized = df['person_name'].notna().sum()
                if recognized > 0:
                    f.write("=" * 60 + "\n")
                    f.write("PERSON RECOGNITION STATISTICS\n")
                    f.write("=" * 60 + "\n")
                    percentage = (recognized / len(df)) * 100
                    f.write(f"Images with Recognized People: {recognized} ({percentage:.1f}%)\n")
                    
                    person_counts = df['person_name'].value_counts()
                    f.write("\nTop People:\n")
                    for person, count in person_counts.head(10).items():
                        f.write(f"  {person}: {count}\n")
                    f.write("\n")
        
        print(f"Summary report saved: {output_path}")
