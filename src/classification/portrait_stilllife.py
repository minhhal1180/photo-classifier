import pandas as pd
from typing import Dict

class PortraitStillLifeClassifier:
    
    @staticmethod
    def classify(
        has_face: bool,
        num_faces: int,
        face_area_ratio: float,
        focal_length_35mm: float,
        f_number: float,
        dof: float,
        ml_prediction: str = None,
        ml_confidence: float = 0.5
    ) -> str:
        
        # Strong indicators for Portrait
        portrait_score = 0
        
        # 1. Face detection (strongest indicator)
        if has_face:
            portrait_score += 40
            if num_faces == 1:
                portrait_score += 10  # Single person portrait
            if face_area_ratio > 0.15:
                portrait_score += 20  # Close-up portrait
            elif face_area_ratio > 0.08:
                portrait_score += 10  # Medium portrait
        
        # 2. Focal length (portrait typically 50-135mm)
        if 50 <= focal_length_35mm <= 135:
            portrait_score += 15
        elif focal_length_35mm > 135:
            portrait_score += 5  # Telephoto can be portrait
        
        # 3. Aperture (portrait typically wide: f/1.4-f/2.8)
        if f_number < 2.8:
            portrait_score += 10
        elif f_number < 4.0:
            portrait_score += 5
        
        # 4. Depth of field (portrait typically shallow)
        if dof > 0 and dof < 2.0:
            portrait_score += 10
        elif dof >= 2.0 and dof < 5.0:
            portrait_score += 5
        
        # 5. ML model prediction (if provided)
        if ml_prediction == 'ChanDung' and ml_confidence > 0.6:
            portrait_score += 15
        elif ml_prediction == 'TinhVat' and ml_confidence > 0.6:
            portrait_score -= 15
        
        # Decision threshold
        # Score > 50 → Portrait
        # Score <= 50 → Still Life
        
        if portrait_score > 50:
            return 'ChanDung'
        else:
            return 'TinhVat'
    
    @staticmethod
    def classify_batch(df: pd.DataFrame) -> pd.Series:
        
        required_columns = [
            'HasFace', 'NumFaces', 'FaceAreaRatio',
            'FocalLength_35mm', 'FNumber', 'DoF_Clean'
        ]
        
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Optional columns
        ml_pred = df.get('predicted_category', None)
        ml_conf = df.get('confidence', None)
        
        results = []
        
        for idx, row in df.iterrows():
            result = PortraitStillLifeClassifier.classify(
                has_face=bool(row['HasFace']),
                num_faces=int(row['NumFaces']),
                face_area_ratio=float(row['FaceAreaRatio']),
                focal_length_35mm=float(row['FocalLength_35mm']),
                f_number=float(row['FNumber']),
                dof=float(row['DoF_Clean']),
                ml_prediction=ml_pred.iloc[idx] if ml_pred is not None else None,
                ml_confidence=ml_conf.iloc[idx] if ml_conf is not None else 0.5
            )
            results.append(result)
        
        return pd.Series(results, index=df.index)
    
    @staticmethod
    def explain_classification(
        has_face: bool,
        num_faces: int,
        face_area_ratio: float,
        focal_length_35mm: float,
        f_number: float,
        dof: float
    ) -> Dict[str, any]:
        
        reasons = []
        score = 0
        
        # Face detection
        if has_face:
            score += 40
            reasons.append(f"Face detected ({num_faces} face(s))")
            if face_area_ratio > 0.15:
                score += 20
                reasons.append(f"Large face area ({face_area_ratio:.1%})")
            elif face_area_ratio > 0.08:
                score += 10
                reasons.append(f"Medium face area ({face_area_ratio:.1%})")
        else:
            reasons.append("No face detected")
        
        # Focal length
        if 50 <= focal_length_35mm <= 135:
            score += 15
            reasons.append(f"Portrait focal length ({focal_length_35mm:.0f}mm)")
        elif focal_length_35mm > 135:
            score += 5
            reasons.append(f"Telephoto lens ({focal_length_35mm:.0f}mm)")
        else:
            reasons.append(f"Wide angle lens ({focal_length_35mm:.0f}mm)")
        
        # Aperture
        if f_number < 2.8:
            score += 10
            reasons.append(f"Wide aperture (f/{f_number:.1f})")
        elif f_number < 4.0:
            score += 5
            reasons.append(f"Moderate aperture (f/{f_number:.1f})")
        else:
            reasons.append(f"Narrow aperture (f/{f_number:.1f})")
        
        # Depth of field
        if dof > 0 and dof < 2.0:
            score += 10
            reasons.append(f"Shallow depth of field ({dof:.1f}m)")
        elif dof >= 2.0 and dof < 5.0:
            score += 5
            reasons.append(f"Moderate depth of field ({dof:.1f}m)")
        elif dof > 0:
            reasons.append(f"Large depth of field ({dof:.1f}m)")
        
        classification = 'ChanDung' if score > 50 else 'TinhVat'
        confidence = min(abs(score - 50) / 50, 1.0)  # Normalize to 0-1
        
        return {
            'classification': classification,
            'score': score,
            'confidence': confidence,
            'reasons': reasons
        }
