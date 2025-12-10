import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict

class CategoryPredictor:
    
    def __init__(
        self,
        model_path: str,
        feature_columns: List[str],
        scaler_path: Optional[str] = None
    ):
        
        self.model_path = Path(model_path)
        self.feature_columns = feature_columns
        self.scaler_path = Path(scaler_path) if scaler_path else None
        
        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = joblib.load(self.model_path)
        print(f"Model loaded: {self.model_path}")
        
        # Load scaler (if applicable)
        self.scaler = None
        if self.scaler_path and self.scaler_path.exists():
            self.scaler = joblib.load(self.scaler_path)
            print(f"Scaler loaded: {self.scaler_path}")
        
        # Get classes
        self.classes = self.model.classes_
        print(f"Categories: {', '.join(self.classes)}")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        
        # Check for missing columns
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        # Prepare features
        X = df[self.feature_columns].copy()
        X = X.fillna(0)
        
        # Scale (if scaler exists)
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X)
        
        return predictions
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        
        # Check for missing columns
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        # Prepare features
        X = df[self.feature_columns].copy()
        X = X.fillna(0)
        
        # Scale (if scaler exists)
        if self.scaler:
            X = self.scaler.transform(X)
        
        # Predict probabilities
        probabilities = self.model.predict_proba(X)
        
        return probabilities
    
    def predict_with_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        
        predictions = self.predict(df)
        probabilities = self.predict_proba(df)
        
        # Get max probability for each prediction
        confidences = np.max(probabilities, axis=1)
        
        result_df = pd.DataFrame({
            'predicted_category': predictions,
            'confidence': confidences
        })
        
        return result_df
    
    def predict_single(self, features: Dict) -> Dict:
        
        # Create DataFrame with single row
        df = pd.DataFrame([features])
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Predict
        prediction = self.predict(df)[0]
        probabilities = self.predict_proba(df)[0]
        confidence = np.max(probabilities)
        
        # Create probability dict
        prob_dict = {
            class_name: float(prob)
            for class_name, prob in zip(self.classes, probabilities)
        }
        
        return {
            'category': prediction,
            'confidence': float(confidence),
            'probabilities': prob_dict
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
