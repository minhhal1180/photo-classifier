import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    
    def __init__(
        self,
        model_config: Dict,
        feature_columns: list,
        use_scaler: bool = False
    ):
        
        self.model_config = model_config
        self.feature_columns = feature_columns
        self.use_scaler = use_scaler
        
        # Initialize model
        rf_params = model_config.get('random_forest', {})
        self.model = RandomForestClassifier(
            n_estimators=rf_params.get('n_estimators', 500),
            max_depth=rf_params.get('max_depth', 15),
            min_samples_split=rf_params.get('min_samples_split', 2),
            min_samples_leaf=rf_params.get('min_samples_leaf', 1),
            class_weight=rf_params.get('class_weight', 'balanced'),
            random_state=rf_params.get('random_state', 42),
            n_jobs=-1,
            verbose=1
        )
        
        # Initialize scaler
        self.scaler = StandardScaler() if use_scaler else None
        
        # Store training info
        self.training_info = {}
    
    def train(
        self,
        df: pd.DataFrame,
        target_column: str = 'Category',
        test_size: float = 0.1,
        random_state: int = 42,
        stratify: bool = True
    ) -> Dict:
        
        print("=" * 60)
        print("TRAINING RANDOM FOREST CLASSIFIER")
        print("=" * 60)
        
        # Check for missing columns
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        # Prepare data
        X = df[self.feature_columns].copy()
        y = df[target_column].copy()
        
        print(f"\nDataset: {len(X)} samples")
        print(f"Features: {len(self.feature_columns)}")
        print(f"Classes: {y.nunique()}")
        print(f"\nClass distribution:")
        print(y.value_counts())
        
        # Handle missing values
        X = X.fillna(0)
        
        # Train-test split
        stratify_param = y if stratify else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )
        
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Scale features (if enabled)
        if self.scaler:
            print("\nScaling features...")
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        # Train model
        print("\nTraining Random Forest...")
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"\n{'='*60}")
        print("TRAINING RESULTS")
        print(f"{'='*60}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy:  {test_accuracy:.4f}")
        
        # Classification report
        print(f"\n{'='*60}")
        print("CLASSIFICATION REPORT (Test Set)")
        print(f"{'='*60}")
        report = classification_report(y_test, y_test_pred)
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n{'='*60}")
        print("TOP 10 FEATURE IMPORTANCES")
        print(f"{'='*60}")
        print(feature_importance.head(10).to_string(index=False))
        
        # Store training info
        self.training_info = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'feature_importance': feature_importance,
            'classes': self.model.classes_.tolist(),
            'n_samples_train': len(X_train),
            'n_samples_test': len(X_test),
            'n_features': len(self.feature_columns)
        }
        
        return self.training_info
    
    def save_model(self, model_path: str, scaler_path: Optional[str] = None):
        
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, model_path)
        print(f"Model saved: {model_path}")
        
        if self.scaler and scaler_path:
            scaler_path = Path(scaler_path)
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler saved: {scaler_path}")
    
    def plot_confusion_matrix(
        self,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ):
        
        if 'confusion_matrix' not in self.training_info:
            raise ValueError("No training info available. Train model first.")
        
        cm = self.training_info['confusion_matrix']
        classes = self.training_info['classes']
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=classes,
            yticklabels=classes
        )
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved: {save_path}")
        
        plt.close()
    
    def plot_feature_importance(
        self,
        save_path: Optional[str] = None,
        top_n: int = 15,
        figsize: Tuple[int, int] = (10, 8)
    ):
        
        if 'feature_importance' not in self.training_info:
            raise ValueError("No training info available. Train model first.")
        
        feature_importance = self.training_info['feature_importance'].head(top_n)
        
        plt.figure(figsize=figsize)
        plt.barh(
            range(len(feature_importance)),
            feature_importance['importance'],
            color='steelblue'
        )
        plt.yticks(
            range(len(feature_importance)),
            feature_importance['feature']
        )
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(f'Top {top_n} Feature Importances', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved: {save_path}")
        
        plt.close()
    
    def save_training_report(self, output_path: str):
        
        if not self.training_info:
            raise ValueError("No training info available. Train model first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("RANDOM FOREST TRAINING REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Train Samples: {self.training_info['n_samples_train']}\n")
            f.write(f"Test Samples: {self.training_info['n_samples_test']}\n")
            f.write(f"Number of Features: {self.training_info['n_features']}\n")
            f.write(f"Number of Classes: {len(self.training_info['classes'])}\n\n")
            
            f.write(f"Train Accuracy: {self.training_info['train_accuracy']:.4f}\n")
            f.write(f"Test Accuracy: {self.training_info['test_accuracy']:.4f}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("CLASSIFICATION REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(self.training_info['classification_report'])
            f.write("\n")
            
            f.write("=" * 60 + "\n")
            f.write("FEATURE IMPORTANCES\n")
            f.write("=" * 60 + "\n")
            f.write(self.training_info['feature_importance'].to_string(index=False))
            f.write("\n")
        
        print(f"Training report saved: {output_path}")
