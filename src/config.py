import yaml
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass

@dataclass
class PathsConfig:
    training_images: Path
    training_exports: Path
    inference_images: Path
    inference_exports: Path
    known_faces: Path
    models_dir: Path
    rf_model: Path
    yolo_face_model: Path
    scaler: Path
    metadata_dir: Path

@dataclass
class ModelConfig:
    type: str
    random_forest: Dict[str, Any]
    test_size: float
    random_state: int
    stratify: bool
    use_scaler: bool

@dataclass
class FaceDetectionConfig:
    enabled: bool
    model_name: str
    device: str
    confidence_threshold: float
    iou_threshold: float
    min_face_size_ratio: float
    max_faces: int
    batch_size: int
    half_precision: bool

@dataclass
class FaceRecognitionConfig:
    enabled: bool
    model_name: str
    detector_backend: str
    distance_metric: str
    distance_threshold: float
    enforce_detection: bool
    align: bool
    normalization: str

class Config:
    def __init__(self, config_path: str = "config.yaml"):
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Không tìm thấy file cấu hình: {config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)
        
        self._setup_paths()
    
    def _setup_paths(self):
        paths_dict = self._config['paths']
        self.paths = PathsConfig(
            training_images=Path(paths_dict['training_images']),
            training_exports=Path(paths_dict['training_exports']),
            inference_images=Path(paths_dict['inference_images']),
            inference_exports=Path(paths_dict['inference_exports']),
            known_faces=Path(paths_dict['known_faces']),
            models_dir=Path(paths_dict['models_dir']),
            rf_model=Path(paths_dict['rf_model']),
            yolo_face_model=Path(paths_dict['yolo_face_model']),
            scaler=Path(paths_dict['scaler']),
            metadata_dir=Path(paths_dict['metadata_dir'])
        )
    
    @property
    def model(self) -> ModelConfig:
        m = self._config['model']
        return ModelConfig(
            type=m['type'],
            random_forest=m['random_forest'],
            test_size=m['test_size'],
            random_state=m['random_state'],
            stratify=m['stratify'],
            use_scaler=m['use_scaler']
        )
    
    @property
    def face_detection(self) -> FaceDetectionConfig:
        fd = self._config['face_detection']
        return FaceDetectionConfig(
            enabled=fd['enabled'],
            model_name=fd['model_name'],
            device=fd['device'],
            confidence_threshold=fd['confidence_threshold'],
            iou_threshold=fd['iou_threshold'],
            min_face_size_ratio=fd['min_face_size_ratio'],
            max_faces=fd['max_faces'],
            batch_size=fd['batch_size'],
            half_precision=fd['half_precision']
        )
    
    @property
    def face_recognition(self) -> FaceRecognitionConfig:
        fr = self._config['face_recognition']
        return FaceRecognitionConfig(
            enabled=fr['enabled'],
            model_name=fr['model_name'],
            detector_backend=fr['detector_backend'],
            distance_metric=fr['distance_metric'],
            distance_threshold=fr['distance_threshold'],
            enforce_detection=fr['enforce_detection'],
            align=fr['align'],
            normalization=fr['normalization']
        )
    
    @property
    def features(self) -> Dict[str, list]:
        return self._config['features']
    
    @property
    def categories(self) -> list:
        return self._config['categories']
    
    @property
    def time_buckets(self) -> Dict[str, Any]:
        return self._config['time_buckets']
    
    @property
    def classification_rules(self) -> Dict[str, Any]:
        return self._config['classification_rules']
    
    @property
    def output(self) -> Dict[str, Any]:
        return self._config['output']
    
    @property
    def performance(self) -> Dict[str, Any]:
        return self._config['performance']
    
    @property
    def logging(self) -> Dict[str, Any]:
        return self._config['logging']
    
    @property
    def advanced(self) -> Dict[str, Any]:
        return self._config['advanced']
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

# Global config instance
_config = None

def get_config(config_path: str = "config.yaml") -> Config:
    
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config

def reload_config(config_path: str = "config.yaml") -> Config:
    
    global _config
    _config = Config(config_path)
    return _config
