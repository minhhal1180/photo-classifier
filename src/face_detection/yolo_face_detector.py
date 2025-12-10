import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from ultralytics import YOLO
import torch

class YOLOFaceDetector:
    
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        min_face_size_ratio: float = 0.02,
        max_faces: int = 20
    ):
        
        self.model_path = Path(model_path)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.min_face_size_ratio = min_face_size_ratio
        self.max_faces = max_faces
        
        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)
        
        print(f"YOLOv8 Face Detector loaded on {self.device}")
    
    def detect_faces(
        self, 
        image_path: str,
        return_crops: bool = False
    ) -> dict:
        
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")
        
        height, width = image.shape[:2]
        image_area = height * width
        
        # Run inference
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Extract detections
        boxes = []
        confidences = []
        crops = []
        
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                # Calculate face area
                face_area = (x2 - x1) * (y2 - y1)
                face_ratio = face_area / image_area
                
                # Filter by minimum size
                if face_ratio >= self.min_face_size_ratio:
                    boxes.append([int(x1), int(y1), int(x2), int(y2)])
                    confidences.append(float(conf))
                    
                    if return_crops:
                        crop = image[int(y1):int(y2), int(x1):int(x2)]
                        crops.append(crop)
        
        # Limit number of faces
        if len(boxes) > self.max_faces:
            # Sort by confidence and take top-k
            sorted_indices = np.argsort(confidences)[::-1][:self.max_faces]
            boxes = [boxes[i] for i in sorted_indices]
            confidences = [confidences[i] for i in sorted_indices]
            if return_crops:
                crops = [crops[i] for i in sorted_indices]
        
        # Calculate total face area ratio
        total_face_area = sum([(b[2]-b[0]) * (b[3]-b[1]) for b in boxes])
        face_area_ratio = total_face_area / image_area if image_area > 0 else 0.0
        
        result = {
            'has_face': len(boxes) > 0,
            'num_faces': len(boxes),
            'face_area_ratio': face_area_ratio,
            'boxes': boxes,
            'confidences': confidences
        }
        
        if return_crops:
            result['crops'] = crops
        
        return result
    
    def detect_batch(
        self,
        image_paths: List[str],
        batch_size: int = 16
    ) -> List[dict]:
        
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            for path in batch_paths:
                try:
                    result = self.detect_faces(path, return_crops=False)
                    results.append(result)
                except Exception as e:
                    print(f"Error processing {path}: {e}")
                    results.append({
                        'has_face': False,
                        'num_faces': 0,
                        'face_area_ratio': 0.0,
                        'boxes': [],
                        'confidences': []
                    })
        
        return results
    
    def extract_face_features(self, image_path: str) -> dict:
        
        result = self.detect_faces(image_path, return_crops=False)
        
        return {
            'HasFace': 1 if result['has_face'] else 0,
            'NumFaces': result['num_faces'],
            'FaceAreaRatio': result['face_area_ratio']
        }
