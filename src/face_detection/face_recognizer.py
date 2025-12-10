import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from deepface import DeepFace
import pandas as pd

class FaceRecognizer:
  
    def __init__(
        self,
        known_faces_dir: str,
        model_name: str = 'Facenet512',
        detector_backend: str = 'skip',
        distance_metric: str = 'cosine',
        distance_threshold: float = 0.4,
        enforce_detection: bool = False
    ):
       
        self.known_faces_dir = Path(known_faces_dir)
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric
        self.distance_threshold = distance_threshold
        self.enforce_detection = enforce_detection
        
        # Build face database
        self.face_database = self._build_face_database()
        
        print(f"Face Recognizer initialized with {len(self.face_database)} known faces")
    
    def _build_face_database(self) -> List[Dict]:
        
        database = []
        
        if not self.known_faces_dir.exists():
            print(f"Warning: Known faces directory not found: {self.known_faces_dir}")
            return database
        
        # Iterate through person folders
        for person_dir in self.known_faces_dir.iterdir():
            if not person_dir.is_dir():
                continue
            
            person_name = person_dir.name
            
            # Find all image files in person's folder
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
            for image_file in person_dir.iterdir():
                if image_file.suffix.lower() in image_extensions:
                    database.append({
                        'person_name': person_name,
                        'image_path': str(image_file)
                    })
        
        return database
    
    def recognize_face(
        self,
        face_image: np.ndarray,
        return_all_matches: bool = False
    ) -> Optional[Dict]:
        
        if len(self.face_database) == 0:
            return None
        
        try:
            # Get embedding for query face
            query_embedding = DeepFace.represent(
                img_path=face_image,
                model_name=self.model_name,
                detector_backend='skip',  # Already cropped
                enforce_detection=False
            )[0]['embedding']
            
            # Compare with all known faces
            matches = []
            
            for known_face in self.face_database:
                try:
                    known_embedding = DeepFace.represent(
                        img_path=known_face['image_path'],
                        model_name=self.model_name,
                        detector_backend=self.detector_backend,
                        enforce_detection=self.enforce_detection
                    )[0]['embedding']
                    
                    # Calculate distance
                    if self.distance_metric == 'cosine':
                        distance = self._cosine_distance(query_embedding, known_embedding)
                    elif self.distance_metric == 'euclidean':
                        distance = self._euclidean_distance(query_embedding, known_embedding)
                    else:
                        distance = self._euclidean_l2_distance(query_embedding, known_embedding)
                    
                    if distance <= self.distance_threshold:
                        matches.append({
                            'person_name': known_face['person_name'],
                            'distance': distance,
                            'image_path': known_face['image_path']
                        })
                
                except Exception as e:
                    print(f"Error processing {known_face['image_path']}: {e}")
                    continue
            
            if len(matches) == 0:
                return None
            
            # Sort by distance (lower is better)
            matches.sort(key=lambda x: x['distance'])
            
            # Get best match
            best_match = matches[0]
            
            result = {
                'person_name': best_match['person_name'],
                'distance': best_match['distance']
            }
            
            if return_all_matches:
                result['all_matches'] = matches
            
            return result
        
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return None
    
    def recognize_faces_in_image(
        self,
        image_path: str,
        face_boxes: List[List[int]]
    ) -> List[Optional[str]]:
        
        image = cv2.imread(str(image_path))
        if image is None:
            return [None] * len(face_boxes)
        
        results = []
        
        for box in face_boxes:
            x1, y1, x2, y2 = box
            face_crop = image[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                results.append(None)
                continue
            
            match = self.recognize_face(face_crop)
            
            if match:
                results.append(match['person_name'])
            else:
                results.append(None)
        
        return results
    
    @staticmethod
    def _cosine_distance(embedding1: List[float], embedding2: List[float]) -> float:
        
        a = np.array(embedding1)
        b = np.array(embedding2)
        return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    @staticmethod
    def _euclidean_distance(embedding1: List[float], embedding2: List[float]) -> float:
        
        a = np.array(embedding1)
        b = np.array(embedding2)
        return np.linalg.norm(a - b)
    
    @staticmethod
    def _euclidean_l2_distance(embedding1: List[float], embedding2: List[float]) -> float:
        
        a = np.array(embedding1)
        b = np.array(embedding2)
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)
        return np.linalg.norm(a_norm - b_norm)
    
    def add_person_to_database(
        self,
        person_name: str,
        image_paths: List[str]
    ):
        
        person_dir = self.known_faces_dir / person_name
        person_dir.mkdir(parents=True, exist_ok=True)
        
        for image_path in image_paths:
            # Copy image to person directory
            import shutil
            image_file = Path(image_path)
            dest_path = person_dir / image_file.name
            shutil.copy(image_path, dest_path)
            
            # Add to database
            self.face_database.append({
                'person_name': person_name,
                'image_path': str(dest_path)
            })
        
        print(f"Added {len(image_paths)} images for {person_name}")
