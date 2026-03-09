"""
Simple Computer Vision Module - No External Dependencies
Uses only OpenCV for reliable face/eye detection
"""
import cv2
import numpy as np
from typing import Dict, Tuple
import math

class SimpleEyeDetector:
    """Simple but effective eye detection using only OpenCV"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        print("✅ Simple OpenCV detection initialized")
        
        self.EAR_THRESHOLD = 0.25
        self.CONSECUTIVE_FRAMES = 3
        self.frame_counter = 0
        self.blink_counter = 0
        self.total_blinks = 0
        self.prev_eye_state = "open"
        self.ear_history = []
        self.alert_level = 1.0
    
    def euclidean_distance(self, p1, p2):
        """Calculate euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def calculate_ear_from_rect(self, eye_rect: Tuple[int, int, int, int]) -> float:
        """Calculate EAR from eye rectangle"""
        ex, ey, ew, eh = eye_rect
        aspect_ratio = eh / max(ew, 1)
        normalized_ear = min(0.4, max(0.1, aspect_ratio * 0.7))
        return normalized_ear
    
    def detect_blink(self, current_ear: float, eye_count: int) -> bool:
        """Simple blink detection"""
        current_state = "closed" if current_ear < self.EAR_THRESHOLD or eye_count == 0 else "open"
        
        if self.prev_eye_state == "closed" and current_state == "open":
            self.total_blinks += 1
            self.prev_eye_state = current_state
            return True
        
        self.prev_eye_state = current_state
        return False
    
    def detect_drowsiness_from_frame(self, frame: np.ndarray) -> Dict:
        """Simple but effective drowsiness detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Face detection
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        
        result = {
            'face_detected': len(faces) > 0,
            'eyes_detected': 0,
            'ear_score': 1.0,
            'left_ear': 1.0,
            'right_ear': 1.0,
            'alertness_cv': 1.0,
            'drowsy_detected': False,
            'blink_detected': False,
            'total_blinks': self.total_blinks,
            'eye_closure_duration': 0,
            'smile_detected': False,
            'alert_level': self.alert_level
        }
        
        if len(faces) == 0:
            return result
        
        # Use largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        
        # Draw face rectangle
        color = (0, 255, 0) if not result['drowsy_detected'] else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Eye detection in upper face region
        roi_gray = gray[y:y+int(h*0.6), x:x+w]
        roi_color = frame[y:y+int(h*0.6), x:x+w]
        
        eyes = self.eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.05, minNeighbors=3, 
            minSize=(int(w*0.1), int(h*0.05)), maxSize=(int(w*0.4), int(h*0.2))
        )
        
        result['eyes_detected'] = len(eyes)
        
        # Smile detection for alertness
        smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        result['smile_detected'] = len(smiles) > 0
        
        if len(eyes) >= 2:
            # Sort eyes left to right
            eyes = sorted(eyes, key=lambda e: e[0])[:2]
            
            # Draw eye rectangles
            for i, (ex, ey, ew, eh) in enumerate(eyes):
                eye_color = (0, 255, 0) if i == 0 else (255, 0, 0)
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), eye_color, 2)
                center = (ex + ew//2, ey + eh//2)
                cv2.circle(roi_color, center, 3, eye_color, -1)
            
            # Calculate EAR
            left_ear = self.calculate_ear_from_rect(eyes[0])
            right_ear = self.calculate_ear_from_rect(eyes[1] if len(eyes) > 1 else eyes[0])
            avg_ear = (left_ear + right_ear) / 2.0
            
            result['left_ear'] = left_ear
            result['right_ear'] = right_ear
            result['ear_score'] = avg_ear
            
            # Blink detection
            result['blink_detected'] = self.detect_blink(avg_ear, len(eyes))
            result['total_blinks'] = self.total_blinks
            
            # Drowsiness detection
            if avg_ear < self.EAR_THRESHOLD:
                self.frame_counter += 1
                result['eye_closure_duration'] = self.frame_counter
                
                if self.frame_counter >= self.CONSECUTIVE_FRAMES:
                    result['drowsy_detected'] = True
                    self.alert_level = max(0.1, self.alert_level - 0.02)
            else:
                self.frame_counter = 0
                self.alert_level = min(1.0, self.alert_level + 0.01)
            
            result['alert_level'] = self.alert_level
            result['alertness_cv'] = self.alert_level
            
        elif len(eyes) == 1:
            # Single eye
            ex, ey, ew, eh = eyes[0]
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)
            result['ear_score'] = 0.2
            result['alertness_cv'] = 0.6
        else:
            # No eyes - likely drowsy
            result['ear_score'] = 0.1
            result['alertness_cv'] = 0.3
            self.frame_counter += 1
            if self.frame_counter >= self.CONSECUTIVE_FRAMES:
                result['drowsy_detected'] = True
        
        # Store EAR history
        if result['face_detected']:
            self.ear_history.append(result['ear_score'])
            if len(self.ear_history) > 30:
                self.ear_history.pop(0)
        
        return result