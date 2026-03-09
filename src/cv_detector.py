"""
Real Computer Vision Module for Driver Drowsiness Detection
Uses OpenCV for face/eye detection and Eye Aspect Ratio (EAR) calculation
"""
import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from scipy.spatial import distance

class EyeDetector:
    """Real-time eye detection and drowsiness analysis using OpenCV + dlib"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        print("✅ Enhanced OpenCV detection enabled")
        
        self.EAR_THRESHOLD = 0.25
        self.CONSECUTIVE_FRAMES = 3
        self.frame_counter = 0
        self.ear_history = []
        self.blink_counter = 0
        self.total_blinks = 0
        self.prev_eye_state = "open"
        self.drowsy_start_time = None
        self.alert_level = 1.0
        
    def calculate_ear_from_rect(self, eye_rect: Tuple[int, int, int, int]) -> float:
        """Enhanced EAR calculation from eye rectangle"""
        ex, ey, ew, eh = eye_rect
        aspect_ratio = eh / max(ew, 1)
        # Improved normalization
        normalized_ear = min(0.4, max(0.1, aspect_ratio * 0.7))
        return normalized_ear
    
    def detect_blink_opencv(self, current_ear: float, eye_count: int) -> bool:
        """Enhanced blink detection using OpenCV only"""
        current_state = "closed" if current_ear < self.EAR_THRESHOLD or eye_count == 0 else "open"
        
        # Detect blink transition
        if self.prev_eye_state == "closed" and current_state == "open":
            self.total_blinks += 1
            self.prev_eye_state = current_state
            return True
        
        self.prev_eye_state = current_state
        return False
    
    def analyze_face_features(self, face_rect, gray_frame):
        """Analyze additional face features for drowsiness"""
        x, y, w, h = face_rect
        roi_gray = gray_frame[y:y+h, x:x+w]
        
        # Check for smile (alertness indicator)
        smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        smile_detected = len(smiles) > 0
        
        # Head tilt detection (basic)
        face_center_x = x + w // 2
        frame_center_x = gray_frame.shape[1] // 2
        head_tilt = abs(face_center_x - frame_center_x) > w * 0.3
        
        return {
            'smile_detected': smile_detected,
            'head_tilted': head_tilt,
            'face_size': w * h  # Larger = closer to camera
        }
    
    def detect_drowsiness_from_frame(self, frame: np.ndarray) -> Dict:
        """Enhanced drowsiness detection using only OpenCV"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Improve face detection
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
            'head_tilted': False,
            'alert_level': self.alert_level,
            'drowsy_duration': 0
        }
        
        if len(faces) == 0:
            return result
        
        # Use largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        
        # Draw enhanced face rectangle
        color = (0, 255, 0) if not result['drowsy_detected'] else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Analyze face features
        face_features = self.analyze_face_features(face, gray)
        result.update(face_features)
        
        # Enhanced eye detection
        roi_gray = gray[y:y+int(h*0.6), x:x+w]
        roi_color = frame[y:y+int(h*0.6), x:x+w]
        
        eyes = self.eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.05, minNeighbors=3, 
            minSize=(int(w*0.1), int(h*0.05)), maxSize=(int(w*0.4), int(h*0.2))
        )
        
        result['eyes_detected'] = len(eyes)
        
        if len(eyes) >= 2:
            # Sort and process eyes
            eyes = sorted(eyes, key=lambda e: e[0])[:2]
            
            # Draw eye rectangles with different colors
            for i, (ex, ey, ew, eh) in enumerate(eyes):
                eye_color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Green for left, blue for right
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), eye_color, 2)
                # Add eye center point
                center = (ex + ew//2, ey + eh//2)
                cv2.circle(roi_color, center, 3, eye_color, -1)
            
            # Calculate EAR for both eyes
            left_ear = self.calculate_ear_from_rect(eyes[0])
            right_ear = self.calculate_ear_from_rect(eyes[1] if len(eyes) > 1 else eyes[0])
            avg_ear = (left_ear + right_ear) / 2.0
            
            result['left_ear'] = left_ear
            result['right_ear'] = right_ear
            result['ear_score'] = avg_ear
            
            # Enhanced blink detection
            result['blink_detected'] = self.detect_blink_opencv(avg_ear, len(eyes))
            result['total_blinks'] = self.total_blinks
            
            # Drowsiness analysis with timing
            if avg_ear < self.EAR_THRESHOLD:
                self.frame_counter += 1
                if self.drowsy_start_time is None:
                    self.drowsy_start_time = cv2.getTickCount()
                
                result['eye_closure_duration'] = self.frame_counter
                
                if self.frame_counter >= self.CONSECUTIVE_FRAMES:
                    result['drowsy_detected'] = True
                    # Calculate drowsy duration in seconds
                    current_time = cv2.getTickCount()
                    result['drowsy_duration'] = (current_time - self.drowsy_start_time) / cv2.getTickFrequency()
                    
                    # Decrease alert level
                    self.alert_level = max(0.1, self.alert_level - 0.02)
            else:
                self.frame_counter = 0
                self.drowsy_start_time = None
                # Increase alert level
                self.alert_level = min(1.0, self.alert_level + 0.01)
            
            result['alert_level'] = self.alert_level
            result['alertness_cv'] = self.alert_level
            
        elif len(eyes) == 1:
            # Single eye detected
            ex, ey, ew, eh = eyes[0]
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 255), 2)
            result['ear_score'] = 0.2
            result['alertness_cv'] = 0.6
            
        else:
            # No eyes detected - likely very drowsy
            result['ear_score'] = 0.1
            result['alertness_cv'] = 0.3
            self.frame_counter += 1
            if self.frame_counter >= self.CONSECUTIVE_FRAMES:
                result['drowsy_detected'] = True
        
        # Store EAR history for trend analysis
        if result['face_detected']:
            self.ear_history.append(result['ear_score'])
            if len(self.ear_history) > 30:  # Keep 1 second of history at 30fps
                self.ear_history.pop(0)
            
            # Calculate EAR trend
            if len(self.ear_history) >= 10:
                recent_avg = np.mean(self.ear_history[-10:])
                older_avg = np.mean(self.ear_history[-20:-10]) if len(self.ear_history) >= 20 else recent_avg
                result['ear_trend'] = 'decreasing' if recent_avg < older_avg - 0.02 else 'stable'
        
        return result
    
    def _calculate_eye_ear(self, eye_rect: Tuple[int, int, int, int]) -> float:
        """Calculate EAR from eye bounding rectangle"""
        ex, ey, ew, eh = eye_rect
        
        # Improved EAR calculation based on aspect ratio
        aspect_ratio = eh / max(ew, 1)
        
        # Normalize to typical EAR range (0.2 - 0.4)
        normalized_ear = min(0.4, max(0.1, aspect_ratio * 0.8))
        return normalized_ear
    
    def process_video_stream(self, video_source: int = 0, duration: int = 5) -> Dict:
        """Process video stream and return aggregated drowsiness metrics"""
        cap = cv2.VideoCapture(video_source)
        frames_analyzed = 0
        drowsy_frames = 0
        ear_scores = []
        
        while frames_analyzed < duration * 30:  # 30 fps assumption
            ret, frame = cap.read()
            if not ret:
                break
                
            result = self.detect_drowsiness_from_frame(frame)
            if result['face_detected']:
                ear_scores.append(result['ear_score'])
                if result['drowsy_detected']:
                    drowsy_frames += 1
            frames_analyzed += 1
            
        cap.release()
        
        return {
            'frames_analyzed': frames_analyzed,
            'drowsy_frames': drowsy_frames,
            'avg_ear': np.mean(ear_scores) if ear_scores else 1.0,
            'alertness_cv': 1.0 - (drowsy_frames / max(frames_analyzed, 1)),
            'drowsiness_probability': drowsy_frames / max(frames_analyzed, 1)
        }
    
    def analyze_image(self, image_path: str) -> Dict:
        """Analyze single image for drowsiness"""
        frame = cv2.imread(image_path)
        if frame is None:
            return {'error': 'Could not read image'}
        return self.detect_drowsiness_from_frame(frame)
