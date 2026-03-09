"""
Professional Computer Vision Module using MediaPipe + OpenCV
Production-grade face detection and drowsiness analysis
"""
import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Tuple, Optional, List
from scipy.spatial import distance
import time

class ProfessionalEyeDetector:
    """Production-grade eye detection using MediaPipe + OpenCV"""
    
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Eye landmark indices for MediaPipe 468-point model
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # EAR calculation points (6 key points per eye)
        self.LEFT_EYE_EAR = [33, 160, 158, 133, 153, 144]  # Outer, top, bottom corners
        self.RIGHT_EYE_EAR = [362, 385, 387, 263, 373, 380]
        
        # Thresholds and counters
        self.EAR_THRESHOLD = 0.25
        self.CONSECUTIVE_FRAMES = 3
        self.frame_counter = 0
        self.blink_counter = 0
        self.total_blinks = 0
        self.ear_history = []
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        print("✅ Professional MediaPipe + OpenCV detection initialized")
    
    def calculate_ear_mediapipe(self, landmarks: List, eye_indices: List) -> float:
        """Calculate EAR using MediaPipe landmarks"""
        try:
            # Get 6 key points for EAR calculation
            points = []
            for idx in eye_indices:
                x = landmarks[idx].x
                y = landmarks[idx].y
                points.append([x, y])
            
            points = np.array(points)
            
            # Calculate EAR: (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
            A = distance.euclidean(points[1], points[5])  # Vertical 1
            B = distance.euclidean(points[2], points[4])  # Vertical 2  
            C = distance.euclidean(points[0], points[3])  # Horizontal
            
            ear = (A + B) / (2.0 * C) if C > 0 else 0
            return ear
        except:
            return 0.3  # Default EAR
    
    def detect_blink(self, ear: float) -> bool:
        """Advanced blink detection"""
        if ear < self.EAR_THRESHOLD:
            self.blink_counter += 1
        else:
            if self.blink_counter >= 2:
                self.total_blinks += 1
                self.blink_counter = 0
                return True
            self.blink_counter = 0
        return False
    
    def draw_eye_landmarks(self, frame: np.ndarray, landmarks, eye_indices: List, color: Tuple[int, int, int]):
        """Draw eye contours on frame"""
        h, w = frame.shape[:2]
        eye_points = []
        
        for idx in eye_indices:
            x = int(landmarks[idx].x * w)
            y = int(landmarks[idx].y * h)
            eye_points.append([x, y])
        
        eye_points = np.array(eye_points, dtype=np.int32)
        cv2.polylines(frame, [eye_points], True, color, 1)
        cv2.fillPoly(frame, [eye_points], (*color, 30))  # Semi-transparent fill
    
    def analyze_head_pose(self, landmarks) -> Dict:
        """Estimate head pose from facial landmarks"""
        # Key facial points for pose estimation
        nose_tip = landmarks[1]
        chin = landmarks[175]
        left_eye_corner = landmarks[33]
        right_eye_corner = landmarks[263]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        
        # Calculate head rotation (basic estimation)
        eye_center_x = (left_eye_corner.x + right_eye_corner.x) / 2
        mouth_center_x = (left_mouth.x + right_mouth.x) / 2
        
        # Head tilt detection
        head_tilt = abs(eye_center_x - mouth_center_x) > 0.02
        
        # Head turn detection  
        face_center_x = (left_eye_corner.x + right_eye_corner.x) / 2
        head_turn = abs(nose_tip.x - face_center_x) > 0.03
        
        return {
            'head_tilted': head_tilt,
            'head_turned': head_turn,
            'nose_x': nose_tip.x,
            'face_center_x': face_center_x
        }
    
    def detect_drowsiness_professional(self, frame: np.ndarray) -> Dict:
        """Professional-grade drowsiness detection"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        # FPS calculation
        self.fps_counter += 1
        current_time = time.time()
        fps = self.fps_counter / (current_time - self.fps_start_time) if current_time > self.fps_start_time else 0
        
        detection_result = {
            'face_detected': False,
            'landmarks_count': 0,
            'left_ear': 1.0,
            'right_ear': 1.0,
            'avg_ear': 1.0,
            'blink_detected': False,
            'total_blinks': self.total_blinks,
            'drowsy_detected': False,
            'eye_closure_duration': 0,
            'head_pose': {},
            'fps': fps,
            'detection_confidence': 0.0,
            'alertness_score': 1.0
        }
        
        if results.multi_face_landmarks:
            detection_result['face_detected'] = True
            face_landmarks = results.multi_face_landmarks[0]
            detection_result['landmarks_count'] = len(face_landmarks.landmark)
            
            # Calculate EAR for both eyes
            left_ear = self.calculate_ear_mediapipe(face_landmarks.landmark, self.LEFT_EYE_EAR)
            right_ear = self.calculate_ear_mediapipe(face_landmarks.landmark, self.RIGHT_EYE_EAR)
            avg_ear = (left_ear + right_ear) / 2.0
            
            detection_result['left_ear'] = left_ear
            detection_result['right_ear'] = right_ear
            detection_result['avg_ear'] = avg_ear
            
            # Draw eye landmarks
            self.draw_eye_landmarks(frame, face_landmarks.landmark, self.LEFT_EYE_INDICES, (0, 255, 0))
            self.draw_eye_landmarks(frame, face_landmarks.landmark, self.RIGHT_EYE_INDICES, (255, 0, 0))
            
            # Blink detection
            detection_result['blink_detected'] = self.detect_blink(avg_ear)
            detection_result['total_blinks'] = self.total_blinks
            
            # Head pose analysis
            detection_result['head_pose'] = self.analyze_head_pose(face_landmarks.landmark)
            
            # Drowsiness detection
            if avg_ear < self.EAR_THRESHOLD:
                self.frame_counter += 1
                detection_result['eye_closure_duration'] = self.frame_counter
                
                if self.frame_counter >= self.CONSECUTIVE_FRAMES:
                    detection_result['drowsy_detected'] = True
                    detection_result['alertness_score'] = max(0.1, avg_ear / self.EAR_THRESHOLD)
            else:
                self.frame_counter = 0
                detection_result['alertness_score'] = min(1.0, avg_ear / self.EAR_THRESHOLD)
            
            # Store EAR history for trend analysis
            self.ear_history.append(avg_ear)
            if len(self.ear_history) > 30:
                self.ear_history.pop(0)
            
            # Calculate detection confidence based on landmark stability
            detection_result['detection_confidence'] = min(1.0, len(self.ear_history) / 30.0)
        
        return detection_result
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            'total_blinks': self.total_blinks,
            'ear_history_length': len(self.ear_history),
            'avg_ear_recent': np.mean(self.ear_history[-10:]) if len(self.ear_history) >= 10 else 1.0,
            'detection_method': 'MediaPipe + OpenCV Professional'
        }