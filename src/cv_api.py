"""
Computer Vision API Endpoints
Integrates real OpenCV-based eye detection with ML pipeline
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from src.cv_detector import EyeDetector
import cv2
import numpy as np
from typing import Dict
import tempfile
import os

router = APIRouter(prefix="/v1/cv", tags=["Computer Vision"])
detector = EyeDetector()

@router.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)) -> Dict:
    """
    Analyze uploaded image for drowsiness using real computer vision
    Returns: Face detection, eye detection, EAR score, alertness level
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(400, "File must be an image")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        result = detector.analyze_image(tmp_path)
        return {
            "status": "success",
            "cv_analysis": result,
            "message": "Real computer vision analysis completed"
        }
    finally:
        os.unlink(tmp_path)

@router.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)) -> Dict:
    """
    Analyze uploaded video for drowsiness patterns
    Returns: Aggregated drowsiness metrics over video duration
    """
    if not file.content_type.startswith('video/'):
        raise HTTPException(400, "File must be a video")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        cap = cv2.VideoCapture(tmp_path)
        frames_analyzed = 0
        drowsy_frames = 0
        ear_scores = []
        
        while frames_analyzed < 150:  # Analyze first 5 seconds
            ret, frame = cap.read()
            if not ret:
                break
            result = detector.detect_drowsiness_from_frame(frame)
            if result['face_detected']:
                ear_scores.append(result['ear_score'])
                if result['drowsy_detected']:
                    drowsy_frames += 1
            frames_analyzed += 1
        
        cap.release()
        
        return {
            "status": "success",
            "frames_analyzed": frames_analyzed,
            "drowsy_frames": drowsy_frames,
            "avg_ear": float(np.mean(ear_scores)) if ear_scores else 1.0,
            "alertness_cv": 1.0 - (drowsy_frames / max(frames_analyzed, 1)),
            "drowsiness_probability": drowsy_frames / max(frames_analyzed, 1)
        }
    finally:
        os.unlink(tmp_path)

@router.get("/webcam-test")
async def webcam_test() -> Dict:
    """
    Test webcam availability for real-time detection
    """
    cap = cv2.VideoCapture(0)
    available = cap.isOpened()
    cap.release()
    
    return {
        "webcam_available": available,
        "message": "Webcam ready for real-time detection" if available else "No webcam detected"
    }
