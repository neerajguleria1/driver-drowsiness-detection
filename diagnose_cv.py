"""
Diagnostic Script - Check what's wrong with CV
"""
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cv_detector import EyeDetector

print("=" * 60)
print("DIAGNOSTIC TEST - What's Not Working?")
print("=" * 60)

detector = EyeDetector()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam not accessible")
    exit(1)

print("\nCapturing 10 frames to diagnose issues...\n")

for i in range(10):
    ret, frame = cap.read()
    if not ret:
        print(f"Frame {i+1}: FAILED to read")
        continue
    
    result = detector.detect_drowsiness_from_frame(frame)
    
    print(f"Frame {i+1}:")
    print(f"  Face detected: {result['face_detected']}")
    print(f"  Eyes detected: {result['eyes_detected']}")
    print(f"  EAR score: {result['ear_score']:.3f}")
    print(f"  Alertness: {result['alertness_cv']:.2f}")
    print(f"  Drowsy: {result['drowsy_detected']}")
    print()

cap.release()

print("=" * 60)
print("DIAGNOSIS:")
print("=" * 60)

# Analyze results
print("\nWhat issues did you see?")
print("1. Face not detected? → Try better lighting or move closer")
print("2. Eyes always 0? → Face camera directly, remove glasses")
print("3. EAR not changing? → Blink slowly and deliberately")
print("4. Never shows drowsy? → Close eyes for 3+ seconds")
print("\nTell me which issue you're seeing!")
