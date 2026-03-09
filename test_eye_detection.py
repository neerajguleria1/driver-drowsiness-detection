"""
Quick test for eye detection functionality
"""
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.cv_detector import EyeDetector

def test_eye_detection():
    print("Testing eye detection...")
    
    detector = EyeDetector()
    
    # Try different camera backends
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    
    for backend in backends:
        print(f"Trying backend {backend}...")
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                print(f"SUCCESS: Backend {backend} working")
                break
            cap.release()
        cap = None
    
    if not cap or not cap.isOpened():
        print("ERROR: Cannot access any webcam")
        print("Solutions:")
        print("1. Close other apps using camera (Skype, Teams, etc.)")
        print("2. Check camera permissions in Windows Settings")
        print("3. Try unplugging/reconnecting USB camera")
        return
    
    print("Testing for 10 seconds... Press 'q' to quit early")
    
    frame_count = 0
    face_detected_count = 0
    eyes_detected_count = 0
    
    while frame_count < 300:  # ~10 seconds at 30fps
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_count}")
            break
            
        result = detector.detect_drowsiness_from_frame(frame)
        frame_count += 1
        
        if result['face_detected']:
            face_detected_count += 1
        if result['eyes_detected'] >= 2:
            eyes_detected_count += 1
            
        # Show live feed with detection info
        status_text = f"Face: {'YES' if result['face_detected'] else 'NO'} | Eyes: {result['eyes_detected']} | EAR: {result['ear_score']:.3f}"
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if result['drowsy_detected']:
            cv2.putText(frame, "DROWSY DETECTED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        cv2.imshow('Eye Detection Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Results - fix division by zero
    print(f"\nTest Results:")
    print(f"Total frames: {frame_count}")
    
    if frame_count > 0:
        print(f"Face detection rate: {face_detected_count/frame_count*100:.1f}%")
        print(f"Eye detection rate: {eyes_detected_count/frame_count*100:.1f}%")
        
        if face_detected_count < frame_count * 0.5:
            print("⚠️  Low face detection - check lighting and camera position")
        if eyes_detected_count < frame_count * 0.3:
            print("⚠️  Low eye detection - may need better lighting or closer positioning")
        else:
            print("✅ Eye detection working properly!")
    else:
        print("❌ No frames processed - camera issue")

if __name__ == "__main__":
    test_eye_detection()