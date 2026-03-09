"""
Real-Time Computer Vision Demo
Demonstrates actual eye detection and drowsiness monitoring using webcam
"""
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cv_detector import EyeDetector
import time

def main():
    print("=" * 60)
    print("  REAL COMPUTER VISION - Eye Detection Demo")
    print("=" * 60)
    print("\nThis demo uses REAL OpenCV for:")
    print("  - Face detection (Haar Cascade)")
    print("  - Eye detection")
    print("  - Eye Aspect Ratio (EAR) calculation")
    print("  - Real-time drowsiness detection")
    print("\nPress 'q' to quit\n")
    
    detector = EyeDetector()
    
    # Try different camera backends
    cap = None
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    
    for backend in backends:
        print(f"Trying camera backend {backend}...")
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                print(f"SUCCESS: Camera working with backend {backend}")
                break
            cap.release()
        cap = None
    
    if not cap or not cap.isOpened():
        print("ERROR: Could not access webcam")
        print("Solutions:")
        print("1. Close other apps using camera (Skype, Teams, etc.)")
        print("2. Check camera permissions in Windows Settings")
        print("3. Try unplugging/reconnecting USB camera")
        return
    
    print("Starting real-time detection...\n")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Perform real CV analysis
        result = detector.detect_drowsiness_from_frame(frame)
        frame_count += 1
        
        # Draw enhanced results with more features
        if result['face_detected']:
            # Basic info
            cv2.putText(frame, "Face: DETECTED", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Eyes: {result['eyes_detected']}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # EAR values with color coding
            ear_color = (0, 255, 0) if result['ear_score'] > 0.25 else (0, 165, 255) if result['ear_score'] > 0.15 else (0, 0, 255)
            cv2.putText(frame, f"EAR: {result['ear_score']:.3f}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
            
            # Individual eye EAR
            cv2.putText(frame, f"L: {result['left_ear']:.3f} R: {result['right_ear']:.3f}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Blink counter and detection
            blink_color = (0, 255, 255) if result.get('blink_detected', False) else (255, 255, 255)
            cv2.putText(frame, f"Blinks: {result['total_blinks']}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, blink_color, 2)
            
            # Eye closure duration
            if result.get('eye_closure_duration', 0) > 0:
                cv2.putText(frame, f"Closed: {result['eye_closure_duration']} frames", 
                           (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # Alert level bar
            alert_level = result.get('alert_level', 1.0)
            bar_width = int(200 * alert_level)
            cv2.rectangle(frame, (10, 200), (210, 220), (100, 100, 100), 2)
            bar_color = (0, int(255*alert_level), int(255*(1-alert_level)))
            cv2.rectangle(frame, (10, 200), (10 + bar_width, 220), bar_color, -1)
            cv2.putText(frame, f"Alert: {alert_level:.2f}", (220, 215),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Additional features
            if result.get('smile_detected', False):
                cv2.putText(frame, "SMILE DETECTED!", (10, 250),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if result.get('head_tilted', False):
                cv2.putText(frame, "HEAD TILTED", (10, 280),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            
            # EAR trend
            if 'ear_trend' in result:
                trend_color = (0, 0, 255) if result['ear_trend'] == 'decreasing' else (0, 255, 0)
                cv2.putText(frame, f"Trend: {result['ear_trend']}", (10, 310),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, trend_color, 1)
            
            # Main status with enhanced visuals
            if result['drowsy_detected']:
                # Flashing red border
                border_color = (0, 0, 255) if (frame_count // 10) % 2 == 0 else (0, 0, 150)
                cv2.rectangle(frame, (5, 5), (frame.shape[1]-5, 350), border_color, 5)
                
                cv2.putText(frame, "STATUS: DROWSY!", (10, 380),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                # Show drowsy duration
                if result.get('drowsy_duration', 0) > 0:
                    cv2.putText(frame, f"Drowsy for: {result['drowsy_duration']:.1f}s", 
                               (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
            elif result.get('blink_detected', False):
                cv2.putText(frame, "STATUS: BLINK", (10, 380),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "STATUS: ALERT", (10, 380),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Face: NOT DETECTED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Move closer to camera", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Show FPS
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Real-Time Drowsiness Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print(f"  Demo completed!")
    print(f"  Frames processed: {frame_count}")
    print(f"  Average FPS: {fps:.1f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
