"""
Simple Computer Vision Demo - Works with just OpenCV
No external dependencies, reliable and impressive
"""
import cv2
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simple_cv import SimpleEyeDetector

def main():
    print("=" * 70)
    print("  🚀 FAANG-LEVEL COMPUTER VISION DEMO")
    print("  Production-Grade OpenCV Detection")
    print("=" * 70)
    
    print("\nFeatures:")
    print("  ✅ Real-time face detection")
    print("  ✅ Precise eye tracking")
    print("  ✅ EAR calculation & analysis")
    print("  ✅ Blink detection & counting")
    print("  ✅ Drowsiness alerts")
    print("  ✅ Professional UI")
    print("  ✅ Performance metrics")
    print("  ✅ Alert level tracking")
    print("\nPress 'q' to quit\n")
    
    detector = SimpleEyeDetector()
    
    # Camera setup
    cap = None
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    
    for backend in backends:
        print(f"Trying camera backend {backend}...")
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret:
                print(f"✅ Camera working with backend {backend}")
                break
            cap.release()
        cap = None
    
    if not cap:
        print("❌ Could not access webcam")
        return
    
    # Optimize camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    start_time = time.time()
    
    print("🎥 Starting real-time detection...")    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        result = detector.detect_drowsiness_from_frame(frame)
        
        # Calculate FPS
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Draw professional UI
        draw_professional_interface(frame, result, fps, frame_count)
        
        # Display
        cv2.imshow('FAANG-Level Computer Vision Demo', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final stats
    print(f"\n{'='*70}")
    print(f"  DEMO COMPLETED - FAANG-LEVEL PERFORMANCE")
    print(f"{'='*70}")
    print(f"Frames processed: {frame_count}")
    print(f"Average FPS: {fps:.1f}")
    print(f"Total blinks: {detector.total_blinks}")
    print(f"Runtime: {elapsed:.1f}s")
    print(f"{'='*70}")

def draw_professional_interface(frame, result, fps, frame_count):
    """Draw FAANG-level professional interface"""
    h, w = frame.shape[:2]
    
    # Semi-transparent overlay
    overlay = frame.copy()
    
    if result['face_detected']:
        # Main info panel
        panel_color = (0, 50, 0) if not result['drowsy_detected'] else (0, 0, 50)
        cv2.rectangle(overlay, (10, 10), (450, 220), panel_color, -1)
        
        # Detection status
        cv2.putText(frame, "FACE: DETECTED ✓", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"EYES: {result['eyes_detected']}", (20, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # EAR with color coding
        ear_color = (0, 255, 0) if result['ear_score'] > 0.25 else (0, 165, 255) if result['ear_score'] > 0.15 else (0, 0, 255)
        cv2.putText(frame, f"EAR: {result['ear_score']:.3f}", (20, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
        
        # Individual eyes
        cv2.putText(frame, f"LEFT: {result['left_ear']:.3f}  RIGHT: {result['right_ear']:.3f}", 
                   (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Blink counter
        blink_color = (0, 255, 255) if result['blink_detected'] else (255, 255, 255)
        cv2.putText(frame, f"BLINKS: {result['total_blinks']}", (20, 155), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, blink_color, 2)
        
        # Alert level bar (FAANG-style)
        alert_level = result['alert_level']
        bar_width = int(250 * alert_level)
        cv2.rectangle(frame, (20, 175), (270, 195), (100, 100, 100), 2)
        bar_color = (0, int(255*alert_level), int(255*(1-alert_level)))
        cv2.rectangle(frame, (20, 175), (20 + bar_width, 195), bar_color, -1)
        cv2.putText(frame, f"ALERTNESS: {alert_level:.2f}", (280, 190), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Additional features
        if result['smile_detected']:
            cv2.putText(frame, "😊 SMILE DETECTED", (20, 210), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        # No face panel
        cv2.rectangle(overlay, (10, 10), (350, 80), (0, 0, 50), -1)
        cv2.putText(frame, "FACE: NOT DETECTED ❌", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Position face in camera view", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Blend overlay
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Main status (bottom)
    status_y = h - 120
    if result['drowsy_detected']:
        # Animated warning
        if (frame_count // 10) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
        cv2.putText(frame, "⚠️  DROWSINESS ALERT  ⚠️", (20, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        cv2.putText(frame, "TAKE A BREAK!", (20, status_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    elif result['blink_detected']:
        cv2.putText(frame, "👁️ BLINK DETECTED", (20, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    else:
        cv2.putText(frame, "✅ DRIVER ALERT", (20, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    
    # Performance metrics (top right)
    perf_y = 30
    cv2.putText(frame, f"FPS: {fps:.1f}", (w-150, perf_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"FRAME: {frame_count}", (w-150, perf_y + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, "FAANG-LEVEL", (w-150, perf_y + 45), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

if __name__ == "__main__":
    main()