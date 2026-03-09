"""
Professional Computer Vision Demo
Uses MediaPipe for production-grade face detection + drowsiness analysis
"""
import cv2
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.professional_cv import ProfessionalEyeDetector
    PROFESSIONAL_MODE = True
except ImportError:
    print("⚠️  MediaPipe not installed. Install with: pip install mediapipe")
    from src.cv_detector import EyeDetector
    PROFESSIONAL_MODE = False

def main():
    print("=" * 70)
    if PROFESSIONAL_MODE:
        print("  🚀 PROFESSIONAL COMPUTER VISION DEMO")
        print("  MediaPipe + OpenCV Production-Grade Detection")
    else:
        print("  📹 STANDARD COMPUTER VISION DEMO") 
        print("  OpenCV Haar Cascade Detection")
    print("=" * 70)
    
    print("\nFeatures:")
    if PROFESSIONAL_MODE:
        print("  ✅ 468-point facial landmarks (MediaPipe)")
        print("  ✅ Precise eye contour detection")
        print("  ✅ Advanced head pose estimation")
        print("  ✅ Production-grade accuracy")
    else:
        print("  ✅ Face detection (Haar Cascade)")
        print("  ✅ Eye detection and tracking")
    
    print("  ✅ Real-time EAR calculation")
    print("  ✅ Blink detection & counting")
    print("  ✅ Drowsiness alerts")
    print("  ✅ Performance metrics")
    print("\nPress 'q' to quit, 's' for stats\n")
    
    # Initialize detector
    if PROFESSIONAL_MODE:
        detector = ProfessionalEyeDetector()
        detect_method = detector.detect_drowsiness_professional
    else:
        detector = EyeDetector()
        detect_method = detector.detect_drowsiness_from_frame
    
    # Camera setup with multiple backends
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
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    start_time = time.time()
    show_stats = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        
        # Perform detection
        result = detect_method(frame)
        
        # Calculate FPS
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        # Draw professional UI
        draw_professional_ui(frame, result, fps, frame_count, PROFESSIONAL_MODE)
        
        # Show stats overlay if requested
        if show_stats and PROFESSIONAL_MODE:
            draw_stats_overlay(frame, detector.get_performance_stats())
        
        # Display frame
        window_title = "Professional CV Demo" if PROFESSIONAL_MODE else "Standard CV Demo"
        cv2.imshow(window_title, frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_stats = not show_stats
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print(f"\n{'='*70}")
    print(f"  DEMO COMPLETED")
    print(f"{'='*70}")
    print(f"Frames processed: {frame_count}")
    print(f"Average FPS: {fps:.1f}")
    print(f"Total runtime: {elapsed:.1f}s")
    if PROFESSIONAL_MODE:
        stats = detector.get_performance_stats()
        print(f"Total blinks detected: {stats['total_blinks']}")
        print(f"Detection method: {stats['detection_method']}")
    print(f"{'='*70}")

def draw_professional_ui(frame, result, fps, frame_count, is_professional):
    """Draw professional-looking UI overlay"""
    h, w = frame.shape[:2]
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    
    if result['face_detected']:
        # Main status panel
        panel_color = (0, 50, 0) if not result.get('drowsy_detected', False) else (0, 0, 50)
        cv2.rectangle(overlay, (10, 10), (400, 200), panel_color, -1)
        
        # Detection info
        y_pos = 35
        cv2.putText(frame, "FACE: DETECTED", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if is_professional and 'landmarks_count' in result:
            y_pos += 25
            cv2.putText(frame, f"LANDMARKS: {result['landmarks_count']}", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # EAR values with color coding
        y_pos += 30
        ear_color = (0, 255, 0) if result.get('avg_ear', 1.0) > 0.25 else (0, 165, 255) if result.get('avg_ear', 1.0) > 0.15 else (0, 0, 255)
        cv2.putText(frame, f"EAR: {result.get('avg_ear', 1.0):.3f}", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
        
        # Individual eye EAR
        y_pos += 25
        cv2.putText(frame, f"L: {result.get('left_ear', 1.0):.3f}  R: {result.get('right_ear', 1.0):.3f}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Blink counter
        y_pos += 25
        blink_color = (0, 255, 255) if result.get('blink_detected', False) else (255, 255, 255)
        cv2.putText(frame, f"BLINKS: {result.get('total_blinks', 0)}", (20, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, blink_color, 2)
        
        # Alertness score bar
        y_pos += 30
        alertness = result.get('alertness_score', 1.0)
        bar_width = int(200 * alertness)
        cv2.rectangle(frame, (20, y_pos), (220, y_pos + 15), (100, 100, 100), 2)
        bar_color = (0, int(255*alertness), int(255*(1-alertness)))
        cv2.rectangle(frame, (20, y_pos), (20 + bar_width, y_pos + 15), bar_color, -1)
        cv2.putText(frame, f"ALERT: {alertness:.2f}", (230, y_pos + 12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Head pose info (if available)
        if 'head_pose' in result and result['head_pose']:
            y_pos += 25
            head_status = []
            if result['head_pose'].get('head_tilted', False):
                head_status.append("TILTED")
            if result['head_pose'].get('head_turned', False):
                head_status.append("TURNED")
            if not head_status:
                head_status.append("CENTERED")
            
            cv2.putText(frame, f"HEAD: {' | '.join(head_status)}", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    else:
        # No face detected
        cv2.rectangle(overlay, (10, 10), (300, 80), (0, 0, 50), -1)
        cv2.putText(frame, "FACE: NOT DETECTED", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Position face in camera", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Blend overlay
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Status indicator
    status_y = h - 100
    if result.get('drowsy_detected', False):
        # Flashing red warning
        if (frame_count // 10) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 8)
        cv2.putText(frame, "⚠️ DROWSINESS DETECTED ⚠️", (20, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    elif result.get('blink_detected', False):
        cv2.putText(frame, "👁️ BLINK", (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "✅ ALERT", (20, status_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    # Performance info
    perf_y = h - 30
    mode_text = "PROFESSIONAL" if is_professional else "STANDARD"
    cv2.putText(frame, f"{mode_text} | FPS: {fps:.1f} | Frame: {frame_count}", 
               (20, perf_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_stats_overlay(frame, stats):
    """Draw detailed statistics overlay"""
    h, w = frame.shape[:2]
    
    # Stats panel
    cv2.rectangle(frame, (w-250, 10), (w-10, 150), (0, 0, 0), -1)
    cv2.rectangle(frame, (w-250, 10), (w-10, 150), (255, 255, 255), 2)
    
    y_pos = 35
    cv2.putText(frame, "STATISTICS", (w-240, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    y_pos += 25
    cv2.putText(frame, f"Method: {stats['detection_method'][:15]}", (w-240, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    y_pos += 20
    cv2.putText(frame, f"Total Blinks: {stats['total_blinks']}", (w-240, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    y_pos += 20
    cv2.putText(frame, f"EAR History: {stats['ear_history_length']}", (w-240, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    y_pos += 20
    cv2.putText(frame, f"Avg EAR: {stats['avg_ear_recent']:.3f}", (w-240, y_pos), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

if __name__ == "__main__":
    main()