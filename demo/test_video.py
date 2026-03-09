"""
Test Computer Vision with Video File
Analyzes a video file for drowsiness detection
"""
import cv2
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cv_detector import EyeDetector

def analyze_video(video_path):
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        return
    
    detector = EyeDetector()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("ERROR: Could not open video file")
        return
    
    print("=" * 60)
    print(f"Analyzing video: {video_path}")
    print("=" * 60)
    print("Press 'q' to quit, SPACE to pause\n")
    
    frame_count = 0
    drowsy_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = detector.detect_drowsiness_from_frame(frame)
            frame_count += 1
            
            if result['drowsy_detected']:
                drowsy_count += 1
            
            # Draw results
            if result['face_detected']:
                cv2.putText(frame, "Face: DETECTED", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Eyes: {result['eyes_detected']}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"EAR: {result['ear_score']:.3f}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                if result['drowsy_detected']:
                    cv2.putText(frame, "STATUS: DROWSY!", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "STATUS: ALERT", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Face: NOT DETECTED", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Video Analysis', frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"Total frames: {frame_count}")
    print(f"Drowsy frames: {drowsy_count}")
    print(f"Drowsiness rate: {(drowsy_count/frame_count*100):.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        print("Usage: python test_video.py <video_file>")
        print("\nExample:")
        print("  python demo/test_video.py video.mp4")
        print("  python demo/test_video.py C:/path/to/video.mp4")
        sys.exit(1)
    
    analyze_video(video_path)
