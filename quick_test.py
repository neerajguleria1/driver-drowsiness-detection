"""
Quick OpenCV Test - Capture and show one frame
"""
import cv2

print("Testing OpenCV and Webcam...")
print("=" * 50)

# Test 1: OpenCV version
print(f"OpenCV Version: {cv2.__version__}")

# Test 2: Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot access webcam")
    exit(1)

print("SUCCESS: Webcam opened")

# Test 3: Read one frame
ret, frame = cap.read()
if not ret:
    print("ERROR: Cannot read frame")
    cap.release()
    exit(1)

print(f"SUCCESS: Frame captured - Size: {frame.shape}")

# Test 4: Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

print(f"Faces detected: {len(faces)}")

# Show the frame
cv2.putText(frame, "OpenCV Working! Press any key to close", (10, 30),
           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

if len(faces) > 0:
    cv2.putText(frame, f"Face Detected!", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('OpenCV Test', frame)
print("\nA window should open showing your webcam.")
print("Press ANY KEY in the window to close it.")
cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 50)
print("OpenCV is working correctly!")
print("=" * 50)
