"""
Test script for Computer Vision implementation
Tests all CV components before demo
"""
import sys
import os

def test_imports():
    """Test if all required packages are installed"""
    print("=" * 60)
    print("TEST 1: Checking Dependencies")
    print("=" * 60)
    
    try:
        import cv2
        print("OK OpenCV installed:", cv2.__version__)
    except ImportError:
        print("FAIL OpenCV not installed. Run: pip install opencv-python")
        return False
    
    try:
        import scipy
        print("OK SciPy installed:", scipy.__version__)
    except ImportError:
        print("FAIL SciPy not installed. Run: pip install scipy")
        return False
    
    try:
        import numpy
        print("OK NumPy installed:", numpy.__version__)
    except ImportError:
        print("FAIL NumPy not installed")
        return False
    
    print("\nPASS All dependencies installed!\n")
    return True

def test_cv_module():
    """Test if CV detector module works"""
    print("=" * 60)
    print("TEST 2: Testing CV Detector Module")
    print("=" * 60)
    
    try:
        from src.cv_detector import EyeDetector
        detector = EyeDetector()
        print("OK EyeDetector class loaded")
        print(f"OK EAR threshold: {detector.EAR_THRESHOLD}")
        print(f"OK Consecutive frames: {detector.CONSECUTIVE_FRAMES}")
        print("\nPASS CV module working!\n")
        return True
    except Exception as e:
        print(f"FAIL Error loading CV module: {e}")
        return False

def test_haar_cascades():
    """Test if Haar Cascades are available"""
    print("=" * 60)
    print("TEST 3: Testing Haar Cascades")
    print("=" * 60)
    
    try:
        import cv2
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        if face_cascade.empty():
            print("FAIL Face cascade not loaded")
            return False
        print("OK Face cascade loaded")
        
        if eye_cascade.empty():
            print("FAIL Eye cascade not loaded")
            return False
        print("OK Eye cascade loaded")
        
        print("\nPASS Haar Cascades working!\n")
        return True
    except Exception as e:
        print(f"FAIL Error loading cascades: {e}")
        return False

def test_webcam():
    """Test if webcam is accessible"""
    print("=" * 60)
    print("TEST 4: Testing Webcam Access")
    print("=" * 60)
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("WARN Webcam not accessible")
            print("   This is OK - you can still use image/video files")
            print("   Make sure no other app is using the camera")
            cap.release()
            return True
        
        ret, frame = cap.read()
        cap.release()
        
        if ret and frame is not None:
            print(f"OK Webcam accessible")
            print(f"OK Frame size: {frame.shape}")
            print("\nPASS Webcam working!\n")
            return True
        else:
            print("WARN Webcam opened but can't read frames")
            return True
            
    except Exception as e:
        print(f"WARN Webcam test failed: {e}")
        print("   This is OK - you can still use image/video files")
        return True

def test_api_integration():
    """Test if API integration works"""
    print("=" * 60)
    print("TEST 5: Testing API Integration")
    print("=" * 60)
    
    try:
        from src.cv_api import router
        print("OK CV API router loaded")
        print(f"OK Router prefix: {router.prefix}")
        print(f"OK Number of routes: {len(router.routes)}")
        print("\nPASS API integration working!\n")
        return True
    except Exception as e:
        print(f"FAIL Error loading API: {e}")
        return False

def test_demo_script():
    """Test if demo script exists and is valid"""
    print("=" * 60)
    print("TEST 6: Testing Demo Script")
    print("=" * 60)
    
    demo_path = "demo/cv_demo.py"
    if not os.path.exists(demo_path):
        print(f"FAIL Demo script not found: {demo_path}")
        return False
    
    print(f"OK Demo script exists: {demo_path}")
    
    try:
        with open(demo_path, 'r') as f:
            content = f.read()
            if 'EyeDetector' in content and 'cv2.VideoCapture' in content:
                print("OK Demo script contains required code")
                print("\nPASS Demo script ready!\n")
                return True
    except Exception as e:
        print(f"FAIL Error reading demo script: {e}")
        return False

def main():
    print("\n")
    print("=" * 60)
    print("          COMPUTER VISION TEST SUITE")
    print("=" * 60)
    print("\n")
    
    tests = [
        ("Dependencies", test_imports),
        ("CV Module", test_cv_module),
        ("Haar Cascades", test_haar_cascades),
        ("Webcam", test_webcam),
        ("API Integration", test_api_integration),
        ("Demo Script", test_demo_script)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"FAIL Test '{name}' crashed: {e}\n")
            results.append((name, False))
    
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {name}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\nSUCCESS! All tests passed! Computer Vision is ready!")
        print("\nNext steps:")
        print("  1. Run: python demo/cv_demo.py")
        print("  2. Or run: RUN_CV_DEMO.bat")
        print("  3. Or start API: uvicorn src.app:app --reload")
    elif passed >= total - 1:
        print("\nMOSTLY WORKING! Minor issues but CV is usable.")
        print("\nYou can proceed with:")
        print("  - python demo/cv_demo.py (if webcam passed)")
        print("  - API endpoints for image/video analysis")
    else:
        print("\nISSUES DETECTED! Please fix the failed tests above.")
        print("\nCommon fixes:")
        print("  - pip install opencv-python scipy")
        print("  - Close other apps using webcam")
        print("  - Check file paths are correct")
    
    print("\n")

if __name__ == "__main__":
    main()
