# 📹 How to Check OpenCV Outputs

## What You Should See

### On the Video Window:

```
┌─────────────────────────────────────────┐
│  Face: DETECTED                         │  ← Shows if face found
│  Eyes: 2                                │  ← Number of eyes detected
│  EAR: 0.285                             │  ← Eye Aspect Ratio score
│  STATUS: ALERT                          │  ← Drowsiness status
│                                         │
│         [Your Face Here]                │
│                                         │
│  FPS: 28.5                              │  ← Processing speed
└─────────────────────────────────────────┘
```

### What Each Output Means:

1. **Face: DETECTED / NOT DETECTED**
   - Green = Face found ✅
   - Red = No face ❌

2. **Eyes: 0, 1, or 2**
   - 2 = Both eyes detected ✅
   - 1 = One eye detected ⚠️
   - 0 = No eyes detected ❌

3. **EAR: 0.XXX** (Eye Aspect Ratio)
   - > 0.25 = Eyes OPEN (Alert) ✅
   - < 0.25 = Eyes CLOSING (Drowsy) ⚠️

4. **STATUS:**
   - Green "ALERT" = You're awake ✅
   - Red "DROWSY!" = Eyes closed detected ⚠️

5. **FPS: XX.X**
   - Shows processing speed
   - 25+ FPS = Good performance ✅

---

## 🧪 Test Scenarios

### Test 1: Normal (Alert)
**Action:** Look at camera normally
**Expected Output:**
```
Face: DETECTED
Eyes: 2
EAR: 0.28-0.35
STATUS: ALERT (Green)
```

### Test 2: Close Eyes (Drowsy)
**Action:** Close your eyes for 2-3 seconds
**Expected Output:**
```
Face: DETECTED
Eyes: 0
EAR: 0.15-0.20
STATUS: DROWSY! (Red)
```

### Test 3: Look Away
**Action:** Turn your head away
**Expected Output:**
```
Face: NOT DETECTED
STATUS: (No status shown)
```

---

## 📊 Console Output

After you press 'q' to quit, you'll see:

```
============================================================
  Demo completed!
  Frames processed: 850
  Average FPS: 28.3
============================================================
```

---

## 🔍 How to Verify It's Working

### Step-by-Step Test:

1. **Run the demo:**
   ```bash
   python demo/cv_demo.py
   ```

2. **Check face detection:**
   - Look at camera → Should say "Face: DETECTED"
   - Turn away → Should say "Face: NOT DETECTED"

3. **Check eye detection:**
   - Open eyes → "Eyes: 2"
   - Close one eye → "Eyes: 1"
   - Close both → "Eyes: 0"

4. **Check drowsiness detection:**
   - Keep eyes open → "STATUS: ALERT" (Green)
   - Close eyes for 3 seconds → "STATUS: DROWSY!" (Red)

5. **Check EAR score:**
   - Eyes open → EAR around 0.28-0.35
   - Eyes closing → EAR drops below 0.25
   - Eyes closed → EAR around 0.10-0.20

---

## 📸 Save Output to File

Want to save the results? Run this:

```bash
python demo/cv_demo.py > cv_output.txt
```

---

## 🎥 Record Video Output

To record the video with annotations:

```python
# Add this to cv_demo.py after line 30:
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

# Add this inside the while loop after drawing:
out.write(frame)

# Add this before cap.release():
out.release()
```

---

## 🐛 Troubleshooting

### "Face: NOT DETECTED" always shows
- **Fix:** Ensure good lighting
- **Fix:** Face the camera directly
- **Fix:** Move closer to camera

### "Eyes: 0" even when eyes open
- **Fix:** Look directly at camera
- **Fix:** Remove glasses (if wearing)
- **Fix:** Increase lighting

### EAR score not changing
- **Fix:** Blink slowly and deliberately
- **Fix:** Close eyes for 2-3 seconds
- **Fix:** Ensure face is detected first

### Window doesn't open
- **Fix:** Check if another app is using webcam
- **Fix:** Try: `python quick_test.py` first
- **Fix:** Restart terminal

---

## ✅ Success Checklist

- [ ] Video window opens
- [ ] Face detection works (green text)
- [ ] Eye count shows (0, 1, or 2)
- [ ] EAR score changes when blinking
- [ ] "DROWSY!" appears when eyes closed
- [ ] FPS shows (20+ is good)
- [ ] Can quit with 'q' key

**If all checked → OpenCV is working perfectly! 🎉**
