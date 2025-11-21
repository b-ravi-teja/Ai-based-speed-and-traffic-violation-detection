# Logic & Algorithms Explained

This document explains the specific algorithms and logic used in `combined_violation_system.py`.

## 1. Vehicle Tracking (YOLOv8 + ByteTrack)
Instead of just detecting vehicles in every frame independently, we "track" them.
- **Code**: `model.track(frame, persist=True)`
- **How it works**: The tracker looks at the position and visual features of a car in Frame 1 and finds the "same" car in Frame 2.
- **Result**: Each car gets a unique `track_id` (e.g., 1, 2, 52). This allows us to remember *when* a specific car entered the zone.

## 2. State-Based Speed Calculation
We use a "State Machine" approach to calculate speed accurately.

### The Problem with Simple Line Crossing
Simply checking `if y == line_y` often fails because vehicles move fast. In one frame, a car might be at $y=240$, and in the next, it's at $y=260$. It never exactly equals $250$, so the code misses it.

### The Solution: State Tracking
We track the **status** of every vehicle ID:
1.  **`approaching`**: The vehicle is above Zone 1.
2.  **`in_zone`**: The vehicle has crossed Zone 1 but hasn't reached Zone 2.
3.  **`past_zone`**: The vehicle has crossed Zone 2.

**Logic Flow:**
- If a vehicle is `approaching` AND its Y-coordinate becomes greater than `ZONE_1_Y`:
    - **Event**: Entered Zone 1.
    - **Action**: Record `start_time`. Change status to `in_zone`.
- If a vehicle is `in_zone` AND its Y-coordinate becomes greater than `ZONE_2_Y`:
    - **Event**: Exited Zone 2.
    - **Action**: Record `end_time`. Calculate Speed.

### The Math
$$ Speed (km/h) = \left( \frac{\text{Distance (meters)}}{\text{Time (seconds)}} \right) \times 3.6 $$
- `DISTANCE_METERS`: The physical distance between the two lines on the road (e.g., 15 meters).
- `Time`: `end_time - start_time`.

## 3. Optical Character Recognition (OCR) Pipeline
Reading license plates from a wide-angle traffic camera is difficult. We use a multi-step pipeline to improve accuracy.

### Step A: Localization
We need to find *where* the plate is on the car.
- **Method 1 (Best)**: Use a custom YOLO model trained on license plates (`license_plate_detector.pt`).
- **Method 2 (Fallback)**: If no model is found, we guess. Plates are usually at the **bottom center** of the vehicle. We crop the bottom 50% of the vehicle image.

### Step B: Preprocessing (The "Enhancement" Phase)
Raw camera images are often too blurry or dark for OCR. We fix this:
1.  **Upscaling**: We resize the plate image by **3x**. This makes small text look larger and clearer to the OCR engine.
2.  **Grayscale**: Color doesn't help read text, so we remove it.
3.  **Thresholding**: We convert the gray image into pure Black & White.
    - *Adaptive Thresholding*: Good for uneven lighting (shadows).
    - *Otsu's Binarization*: Good for high-contrast plates.

### Step C: Recognition (EasyOCR)
We feed the processed images to EasyOCR.
- We try multiple versions (Raw Gray, Adaptive Thresh, Otsu) and accept the first one that returns a valid alphanumeric string (length > 3).
