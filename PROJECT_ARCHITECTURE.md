# Project Architecture: Smart Traffic Violation Detection

## System Overview
This project is a real-time computer vision application designed to detect traffic violations. It processes video input (from a file or live camera), identifies vehicles, tracks their movement, calculates their speed, and reads their license plates if they commit a violation.

## Core Components

### 1. Input Layer
- **Source**: Video file (`red_light_violation.mp4`) or IP Camera Stream.
- **Preprocessing**: Frames are resized to 800x600 for consistent processing speed and coordinate mapping.

### 2. Detection & Tracking Layer (YOLOv8)
- **Model**: `yolov8n.pt` (Nano version for speed).
- **Function**: Detects vehicles (Car, Bus, Truck, Motorbike) in every frame.
- **Tracking**: Uses the **ByteTrack** algorithm (built into YOLOv8 via `persist=True`) to assign a unique ID to each vehicle. This ID stays the same as long as the vehicle is in the frame, which is crucial for calculating speed.

### 3. Logic Layer (The "Brain")
- **Speed Calculation**:
    - Uses two virtual lines (`ZONE_1` and `ZONE_2`).
    - Measures the time taken for a specific Vehicle ID to travel between these lines.
    - Formula: $Speed = \frac{Distance}{Time} \times 3.6$ (to convert m/s to km/h).
- **Red Light Detection**:
    - Checks if the traffic signal state is "RED".
    - If a vehicle crosses the stop line (`ZONE_2`) while the signal is RED, it's flagged.

### 4. Recognition Layer (OCR)
- **Trigger**: Activates ONLY when a violation (Speeding or Red Light) is detected.
- **Process**:
    1.  **Crop**: Extracts the image of the violating vehicle.
    2.  **Plate Detection**: Uses a heuristic (bottom center) or a specific YOLO model to find the plate.
    3.  **Enhancement**: Upscales and thresholds the plate image to make text clearer.
    4.  **Reading**: Uses **EasyOCR** to convert the image text into a string (e.g., "MH12DE1433").

### 5. Output Layer
- **Visuals**: Draws bounding boxes, speed, and violation alerts on the video.
- **Logging**: Saves details to `violations_log.csv`.
- **Evidence**: Saves a snapshot of the violation to the `violations/` folder.

## Data Flow
```mermaid
graph TD
    A[Video Input] --> B[Resize Frame]
    B --> C[YOLOv8 Tracking]
    C --> D{Vehicle in Zone?}
    D -- Yes --> E[Update State (Enter/Exit)]
    E --> F{Violation?}
    F -- Speed > Limit --> G[Trigger OCR]
    F -- Red Light Jump --> G
    G --> H[Save Image & Log to CSV]
    F -- No --> I[Continue]
```
