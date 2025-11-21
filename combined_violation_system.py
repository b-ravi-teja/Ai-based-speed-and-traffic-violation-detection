import cv2
from ultralytics import YOLO
import easyocr
import os
from datetime import datetime
import csv
import time
import numpy as np

# === CONFIGURATION ===
# 🎥 Input Source
TEST_MODE = True  # Set to False for Live IP Camera
TEST_VIDEO_PATH = "red_light_violation.mp4"
IP_CAMERA_URL = "http://10.52.2.150:8080/video"

# 🧠 Models
VEHICLE_MODEL_PATH = "yolov8n.pt"
# If you have a specific plate model, put its path here. 
# Otherwise, we'll use a fallback heuristic.
LICENSE_PLATE_MODEL_PATH = None # e.g., "license_plate_detector.pt"

# 📏 Speed Detection Settings
# Adjust these lines based on your camera angle/video resolution (800x600)
ZONE_1_Y = 250   # First line (Entry)
ZONE_2_Y = 450   # Second line (Exit)
DISTANCE_METERS = 1.0 # ⚠️ ADJUSTED: 200px in this video is approx 1m physical distance
SPEED_LIMIT = 40  # km/h

# ⚙️ Detection Settings
CONF_THRESHOLD = 0.5
VEHICLE_CLASSES = [2, 3, 5, 7] # COCO IDs: 2=car, 3=motorcycle, 5=bus, 7=truck

# 📂 File Storage
VIOLATIONS_IMG_DIR = "violations"
VIOLATIONS_VIDEO_DIR = "violations_videos"
CSV_FILE = os.path.join(os.getcwd(), "violations", "speed_violations_log.csv")

# Ensure directories exist
os.makedirs(VIOLATIONS_IMG_DIR, exist_ok=True)
os.makedirs(VIOLATIONS_VIDEO_DIR, exist_ok=True)

# Initialize CSV
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp", "ViolationID", "Type", "LicensePlate",
            "Speed (km/h)", "Image", "Video"
        ])

# === INITIALIZATION ===
print("⏳ Loading models...")
vehicle_model = YOLO(VEHICLE_MODEL_PATH)
plate_model = YOLO(LICENSE_PLATE_MODEL_PATH) if LICENSE_PLATE_MODEL_PATH and os.path.exists(LICENSE_PLATE_MODEL_PATH) else None
reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if you have CUDA
print("✅ Models loaded.")

# Track history for speed calculation: {track_id: start_time}
track_history = {}
# To prevent duplicate logging for the same vehicle
logged_ids = set()

def preprocess_plate(img):
    """
    Preprocess image for better OCR: Grayscale -> Thresholding
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Adaptive thresholding to handle varying lighting
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    return denoised

def get_license_plate_text(vehicle_crop):
    """
    Extract license plate text from a vehicle crop.
    Uses a dedicated YOLO model if available, otherwise falls back to heuristics.
    """
    plate_crop = None
    
    if plate_model:
        # Use specific model to find plate within the vehicle crop
        results = plate_model(vehicle_crop, verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = vehicle_crop[y1:y2, x1:x2]
                break # Take the first confident plate
    
    if plate_crop is None:
        # Fallback: Heuristic crop (Bottom 40% of the vehicle, center area)
        h, w, _ = vehicle_crop.shape
        # Focus on bottom half, slightly narrower width
        plate_crop = vehicle_crop[int(h*0.5):h, int(w*0.1):int(w*0.9)]
    
    if plate_crop.size == 0:
        return "Unknown"

    # --- OCR Enhancement Strategy ---
    # 1. Upscale the image (Helps significantly with small text)
    plate_crop = cv2.resize(plate_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    
    # 2. Convert to Grayscale
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    
    # 3. Try multiple preprocessing techniques
    attempts = []
    attempts.append(gray) # Raw grayscale
    
    # Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    attempts.append(thresh)
    
    # Simple Thresholding (Otsu)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    attempts.append(otsu)

    # 4. Run OCR on each attempt until we find text
    for img in attempts:
        # allowlist: only alphanumeric characters
        ocr_results = reader.readtext(img, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        
        # Filter results: join them and check length
        text = "".join([t for t in ocr_results if len(t) > 2])
        
        if len(text) > 3: # If we found a decent string, return it
            return text
            
    return "Unknown"

def main():
    global track_history, logged_ids
    
    # Source Selection
    source = TEST_VIDEO_PATH if TEST_MODE else IP_CAMERA_URL
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open source {source}")
        return

    violation_count = 0
    
    cv2.namedWindow("Traffic Violation System", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Traffic Violation System", 1024, 768)

    print("▶️ System Started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            if TEST_MODE:
                print("🔁 Video ended, restarting...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                track_history.clear() # Clear history on restart
                logged_ids.clear()
                continue
            else:
                print("❌ Stream disconnected.")
                break

        # Resize for consistent processing speed and coordinate mapping
        frame = cv2.resize(frame, (800, 600))
        
        # --- 1. Vehicle Tracking ---
        # persist=True is CRITICAL for tracking IDs across frames
        results = vehicle_model.track(frame, persist=True, verbose=False, classes=VEHICLE_CLASSES, conf=CONF_THRESHOLD)
        
        # --- Time Measurement Fix ---
        # Use video timestamp for files, system time for live camera
        if TEST_MODE:
            # Get current timestamp in seconds from the video file
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        else:
            current_time = time.time()

        # Simulated Traffic Light (For demo purposes: 5s Red, 5s Green)
        is_red_light = (int(current_time) % 10) < 5 
        signal_color = (0, 0, 255) if is_red_light else (0, 255, 0)
        signal_text = "RED LIGHT" if is_red_light else "GREEN LIGHT"

        # Draw Interface
        cv2.line(frame, (0, ZONE_1_Y), (800, ZONE_1_Y), (255, 255, 0), 2) # Entry Line
        cv2.line(frame, (0, ZONE_2_Y), (800, ZONE_2_Y), (0, 255, 255), 2) # Exit Line
        cv2.putText(frame, f"Signal: {signal_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, signal_color, 3)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls in zip(boxes, track_ids, clss):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                vehicle_type = vehicle_model.names[cls]
                
                # Draw Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id} {vehicle_type}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # --- 2. Speed Calculation Logic (State-Based) ---
                speed = 0
                
                # Initialize state if new vehicle
                if track_id not in track_history:
                    track_history[track_id] = {"start_time": 0, "end_time": 0, "status": "approaching"}

                # Check Zone 1 Crossing (Entry)
                # If vehicle was 'approaching' and is now below Zone 1
                if track_history[track_id]["status"] == "approaching" and cy > ZONE_1_Y:
                    track_history[track_id]["start_time"] = current_time
                    track_history[track_id]["status"] = "in_zone"
                    print(f"✅ DEBUG: Vehicle {track_id} ENTERED Zone 1 at {track_history[track_id]['start_time']:.2f}")

                # Check Zone 2 Crossing (Exit)
                # If vehicle was 'in_zone' and is now below Zone 2
                elif track_history[track_id]["status"] == "in_zone" and cy > ZONE_2_Y:
                    track_history[track_id]["end_time"] = current_time
                    track_history[track_id]["status"] = "past_zone"
                    
                    start_time = track_history[track_id]["start_time"]
                    end_time = track_history[track_id]["end_time"]
                    time_diff = end_time - start_time
                    
                    if time_diff > 0.01: # Filter out instant glitches
                        speed = (DISTANCE_METERS / time_diff) * 3.6
                        print(f"🚀 DEBUG: Vehicle {track_id} EXITED Zone 2. Time: {time_diff:.2f}s, Speed: {speed:.2f} km/h")
                    else:
                        print(f"⚠️ DEBUG: Vehicle {track_id} crossed too fast (glitch?). Time diff: {time_diff}")

                # --- 3. Violation Detection ---
                violation_type = None
                
                # Check Speed Violation
                if speed > SPEED_LIMIT:
                    violation_type = "OVERSPEEDING"
                
                # Check Red Light Violation
                # If signal is RED and vehicle crosses the STOP line (ZONE_2)
                elif is_red_light and (ZONE_2_Y - 15 < cy < ZONE_2_Y + 15):
                     violation_type = "RED LIGHT JUMP"
                     # If speed wasn't calculated yet (because it's not fully 'past' the zone), calculate it now
                     if speed == 0 and track_id in track_history and track_history[track_id]["start_time"] > 0:
                        start_time = track_history[track_id]["start_time"]
                        time_diff = current_time - start_time
                        if time_diff == 0:
                            time_diff = 0.04 # Assume at least 1 frame (approx 25fps)
                        
                        if time_diff > 0:
                            speed = (DISTANCE_METERS / time_diff) * 3.6

                # --- 4. Logging & OCR ---
                if violation_type and track_id not in logged_ids:
                    violation_count += 1
                    logged_ids.add(track_id)
                    
                    # Crop Vehicle for OCR
                    vehicle_crop = frame[y1:y2, x1:x2]
                    plate_text = get_license_plate_text(vehicle_crop)
                    
                    # DEBUG: Save the plate crop
                    debug_plate_dir = "debug_plates"
                    os.makedirs(debug_plate_dir, exist_ok=True)
                    cv2.imwrite(f"{debug_plate_dir}/plate_{track_id}_{violation_count}.jpg", vehicle_crop)
                    
                    print(f"🚨 VIOLATION: {violation_type} | ID: {track_id} | Speed: {speed:.2f} km/h | Plate: {plate_text}")
                    
                    # Save Evidence
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    img_name = f"violation_{violation_count}_{ts}.jpg"
                    img_path = os.path.join(VIOLATIONS_IMG_DIR, img_name)
                    cv2.imwrite(img_path, frame)
                    
                    # Log to CSV
                    with open(CSV_FILE, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            ts, violation_count, violation_type, plate_text,
                            f"{speed:.2f}", img_path, "N/A" # Video clip logic removed for performance
                        ])
                    
                    # Visual Alert on Frame
                    cv2.putText(frame, f"VIOLATION: {violation_type}", (50, 300), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

                # Display Speed if available
                if speed > 0:
                     cv2.putText(frame, f"{speed:.1f} km/h", (x1, y2 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Traffic Violation System", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
