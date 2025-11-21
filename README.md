# Smart Traffic Violation Detection System

An AI-powered system to detect traffic violations including Red Light Jumping and Overspeeding using YOLOv8 and OpenCV.

## 🚀 Features
- **Vehicle Detection & Tracking**: Uses YOLOv8 to detect and track vehicles (Car, Bus, Truck, Motorbike) with unique IDs.
- **Speed Estimation**: Calculates speed using virtual lines and pixel-to-meter mapping.
- **Red Light Violation**: Detects vehicles crossing the stop line when the signal is Red.
- **License Plate Recognition**: Extracts license plate numbers using YOLOv8 (optional) and EasyOCR.
- **Evidence Logging**: Saves violation snapshots and logs details to a CSV file.

## 📂 Directory Structure
```
smart_traffic_violation_system/
├── combined_violation_system.py  # Main script
├── yolov8n.pt                    # Vehicle detection model
├── license_plate_detector.pt     # (Optional) Plate detection model
├── red_light_violation.mp4       # Sample video for testing
├── violations/                   # Saved violation images
├── violations_videos/            # Saved violation clips (optional)
├── violations_log.csv            # Log file
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🛠️ Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Ankita-Verma-06/smart_traffic_violation_system.git
   cd smart_traffic_violation_system
   ```

2. **Install Dependencies**:
   ```bash
   pip install ultralytics opencv-python easyocr numpy
   ```

3. **Download Models**:
   - The script automatically downloads `yolov8n.pt` (COCO model) on first run.
   - **Recommended**: For better license plate detection, train or download a custom YOLOv8 plate model and place it in the root directory as `license_plate_detector.pt`. Update the path in the script if named differently.

## 🚦 Usage

### 1. Test Mode (Video File)
By default, the system runs in **Test Mode** using `red_light_violation.mp4`.
Run the script:
```bash
python combined_violation_system.py
```

### 2. Live Camera Mode
To use a live IP camera:
1. Open `combined_violation_system.py`.
2. Set `TEST_MODE = False`.
3. Update `IP_CAMERA_URL` with your camera's stream URL.

## ⚙️ Configuration
You can tweak the following variables in `combined_violation_system.py` to match your camera setup:
- `ZONE_1_Y` & `ZONE_2_Y`: Y-coordinates for the entry and exit lines.
- `DISTANCE_METERS`: **Crucial!** The real-world distance between the two lines. Measure this on the road for accurate speed.
- `SPEED_LIMIT`: Speed limit in km/h.

## 🔍 Troubleshooting
- **"Speed not available"**: Ensure the vehicle crosses *both* lines. Adjust `ZONE_1_Y` and `ZONE_2_Y` if the camera angle is different.
- **"License plate not available"**: OCR is difficult on low-res video. Use a high-resolution camera or a dedicated License Plate Recognition (LPR) model for best results.
# Ai-based-speed-and-traffic-violation-detection-
