# cheating_detection

# 👁️ Cheating Detection System  
### Real-Time AI-Based Exam Monitoring using Computer Vision

---

## 🚀 Overview

The **Cheating Detection System** is a real-time computer vision application designed to monitor exam environments and detect suspicious behavior using AI.

It combines multiple intelligent modules such as **gaze tracking, head pose estimation, posture analysis, and object detection** to compute a **cheating risk score (0–100)** and display it on a live dashboard.

This project was built as a **science fair project** to demonstrate how AI can improve exam integrity in a non-invasive way.

---

## 🎯 Features

### 🔍 Detection Modules
- 👁️ **Gaze Tracking**
  - Detects eye direction (left, right, up, center)
  - Flags prolonged gaze deviation

- 🧠 **Head Pose Estimation**
  - Calculates pitch, yaw, and roll
  - Detects head turning

- 📱 **Object Detection**
  - Detects prohibited items:
    - Mobile phones
    - Books

- 🧍 **Posture Analysis**
  - Detects:
    - Leaning out of frame
    - Looking down for extended periods

---

### ⚡ Intelligence System
- 🎯 Risk Score (0–100)
- Weighted scoring system
- Time-based smoothing
- Cooldown mechanism to reduce repeated alerts

---

### 📊 Live Dashboard
- Real-time webcam stream
- Live risk score display
- Alert notifications
- Event timeline
- Auto-refresh (no manual reload required)

---

### 🗂️ Logging
- Saves:
  - Timestamp
  - Event type
  - Risk score
- CSV logging (`logs/events.csv`)
- Snapshot capture for high-risk events (`snapshots/`)

---

## 🏗️ Project Structure
cheating_detection_system/
│
├── main.py
├── config.py
├── shared_state.py
├── requirements.txt
│
├── modules/
│ ├── gaze_tracking.py
│ ├── head_pose.py
│ ├── object_detection.py
│ ├── posture_analysis.py
│ ├── cheating_logic.py
│ ├── logger.py
│ └── utils.py
│
├── dashboard/
│ ├── app.py
│ ├── templates/
│ │ └── index.html
│ └── static/
│ ├── style.css
│ └── script.js
│
├── logs/
│ └── events.csv
│
└── snapshots/

---

## ⚙️ Tech Stack

- Python  
- OpenCV  
- MediaPipe  
- PyTorch (YOLOv5)  
- Flask  
- NumPy  

---

## 🛠️ Setup

### 1. Clone Repository

### 2. Create Virtual Environment

### 3. Install Dependencies

### 4. (Optional) Preload YOLO Model

### 5. Run
(Single Command)
python main.py
- Starts detection system  
- Launches dashboard at: http://localhost:5000

---

### ⏹️ Stop
Press **Q** in the OpenCV window.

---

## 🎛️ Configuration

All settings can be modified in `config.py`:

- `CAMERA_INDEX`
- `GAZE_ALERT_SECONDS`
- `HEAD_YAW_THRESHOLD`
- `YOLO_CONF_THRESHOLD`
- `COOLDOWN_SECONDS`
- `SMOOTHING_WINDOW`
- `RISK_HIGH_THRESHOLD`

---

## 📊 Risk Score Breakdown

| Signal            | Max Score |
|------------------|----------|
| Object Detection | 30       |
| Gaze Deviation   | 25       |
| Head Turn        | 25       |
| Posture          | 20       |
| **Total**        | **100**  |

---

## 🧠 How It Works

1. Webcam feed is captured using OpenCV  
2. Each frame is processed through:
   - Face Mesh (gaze tracking)
   - Pose model (posture analysis)
   - Head pose estimation
   - YOLO object detection  
3. Outputs are combined into a risk score  
4. Alerts are generated based on thresholds  
5. Results are displayed and logged  

---

## 📸 Use Cases

- Online exam monitoring  
- School demonstrations  
- AI/ML portfolio projects  
- Research prototypes  

---

## ⚠️ Limitations

- Sensitive to lighting conditions  
- May produce false positives  
- Performance depends on hardware  
- Not a replacement for human invigilation  

---

## 🔮 Future Improvements

- AI-based anomaly detection (deep learning)
- Multi-student monitoring
- Cloud-based dashboard
- Advanced analytics
- Behavioral profiling

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to fork and submit pull requests.

---

## 📜 License

For educational and research purposes.

---

## ⭐ Acknowledgment

Built as a student project exploring real-world applications of AI in education.
