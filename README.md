# Autonomous Car — Real-Time Lane Detection System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A lightweight, high-performance classical computer vision pipeline for real-time lane boundary detection and vehicle lateral error estimation on autonomous scale vehicles (e.g., 1:10 RC autonomous cars).

Designed for low latency and pure CPU execution without requiring deep learning frameworks, heavy neural network weights, or GPU acceleration.

---

## 📌 Features

- **Robust Classical Vision Pipeline**:
  - Grayscale conversion and Gaussian noise reduction
  - Canny edge detection with configurable thresholds
  - Trapezoidal Region of Interest (ROI) dynamic masking
  - Probabilistic Hough Transform (`cv2.HoughLinesP`) segment extraction
  - Slope-based line categorization (left vs. right boundary)
  - Robust least-squares polynomial line fitting
- **Autonomous Guidance Signals**:
  - Calculates lane midpoint at configurable lookahead distances
  - Computes lateral cross-track error (offset from image center)
  - Measures instantaneous and average FPS
- **Multi-Source Support**:
  - Live USB webcams / onboard CSI cameras
  - Prerecorded video files (`.mp4`, `.avi`, etc.)
  - Single test pictures
  - Network video streams (RTSP / HTTP)

---

## 📁 Repository Structure

```text
.
├── lane_detection/
│   ├── lane_detection.py      # Core lane detection pipeline implementation
│   └── README.md              # Detailed technical documentation & algorithm notes
├── videos/
│   └── test.mp4               # Sample test track driving footage
├── .gitignore                 # Standard git exclusions (venv, caches, etc.)
├── LICENSE                    # MIT License
├── README.md                  # Project overview & quickstart guide
└── requirements.txt           # Python package dependencies
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create and Activate a Virtual Environment (Recommended)

**On Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Run with the Sample Video

```bash
python lane_detection/lane_detection.py --source "videos/test.mp4"
```

To playback in real-time or loop continuously:
```bash
python lane_detection/lane_detection.py --source "videos/test.mp4" --realtime --loop
```

### 2. Run with Live Camera Feed

```bash
# Uses default webcam (index 0)
python lane_detection/lane_detection.py --source 0

# Uses external camera (index 1)
python lane_detection/lane_detection.py --source 1
```

### 3. Run with RTSP Network Stream

```bash
python lane_detection/lane_detection.py --source "rtsp://192.168.1.100:554/stream"
```

---

## ⚙️ Configuration & Tuning Options

| Argument | Default | Description |
|---|---|---|
| `--source` | *Required* | Camera index (`0`), video/image file path, or stream URL |
| `--width` | `640` | Video frame capture width |
| `--height` | `480` | Video frame capture height |
| `--canny-low` | `50` | Canny edge detector lower threshold |
| `--canny-high` | `150` | Canny edge detector higher threshold |
| `--hough-threshold` | `25` | Hough accumulator threshold |
| `--min-line-length` | `25` | Minimum line segment length in pixels |
| `--max-line-gap` | `60` | Maximum allowable gap to connect line segments |
| `--roi-top` | `0.55` | Top boundary of trapezoid ROI (fraction of height) |
| `--lookahead` | `0.75` | Height row where lane center & error are sampled |
| `--realtime` | `False` | Pace video playback to match original file FPS |
| `--loop` | `False` | Automatically restart video upon reaching the end |
| `--no-debug` | `False` | Suppress the Canny edge debugging window |

Press `q` or `ESC` while focused on any display window to exit cleanly.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
