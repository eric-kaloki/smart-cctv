# Smart CCTV: Physical Intrusion & Tracking System

A real-time surveillance intelligence system that detects, tracks, and correlates person movements across multiple camera feeds using YOLO computer vision and a heuristic correlation engine.

## 🚀 Overview

This system goes beyond simple motion detection. It builds **"Trails"** of individuals as they move through a physical space, correlating their appearance and transit times between known camera locations. When suspicious movement patterns (e.g., traversing multiple secured zones) are detected, it triggers automated alerts.

## ✨ Key Features

- **Real-Time Person Detection**: Powered by yolo26n for high-accuracy person identification.
- **Multi-Camera Correlation**:
  - **Physical Adjacency**: Uses a camera map to ensure transitions only occur between connected zones.
  - **Transit Time Logic**: Validates if the time taken to move between cameras is physically possible.
  - **Appearance Matching**: Heuristic-based matching using aspect ratios and RGB color profiles (torso and legs).
- **Intelligent Alerting**: Telegram integration for instant notifications when high-risk "Incidents" are identified.
- **Dual Feed Support**: Simultaneously processes local webcam and remote RTSP (phone) streams.

## 🛠️ Project Structure

- `main.py`: The entry point. Manages camera workers and the central correlation brain.
- `correlation_engine.py`: Contains the logic for matching events to existing trails and detecting incidents.
- `feature_extractor.py`: Extracts visual "fingerprints" (vectors) from detected persons.
- `models.py`: Data structures for `DetectionEvent` and `Trail`.
- `camera_map.json`: Configures the physical layout of your surveillance network.
- `yolo26n.pt`: Optimized YOLO detection model.

## 🚦 Getting Started

### Prerequisites

- Python 3.9+
- OpenCV
- PyTorch
- Ultralytics (yolo26n)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd cctv
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   Create a `.env` file in the root directory:
   ```env
   # Your phone's RTSP stream (e.g., from IP Webcam app)
   RTSP_URL='rtsp://192.168.1.5:8080/video'

   # Telegram Bot Configuration
   TELEGRAM_BOT_TOKEN='your_bot_token'
   TELEGRAM_CHAT_ID='your_chat_id'
   ```

4. **Define your Camera Map**:
   Edit `camera_map.json` to reflect your physical environment:
   ```json
   {
       "cameras": {
           "cam_01_gate": { "adjacent_to": ["cam_02_corridor"] },
           "cam_02_corridor": { "adjacent_to": ["cam_01_gate", "cam_03_backdoor"] }
       }
   }
   ```

### Running the System

```bash
python main.py
```

## 🧠 How it Works

1. **Detection**: Each camera worker runs YOLO on its stream.
2. **Extraction**: When a person is found, their appearance (torso/legs color + shape) is extracted.
3. **Correlation**: The "Brain" (Correlation Engine) checks if this person matches any active trails based on:
   - "Does their appearance match?"
   - "Were they just seen on an adjacent camera?"
   - "Did the travel time make physical sense?"
4. **Alerting**: If a trail visits multiple unique cameras, an incident is flagged and pushed to Telegram.
