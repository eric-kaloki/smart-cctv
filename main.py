import os
import cv2
import json
import time
import threading
import queue
from dotenv import load_dotenv
from ultralytics import YOLO

from feature_extractor import AppearanceExtractor
from models import DetectionEvent
from correlation_engine import CorrelationEngine

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
load_dotenv()

EVENT_BUS = queue.Queue()

# Load Telegram Configs
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

with open("camera_map.json", "r") as f:
    CAMERA_MAP = json.load(f)

# Initialize the Brain
correlation_engine = CorrelationEngine(CAMERA_MAP, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

def engine_worker():
    print("🧠 Correlation Engine started. Listening for physical events...")
    while True:
        try:
            event = EVENT_BUS.get(timeout=1.0) 
            correlation_engine.process_event(event)
            EVENT_BUS.task_done()
        except queue.Empty:
            continue

def camera_worker(camera_id, stream_url):
    print(f"📹 Starting physical worker for {camera_id}...")
    model = YOLO("yolo26n.pt")
    extractor = AppearanceExtractor(mode="heuristic")
    
    stride_rate = 6 
    results = model.predict(source=stream_url, stream=True, vid_stride=stride_rate, classes=[0], verbose=False)
    
    for result in results:
        frame = result.orig_img
        
        for box in result.boxes:
            coords = box.xyxy[0].cpu().numpy()
            appearance_vector = extractor.get_vector(frame, coords)
            
            if appearance_vector:
                # Real time, real camera ID! No more faking.
                event = DetectionEvent(
                    camera_id=camera_id,
                    timestamp=time.time(),
                    bbox=coords.tolist(),
                    appearance_vector=appearance_vector
                )
                EVENT_BUS.put(event)
                
        annotated = result.plot()
        cv2.imshow(f"Live Feed: {camera_id}", annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    engine_thread = threading.Thread(target=engine_worker, daemon=True)
    engine_thread.start()

    # Camera 1: Your Laptop Webcam
    LAPTOP_CAM = 0
    
    # Camera 2: Your Phone's RTSP URL
    PHONE_CAM = os.getenv("RTSP_URL", "").strip()
    
    if not PHONE_CAM or PHONE_CAM == "0":
        print("\n❌ ERROR: You must set your phone's RTSP_URL in the .env file!")
        print("Example: RTSP_URL='rtsp://192.168.1.5:8080/video'\n")
    else:
        # Start TWO separate threads for the two physical cameras
        cam_thread_1 = threading.Thread(target=camera_worker, args=("cam_01_gate", LAPTOP_CAM), daemon=True)
        cam_thread_2 = threading.Thread(target=camera_worker, args=("cam_02_corridor", PHONE_CAM), daemon=True)
        
        cam_thread_1.start()
        cam_thread_2.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down system...")