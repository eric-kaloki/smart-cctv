"""
main.py — Smart CCTV System Entry Point (v4)
=============================================
Added MJPEG Streaming Bridge, OOM Queue Protections, 
and Dynamic Spatial Camera Loading.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import queue
import signal
import sys
import threading
import time

import cv2
import requests
from dotenv import load_dotenv
from ultralytics import YOLO

from correlation_engine import CorrelationEngine
from feature_extractor import AppearanceExtractor
from models import DetectionEvent


def _configure_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s", "%H:%M:%S"))

    file_handler = logging.handlers.RotatingFileHandler(
        "smart_cctv.log", maxBytes=2 * 1024 * 1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)-8s] %(name)s — %(message)s")
    )

    root.addHandler(console)
    root.addHandler(file_handler)


_configure_logging()
logger = logging.getLogger("main")

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
load_dotenv()

# Maxsize prevents Out-Of-Memory (OOM) crashes during heavy traffic
EVENT_BUS: queue.Queue[DetectionEvent] = queue.Queue(maxsize=200)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()
RTSP_URL           = os.getenv("RTSP_URL", "").strip()
FRAME_STRIDE       = int(os.getenv("FRAME_STRIDE", "6"))

def load_camera_map():
    """Helper to load the map JSON dynamically."""
    if not os.path.exists("camera_map.json"):
        return {"cameras": {}, "rooms": [], "transit_model": {}}
        
    try:
        with open("camera_map.json", "r") as f:
            content = f.read().strip()
            if not content:
                return {"cameras": {}, "rooms": [], "transit_model": {}}
            return json.loads(content)
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Error loading camera_map.json: {e}")
        return {"cameras": {}, "rooms": [], "transit_model": {}}

CAMERA_MAP = load_camera_map()

correlation_engine = CorrelationEngine(
    camera_map=CAMERA_MAP,
    bot_token=TELEGRAM_BOT_TOKEN,
    chat_id=TELEGRAM_CHAT_ID,
    transit_db_path="transit_memory.db",
)


def engine_worker() -> None:
    logger.info("Correlation engine worker started.")
    while True:
        try:
            event = EVENT_BUS.get(timeout=1.0)
            correlation_engine.process_event(event)
            EVENT_BUS.task_done()
        except queue.Empty:
            continue
        except Exception as exc:
            logger.exception("Engine worker error: %s", exc)


def camera_worker(camera_id: str, stream_url) -> None:
    logger.info("Camera worker starting: %s", camera_id)
    model = YOLO("yolo26n.pt") # Ensure this model file exists
    extractor = AppearanceExtractor(mode="heuristic")

    try:
        results = model.predict(
            source=stream_url,
            stream=True,
            vid_stride=FRAME_STRIDE,
            classes=[0], # 0 = Person
            verbose=False,
        )
        for result in results:
            frame = result.orig_img
            for box in result.boxes:
                coords = box.xyxy[0].cpu().numpy()
                appearance_vector = extractor.get_vector(frame, coords)
                if appearance_vector is None:
                    continue
                
                # Push to event bus safely
                try:
                    EVENT_BUS.put_nowait(DetectionEvent(
                        camera_id=camera_id,
                        timestamp=time.time(),
                        bbox=coords.tolist(),
                        appearance_vector=appearance_vector,
                    ))
                except queue.Full:
                    pass # Drop event to keep system alive if busy
            
            # --- MJPEG STREAMING BRIDGE ---
            annotated = result.plot()
            
            # Compress the annotated frame into JPEG format
            ret, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if ret:
                try:
                    # Push the frame directly to the local FastAPI server in memory
                    requests.post(
                        f"http://127.0.0.1:8000/api/internal/frame/{camera_id}",
                        data=buffer.tobytes(),
                        headers={'Content-Type': 'application/octet-stream'},
                        timeout=0.1 # Fail fast so camera loop never lags
                    )
                except Exception:
                    pass # Ignore if web server is offline

    except Exception as exc:
        logger.exception("Camera worker %s crashed: %s", camera_id, exc)
    finally:
        logger.info("Camera worker stopped: %s.", camera_id)


def _print_learning_status() -> None:
    status = correlation_engine._learner.learning_status()
    if not status:
        return
    logger.info("─── Transit Model Learning Status ───")
    for pair, hours in status.items():
        total = sum(s["samples"] for s in hours.values())
        confident = [h for h, s in hours.items() if s["confident"]]
        logger.info("  %s | observations: %d | confident hours: %s",
                    pair, total, confident or "none yet")
    logger.info("─────────────────────────────────────")


def main() -> None:
    logger.info("Smart CCTV System v4 starting (Dynamic Mode).")
    _print_learning_status()

    # Start the Correlation Engine thread
    threading.Thread(target=engine_worker, daemon=True, name="engine").start()
    
# --- DYNAMIC CAMERA LOADING ---
    cameras_to_load = CAMERA_MAP.get("cameras", {})
    
    if not cameras_to_load:
        logger.warning("No cameras found in camera_map.json! Please draw them in the UI and restart.")
    else:
        logger.info(f"Found {len(cameras_to_load)} cameras in layout. Spinning up workers...")
        
        for cam_id, cam_info in cameras_to_load.items():
            # Grab the stream_url from the JSON (the UI you just updated)
            json_url = cam_info.get("stream_url", "").strip()
            
            # If the user left it blank in the UI, fall back to laptop webcam (0)
            if not json_url:
                stream_target = 0
                logger.info(f"Camera {cam_id} has empty URL. Defaulting to Laptop Webcam (0).")
            # Convert numeric strings (like "0" or "1") to actual integers for OpenCV
            elif json_url.isdigit():
                stream_target = int(json_url)
                logger.info(f"Assigning Hardware Camera {stream_target} to {cam_id}")
            # Otherwise, use the HTTP/RTSP link provided
            else:
                stream_target = json_url
                logger.info(f"Assigning Stream {stream_target} to {cam_id}")

            # Spin up a worker for the camera
            threading.Thread(
                target=camera_worker, 
                args=(cam_id, stream_target), 
                daemon=True, 
                name=cam_id
            ).start()

    logger.info("All workers started. Press Ctrl+C to stop.")

    def _shutdown(signum, frame):
        logger.info("Shutting down cleanly.")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()