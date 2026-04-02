"""
main.py — Smart CCTV System Entry Point (v2)
=============================================
Changes from v1:
  - Replaced print() with Python logging (console + rotating file).
  - Learning status report on startup shows transit model confidence.
  - Graceful shutdown on SIGINT / SIGTERM.
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

EVENT_BUS: queue.Queue[DetectionEvent] = queue.Queue()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "").strip()
RTSP_URL           = os.getenv("RTSP_URL", "").strip()
FRAME_STRIDE       = int(os.getenv("FRAME_STRIDE", "6"))

with open("camera_map.json", "r") as f:
    CAMERA_MAP = json.load(f)

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
    model = YOLO("yolo26n.pt")
    extractor = AppearanceExtractor(mode="heuristic")

    try:
        results = model.predict(
            source=stream_url,
            stream=True,
            vid_stride=FRAME_STRIDE,
            classes=[0],
            verbose=False,
        )
        for result in results:
            frame = result.orig_img
            for box in result.boxes:
                coords = box.xyxy[0].cpu().numpy()
                appearance_vector = extractor.get_vector(frame, coords)
                if appearance_vector is None:
                    continue
                EVENT_BUS.put(DetectionEvent(
                    camera_id=camera_id,
                    timestamp=time.time(),
                    bbox=coords.tolist(),
                    appearance_vector=appearance_vector,
                ))
            annotated = result.plot()
            cv2.imshow(f"Feed: {camera_id}", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as exc:
        logger.exception("Camera worker %s crashed: %s", camera_id, exc)
    finally:
        cv2.destroyAllWindows()
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
    logger.info("Smart CCTV System v2 starting.")
    _print_learning_status()

    if not RTSP_URL:
        logger.error("RTSP_URL not set in .env. Example: RTSP_URL=rtsp://192.168.1.5:8080/video")
        sys.exit(1)

    threading.Thread(target=engine_worker, daemon=True, name="engine").start()
    threading.Thread(target=camera_worker, args=("cam_01_gate", 0), daemon=True, name="cam_01").start()
    threading.Thread(target=camera_worker, args=("cam_02_corridor", RTSP_URL), daemon=True, name="cam_02").start()

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