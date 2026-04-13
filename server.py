import json
import asyncio
import time
import math
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import List
import uvicorn

import database

# --- Initialize App & DB ---
database.init_db()
app = FastAPI(title="Guardly Local Server")

# --- MJPEG STREAMING MEMORY ---
LATEST_FRAMES = {}

# --- WebSocket Connection Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast_live_event(self, event_type: str, data: dict):
        message = json.dumps({"type": event_type, "data": data})
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"Failed to send to a websocket client: {e}")

manager = ConnectionManager()

# --- REST Endpoints ---

class AcknowledgeRequest(BaseModel):
    incident_id: int

@app.get("/api/incidents/active")
async def get_active_incidents():
    try:
        incidents = database.get_active_incidents()
        return {"status": "success", "data": incidents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/incidents/acknowledge")
async def acknowledge_incident(req: AcknowledgeRequest):
    try:
        database.acknowledge_incident(req.incident_id)
        await manager.broadcast_live_event("incident_cleared", {"incident_id": req.incident_id})
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cameras")
async def get_camera_layout():
    if not os.path.exists("camera_map.json"):
        return {"cameras": {}, "rooms": [], "transit_model": {}}
    
    try:
        with open("camera_map.json", "r") as f:
            content = f.read().strip()
            if not content:
                return {"cameras": {}, "rooms": [], "transit_model": {}}
            return json.loads(content)
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error loading camera_map.json: {e}")
        return {"cameras": {}, "rooms": [], "transit_model": {}}

# --- DYNAMIC SPATIAL AWARENESS ENDPOINT ---
@app.post("/api/cameras/save")
async def save_camera_layout(request: Request):
    """Takes the exact layout and links from the UI and saves them."""
    try:
        new_data = await request.json()
        ui_cameras = new_data.get("cameras", {})
        
        # Merge with existing file to keep transit_model safe
        existing_data = {"transit_model": {}, "cameras": {}, "rooms": []}
        if os.path.exists("camera_map.json"):
            try:
                with open("camera_map.json", "r") as f:
                    content = f.read().strip()
                    if content:
                        existing_data = json.loads(content)
            except json.JSONDecodeError:
                print("Warning: camera_map.json was corrupted, overwriting with fresh data")
                
        existing_data["cameras"] = ui_cameras
        existing_data["rooms"] = new_data.get("rooms", [])
        
        with open("camera_map.json", "w") as f:
            json.dump(existing_data, f, indent=4)
            
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/api/internal/push")
async def push_to_ui(payload: dict):
    await manager.broadcast_live_event(payload["type"], payload["data"])
    return {"status": "sent"}

# --- MJPEG VIDEO STREAMING ENDPOINTS ---

@app.post("/api/internal/frame/{camera_id}")
async def receive_frame(camera_id: str, request: Request):
    LATEST_FRAMES[camera_id] = await request.body()
    return {"status": "ok"}

def generate_mjpeg_stream(camera_id: str):
    while True:
        frame = LATEST_FRAMES.get(camera_id)
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)

@app.get("/api/stream/{camera_id}")
async def video_feed(camera_id: str):
    return StreamingResponse(
        generate_mjpeg_stream(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# --- WebSocket & Static Routing ---

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_guard_dashboard():
    return FileResponse("static/guard.html")

if __name__ == "__main__":
    print("Starting Guardly Web Server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)