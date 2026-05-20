"""
Web-based GUI for the Auto DJ Script using FastAPI and WebSockets.
v5.5.0: Dynamic Status Polling and Interactive Tracklist.
"""
from fastapi import FastAPI, Request, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, JSONResponse, Response
import uvicorn
import os
import asyncio
import config
from .core import compile_master_set
from .version import __version__
from .dsp import ArchetypeRegistry

app = FastAPI(title=f"Auto DJ v{__version__} Console")
templates = Jinja2Templates(directory="templates")

mixing_status = {
    "status": "Idle",
    "tracklist": [],
    "progress": 0
}

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

@app.get("/favicon.ico")
async def favicon():
    svg = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><circle cx="16" cy="16" r="14" fill="#a020f0"/><text x="16" y="22" text-anchor="middle" fill="#fff" font-size="16" font-weight="bold" font-family="sans-serif">DJ</text></svg>'
    return Response(content=svg, media_type="image/svg+xml")

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "version": __version__,
            "config": config,
            "status": mixing_status,
            "archetypes": ArchetypeRegistry.get_all()
        }
    )

@app.get("/status")
async def get_status():
    return JSONResponse(mixing_status)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await websocket.accept()
        manager.active_connections.append(websocket)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if websocket in manager.active_connections:
            manager.active_connections.remove(websocket)

@app.post("/start")
async def start_mixing(background_tasks: BackgroundTasks, bpm: float = Form(...), end_bpm: float = Form(None), reorder: bool = Form(False), archetype: str = Form("auto"), mastering_intensity: float = Form(0.5), dynamic_transitions: bool = Form(True), genre_aware_mastering: bool = Form(True)):
    class Args:
        def __init__(self):
            self.input = config.INPUT_FOLDER
            self.output = config.OUTPUT_FILE
            self.bpm = bpm
            self.end_bpm = end_bpm
            self.dbfs = config.TARGET_DBFS
            self.lowpass = config.LOWPASS_CUTOFF
            self.highpass = config.HIGHPASS_CUTOFF
            self.transition_bars = config.TRANSITION_BARS
            self.beats_per_bar = config.BEATS_PER_BAR
            self.dry_run = False
            self.reorder = reorder
            self.archetype = archetype
            self.mastering_intensity = mastering_intensity
            self.dynamic_transitions = dynamic_transitions
            self.genre_aware_mastering = genre_aware_mastering

    mixing_status["status"] = "Preparing Engine..."
    mixing_status["progress"] = 0
    mixing_status["tracklist"] = []

    background_tasks.add_task(compile_master_set, Args(), status_obj=mixing_status)
    return {"message": "Engine ignited."}

@app.get("/download")
async def download_master():
    if os.path.exists(config.OUTPUT_FILE):
        return FileResponse(config.OUTPUT_FILE, filename=os.path.basename(config.OUTPUT_FILE))
    return JSONResponse({"error": "Mix not found."}, status_code=404)

def run_gui(host="0.0.0.0", port=8000):
    print(f"[*] Auto DJ v{__version__} Console launching...")
    uvicorn.run(app, host=host, port=port)
