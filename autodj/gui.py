"""
Web-based GUI for the Auto DJ Script using FastAPI and WebSockets (7.3.0).
7.3.0: Integration Bridge & Staging Era.
"""
from fastapi import FastAPI, Request, Form, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, JSONResponse, Response
import uvicorn
import os
import asyncio
import psutil
import config
from .core import compile_master_set
from .version import __version__
from .dsp import ArchetypeRegistry
from .cluster import cluster

app = FastAPI(title=f"Auto DJ v{__version__} Console")
templates = Jinja2Templates(directory="templates")

mixing_status = {
    "status": "Idle",
    "tracklist": [],
    "progress": 0,
    "version": __version__,
    "parallel_cores": os.cpu_count() or 1,
    "telemetry": {
        "cpu_usage": 0,
        "memory_usage": 0,
        "active_threads": 0,
        "is_healthy": True,
        "is_throttled": False
    },
    "live_params": {
        "mastering_intensity": 0.5,
        "target_bpm": config.TARGET_BPM,
        "paused": False
    }
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

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(update_telemetry())

async def update_telemetry():
    """Background task to poll system performance metrics."""
    psutil.cpu_percent()  # Initialize first call
    while True:
        try:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            mixing_status["telemetry"]["cpu_usage"] = cpu
            mixing_status["telemetry"]["memory_usage"] = mem
            mixing_status["telemetry"]["active_threads"] = len(psutil.Process().threads())

            # Health Logic (v7.2.0)
            is_healthy = (cpu < config.MAX_CPU_LOAD and mem < config.MAX_RAM_LOAD)
            mixing_status["telemetry"]["is_healthy"] = is_healthy
            mixing_status["telemetry"]["is_throttled"] = not is_healthy

        except Exception:
            pass
        await asyncio.sleep(config.HEALTH_CHECK_INTERVAL)

@app.get("/status")
async def get_status():
    status_data = dict(mixing_status)
    status_data["cluster"] = cluster.get_status()
    return JSONResponse(status_data)

@app.post("/update_params")
async def update_params(
    mastering_intensity: float = Form(None),
    target_bpm: float = Form(None),
    paused: bool = Form(None)
):
    """Real-time parameter adjustment endpoint."""
    if mastering_intensity is not None:
        mixing_status["live_params"]["mastering_intensity"] = mastering_intensity
    if target_bpm is not None:
        mixing_status["live_params"]["target_bpm"] = target_bpm
    if paused is not None:
        mixing_status["live_params"]["paused"] = paused
    return {"status": "Updated", "params": mixing_status["live_params"]}

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
async def start_mixing(
    background_tasks: BackgroundTasks,
    bpm: float = Form(...),
    end_bpm: float = Form(None),
    reorder: bool = Form(False),
    archetype: str = Form("auto"),
    mastering_intensity: float = Form(0.5),
    dynamic_transitions: bool = Form(True),
    genre_aware_mastering: bool = Form(True),
    dynamic_energy_mastering: bool = Form(True),
    adaptive_spectral_balancing: bool = Form(True),
    broadcast_mode: bool = Form(False),
    stream_url: str = Form(None)
):
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
            self.dynamic_energy_mastering = dynamic_energy_mastering
            self.adaptive_spectral_balancing = adaptive_spectral_balancing
            self.broadcast_mode = broadcast_mode
            self.stream_url = stream_url

    mixing_status["status"] = "Preparing Engine..."
    mixing_status["progress"] = 0
    mixing_status["tracklist"] = []
    mixing_status["live_params"]["target_bpm"] = bpm
    mixing_status["live_params"]["mastering_intensity"] = mastering_intensity

    background_tasks.add_task(compile_master_set, Args(), status_obj=mixing_status)
    return {"message": "Engine ignited."}

@app.post("/cluster/join")
async def cluster_join(node_id: str = Form(...), cores: int = Form(...)):
    """API endpoint for remote nodes to join the cluster."""
    cluster.register_node(node_id, cores)
    return {"status": "Accepted", "node": node_id}

@app.get("/download")
async def download_master():
    if os.path.exists(config.OUTPUT_FILE):
        return FileResponse(config.OUTPUT_FILE, filename=os.path.basename(config.OUTPUT_FILE))
    return JSONResponse({"error": "Mix not found."}, status_code=404)

def run_gui(host="0.0.0.0", port=8000):
    print(f"[*] Auto DJ v{__version__} Console launching...")
    uvicorn.run(app, host=host, port=port)
