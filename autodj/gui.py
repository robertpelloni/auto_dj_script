"""
Web-based GUI for the Auto DJ Script using FastAPI and WebSockets (7.6.0).
7.6.0: The Visual Era (Spectral Terrain 3D).
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
from .monitoring import monitor
from .plugins import PluginRegistry
from .scheduling import get_scheduler

app = FastAPI(title=f"Auto DJ v{__version__} Console")
templates = Jinja2Templates(directory="templates")

mixing_status = {
    "status": "Idle",
    "tracklist": [],
    "playlist": [],
    "active_tasks": {},
    "progress": 0,
    "version": __version__,
    "parallel_cores": os.cpu_count() or 1,
    "telemetry": {
        "cpu_usage": 0,
        "memory_usage": 0,
        "active_threads": 0,
        "disk_read": 0,
        "disk_write": 0,
        "net_sent": 0,
        "net_recv": 0,
        "midi_active": False,
        "midi_device": "None",
        "is_healthy": True,
        "is_throttled": False
    },
    "live_params": {
        "mastering_intensity": 0.5,
        "target_bpm": config.TARGET_BPM,
        "paused": False,
        "continuous_mode": False,
        "energy_bias": 0.5,
        "genre_preference": "Any"
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
            "archetypes": ArchetypeRegistry.get_all(),
            "sources": PluginRegistry.get_sources(),
            "outputs": PluginRegistry.get_outputs()
        }
    )

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(update_telemetry())
    get_scheduler(mixing_status).start()

async def update_telemetry():
    """Background task to poll system performance metrics."""
    psutil.cpu_percent()  # Initialize first call
    last_disk = psutil.disk_io_counters()
    last_net = psutil.net_io_counters()

    while True:
        try:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            mixing_status["telemetry"]["cpu_usage"] = cpu
            mixing_status["telemetry"]["memory_usage"] = mem
            mixing_status["telemetry"]["active_threads"] = len(psutil.Process().threads())

            # Disk & Net Throughput (v7.8.0)
            now_disk = psutil.disk_io_counters()
            now_net = psutil.net_io_counters()

            if last_disk and now_disk:
                mixing_status["telemetry"]["disk_read"] = (now_disk.read_bytes - last_disk.read_bytes) / config.HEALTH_CHECK_INTERVAL
                mixing_status["telemetry"]["disk_write"] = (now_disk.write_bytes - last_disk.write_bytes) / config.HEALTH_CHECK_INTERVAL

            if last_net and now_net:
                mixing_status["telemetry"]["net_sent"] = (now_net.bytes_sent - last_net.bytes_sent) / config.HEALTH_CHECK_INTERVAL
                mixing_status["telemetry"]["net_recv"] = (now_net.bytes_recv - last_net.bytes_recv) / config.HEALTH_CHECK_INTERVAL

            last_disk = now_disk
            last_net = now_net

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
    status_data["monitoring"] = monitor.get_status()
    # Populate available tracks if idle
    if mixing_status["status"] == "Idle":
        import glob
        status_data["available_tracks"] = [os.path.basename(f) for f in glob.glob(os.path.join(config.INPUT_FOLDER, "*")) if any(f.endswith(ext) for ext in config.SUPPORTED_EXTENSIONS)]

    # Update MIDI telemetry if handler exists
    from .midi import PluginRegistry
    midi_tool = PluginRegistry.get_tools().get("midi_hardware")
    if midi_tool and hasattr(midi_tool, 'handler') and midi_tool.handler:
        status_data["telemetry"]["midi_active"] = midi_tool.handler.running
        status_data["telemetry"]["midi_device"] = midi_tool.handler.port_name or "Auto-Detect"

    return JSONResponse(status_data)

@app.get("/midi/devices")
async def get_midi_devices():
    """Returns a list of available MIDI input devices."""
    import mido
    try:
        return {"devices": mido.get_input_names()}
    except Exception:
        return {"devices": []}

@app.post("/playlist/add")
async def playlist_add(filename: str = Form(...)):
    """Adds a track to the dynamic playlist."""
    mixing_status["playlist"].append(filename)
    return {"status": "Added", "playlist": mixing_status["playlist"]}

@app.post("/playlist/remove")
async def playlist_remove(index: int = Form(...)):
    """Removes a track from the dynamic playlist."""
    if 0 <= index < len(mixing_status["playlist"]):
        mixing_status["playlist"].pop(index)
    return {"status": "Removed", "playlist": mixing_status["playlist"]}

@app.post("/cluster/reset")
async def cluster_reset(node_id: str = Form(...)):
    """Resets failed states for a node."""
    cluster.reset_node(node_id)
    return {"status": "Reset", "node": node_id}

@app.post("/update_params")
async def update_params(
    mastering_intensity: float = Form(None),
    target_bpm: float = Form(None),
    paused: bool = Form(None),
    continuous_mode: bool = Form(None),
    energy_bias: float = Form(None),
    genre_preference: str = Form(None)
):
    """Real-time parameter adjustment endpoint."""
    if mastering_intensity is not None:
        mixing_status["live_params"]["mastering_intensity"] = mastering_intensity
    if target_bpm is not None:
        mixing_status["live_params"]["target_bpm"] = target_bpm
    if paused is not None:
        mixing_status["live_params"]["paused"] = paused
    if continuous_mode is not None:
        mixing_status["live_params"]["continuous_mode"] = continuous_mode
    if energy_bias is not None:
        mixing_status["live_params"]["energy_bias"] = energy_bias
    if genre_preference is not None:
        mixing_status["live_params"]["genre_preference"] = genre_preference
    return {"status": "Updated", "params": mixing_status["live_params"]}

@app.get("/scheduler/events")
async def scheduler_events():
    sched = get_scheduler(mixing_status)
    return {"events": [{"id": i, "time": e["time"].isoformat(), "action": e["action"], "params": e["params"], "status": e["status"]} for i, e in enumerate(sched.events)]}

@app.post("/scheduler/add")
async def scheduler_add(timestamp: str = Form(...), action: str = Form(...), params: str = Form("{}")):
    import json
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        p = json.loads(params)
        get_scheduler(mixing_status).add_event(dt, action, p)
        return {"status": "Added"}
    except Exception as e:
        return JSONResponse({"status": "Error", "message": str(e)}, status_code=400)

@app.post("/feedback")
async def post_feedback(track_index: int = Form(...), rating: int = Form(...)):
    """Logs user feedback (1 for Up, -1 for Down) for a track."""
    import json
    from datetime import datetime
    feedback_file = "logs/feedback.json"
    os.makedirs("logs", exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "track_index": track_index,
        "rating": rating
    }

    if 0 <= track_index < len(mixing_status["tracklist"]):
        entry["track"] = mixing_status["tracklist"][track_index]["file"]
        entry["genre"] = mixing_status["tracklist"][track_index]["genre"]
        entry["key"] = mixing_status["tracklist"][track_index]["key"]

    try:
        data = []
        if os.path.exists(feedback_file):
            with open(feedback_file, "r") as f:
                data = json.load(f)
        data.append(entry)
        with open(feedback_file, "w") as f:
            json.dump(data, f, indent=4)
        return {"status": "Feedback Recorded"}
    except Exception as e:
        return JSONResponse({"status": "Error", "message": str(e)}, status_code=500)

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
    source_plugin: str = Form("local_folder"),
    output_plugin: str = Form("local_file"),
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
            self.source_plugin = source_plugin
            self.output_plugin = output_plugin
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
