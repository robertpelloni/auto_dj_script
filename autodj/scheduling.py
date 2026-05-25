import time
import threading
import json
import os
from datetime import datetime, timedelta

class MixScheduler:
    """
    Handles timed and event-driven tasks for the Auto DJ engine.
    Allows for scheduled starts, tempo ramps, and automated queue adjustments.
    """
    def __init__(self, gui_status=None):
        self.gui_status = gui_status
        self.events = []
        self.running = False
        self._thread = None
        self.log_file = "logs/scheduler.json"
        os.makedirs("logs", exist_ok=True)

    def add_event(self, timestamp, action, params=None):
        """Adds a scheduled event."""
        event = {
            "id": len(self.events),
            "time": timestamp,  # ISO format
            "action": action,
            "params": params or {},
            "status": "pending"
        }
        self.events.append(event)
        self._save_events()
        return event

    def start(self):
        if not self.running:
            self.running = True
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self):
        self.running = False

    def _run_loop(self):
        while self.running:
            now = datetime.now()
            for event in self.events:
                if event["status"] == "pending":
                    try:
                        # Handle Z and other timezone offsets
                        ts = event["time"].replace('Z', '+00:00')
                        event_time = datetime.fromisoformat(ts)
                        if event_time.tzinfo is None:
                            if now >= event_time:
                                self._execute_event(event)
                        else:
                            if datetime.now(event_time.tzinfo) >= event_time:
                                self._execute_event(event)
                    except:
                        pass
            time.sleep(1)

    def _execute_event(self, event):
        event["status"] = "executing"
        print(f"[*] Scheduler: Executing {event['action']}...")

        try:
            if event["action"] == "set_bpm":
                if self.gui_status:
                    self.gui_status["live_params"]["target_bpm"] = float(event["params"].get("bpm", 128.0))

            elif event["action"] == "toggle_continuous":
                val = event["params"].get("enabled", True)
                if isinstance(val, str): val = val.lower() == 'true'
                if self.gui_status:
                    self.gui_status["live_params"]["continuous_mode"] = val

            elif event["action"] == "inject_track":
                if self.gui_status:
                    self.gui_status["playlist"].append(event["params"].get("track_file"))

            elif event["action"] == "set_preference":
                if self.gui_status:
                    if "energy_bias" in event["params"]:
                        self.gui_status["live_params"]["energy_bias"] = float(event["params"]["energy_bias"])
                    if "genre_preference" in event["params"]:
                        self.gui_status["live_params"]["genre_preference"] = event["params"]["genre_preference"]

            elif event["action"] == "clear_queue":
                if self.gui_status:
                    self.gui_status["playlist"] = []
            elif event["action"] == "trigger_replenish":
                if self.gui_status:
                    # SmartReplenishTool will pick this up on next track start,
                    # but we can force it here by calling it manually if we had an instance.
                    # For now, we ensure continuous_mode is ON.
                    self.gui_status["live_params"]["continuous_mode"] = True


            event["status"] = "completed"
        except Exception as e:
            event["status"] = f"failed: {str(e)}"

        self._save_events()

    def _save_events(self):
        try:
            with open(self.log_file, "w") as f:
                json.dump(self.events, f, indent=2)
        except:
            pass

    def get_events(self):
        return self.events

scheduler = None

def get_scheduler(gui_status=None):
    global scheduler
    if scheduler is None:
        scheduler = MixScheduler(gui_status)
    return scheduler
