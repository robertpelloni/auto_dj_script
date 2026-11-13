"""Timed and Event-Driven Automation Scheduler for the Auto DJ Engine."""

import json
import os
import threading
import time
from datetime import datetime


class MixScheduler:
    """Handles timed and event-driven tasks for the Auto DJ engine.

    Supports scheduled BPM changes, continuous-mode toggles, track injections,
    energy/genre preference shifts, queue clearing, and immediate smart replenishment.
    """

    def __init__(self, gui_status=None):
        self.gui_status = gui_status
        self.events = []
        self._next_id = 0
        self.running = False
        self._thread = None
        self.log_file = "logs/scheduler.json"
        os.makedirs("logs", exist_ok=True)

    @staticmethod
    def _coerce_datetime(timestamp) -> datetime:
        if isinstance(timestamp, datetime):
            return timestamp
        ts = str(timestamp).strip().replace("Z", "+00:00")
        return datetime.fromisoformat(ts)

    def add_event(self, timestamp, action, params=None):
        """Adds a scheduled event (`timestamp` may be `datetime` or ISO-8601 `str`)."""
        dt = self._coerce_datetime(timestamp)
        event = {
            "id": self._next_id,
            "time": dt,
            "action": action,
            "params": params or {},
            "status": "pending",
        }
        self._next_id += 1
        self.events.append(event)
        self._save_events()
        return event

    def delete_event(self, event_id: int) -> bool:
        """Removes an event by its `id` or list index."""
        for idx, ev in enumerate(self.events):
            if ev.get("id") == event_id:
                self.events.pop(idx)
                self._save_events()
                return True
        if 0 <= event_id < len(self.events):
            self.events.pop(event_id)
            self._save_events()
            return True
        return False

    def clear_completed(self) -> int:
        before = len(self.events)
        self.events = [e for e in self.events if e.get("status") == "pending"]
        self._save_events()
        return before - len(self.events)

    def start(self):
        if not self.running:
            self.running = True
            self._thread = threading.Thread(
                target=self._run_loop, name="AutoDJ-Scheduler", daemon=True
            )
            self._thread.start()

    def stop(self):
        self.running = False

    def _run_loop(self):
        while self.running:
            now_naive = datetime.now()
            for event in list(self.events):
                if event.get("status") == "pending":
                    try:
                        event_time = self._coerce_datetime(event["time"])
                        if event_time.tzinfo is None:
                            due = now_naive >= event_time
                        else:
                            due = datetime.now(event_time.tzinfo) >= event_time
                        if due:
                            self._execute_event(event)
                    except Exception as exc:
                        event["status"] = f"failed: {exc}"
            time.sleep(1)

    def _execute_event(self, event):
        event["status"] = "executing"
        print(f"[*] Scheduler: Executing {event['action']}...")

        try:
            action = event["action"]
            params = event.get("params") or {}

            if action == "set_bpm":
                if self.gui_status:
                    self.gui_status["live_params"]["target_bpm"] = float(
                        params.get("bpm", 145.0)
                    )

            elif action == "toggle_continuous":
                val = params.get("enabled", True)
                if isinstance(val, str):
                    val = val.lower() == "true"
                if self.gui_status:
                    self.gui_status["live_params"]["continuous_mode"] = bool(val)

            elif action == "inject_track":
                track_file = params.get("track_file")
                if self.gui_status and track_file:
                    self.gui_status["playlist"].append(str(track_file))

            elif action == "set_preference":
                if self.gui_status:
                    if "energy_bias" in params:
                        self.gui_status["live_params"]["energy_bias"] = float(
                            params["energy_bias"]
                        )
                    if "genre_preference" in params:
                        self.gui_status["live_params"]["genre_preference"] = str(
                            params["genre_preference"]
                        )

            elif action == "clear_queue":
                if self.gui_status:
                    self.gui_status["playlist"] = []

            elif action == "trigger_replenish":
                if self.gui_status:
                    self.gui_status["live_params"]["continuous_mode"] = True
                    try:
                        from .core import SmartReplenishTool

                        SmartReplenishTool().replenish_once(self.gui_status)
                    except Exception:
                        pass

            event["status"] = "completed"
        except Exception as e:
            event["status"] = f"failed: {str(e)}"

        self._save_events()

    def _save_events(self):
        try:
            serializable = [
                {
                    **ev,
                    "time": ev["time"].isoformat()
                    if hasattr(ev["time"], "isoformat")
                    else str(ev["time"]),
                }
                for ev in self.events
            ]
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(serializable, f, indent=2)
        except Exception:
            pass

    def get_events(self):
        return self.events


scheduler = None


def get_scheduler(gui_status=None):
    global scheduler
    if scheduler is None:
        scheduler = MixScheduler(gui_status)
    elif gui_status is not None and scheduler.gui_status is None:
        scheduler.gui_status = gui_status
    return scheduler
