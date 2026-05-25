"""
Autonomous Scaling Engine for Auto DJ (v8.12.0).
Analyzes system health and queue depth to optimize cluster worker counts.
"""
import psutil
import config
import os
from .monitoring import monitor

class AutoScaler:
    """
    Decides when to scale cluster resources up or down.
    """
    def __init__(self, cluster_instance):
        self.cluster = cluster_instance
        self.min_workers = 1
        self.max_workers = os.cpu_count() or 4
        self.scaling_cooldown = 10  # Seconds between scaling actions
        self.last_scale_time = 0
        self.decision_history = []

    def evaluate(self, status_obj=None):
        """
        Analyzes metrics and returns a scaling command.
        """
        import time
        now = time.time()
        if now - self.last_scale_time < self.scaling_cooldown:
            return None

        cpu_load = psutil.cpu_percent()
        ram_load = psutil.virtual_memory().percent

        # Determine Queue Depth (tasks waiting for processing)
        queue_depth = 0
        if status_obj:
            playlist = status_obj.get("playlist", [])
            # Remaining tracks to mix
            queue_depth = len(playlist)

        current_workers = self.cluster.get_worker_count()

        decision = None
        reason = ""

        # Scaling Logic
        if cpu_load > config.MAX_CPU_LOAD or ram_load > config.MAX_RAM_LOAD:
            if current_workers > self.min_workers:
                decision = "DOWN"
                reason = f"High system load (CPU: {cpu_load}%, RAM: {ram_load}%)"

        elif queue_depth > 2 and cpu_load < (config.MAX_CPU_LOAD - 20):
            if current_workers < self.max_workers:
                decision = "UP"
                reason = f"Heavy queue depth ({queue_depth} tracks) with spare capacity"

        if decision:
            self.last_scale_time = now
            self.decision_history.append({"time": now, "action": decision, "reason": reason})
            monitor.log_incident("INFO", "AutoScaler", f"Scaling {decision}: {reason}")
            return decision

        return None

    def get_scaling_metrics(self):
        return {
            "decisions": self.decision_history[-10:],
            "max_potential_workers": self.max_workers
        }
