"""
Centralized Monitoring and Fault Management for Auto DJ (v7.4.0).
Manages structured logs, error metrics, and incident reports.
"""
import logging
import json
import time
from datetime import datetime

class ExecutionMonitor:
    """
    Tracks system robustness metrics and manages the incident log.
    """
    def __init__(self):
        self.incidents = []
        self.metrics = {
            "total_tasks": 0,
            "failed_tasks": 0,
            "retries": 0,
            "uptime_start": time.time()
        }
        self.logger = logging.getLogger("AutoDJ.Monitor")
        self.logger.setLevel(logging.INFO)

    def log_incident(self, severity, module, message, traceback=None):
        """Records a structured incident for the recovery console."""
        incident = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "severity": severity,
            "module": module,
            "message": message,
            "traceback": traceback
        }
        self.incidents.append(incident)
        # Keep only the last 50 incidents
        if len(self.incidents) > 50:
            self.incidents.pop(0)

        log_msg = f"[{severity}] {module}: {message}"
        if severity == "ERROR":
            self.logger.error(log_msg)
        else:
            self.logger.info(log_msg)

    def record_success(self):
        self.metrics["total_tasks"] += 1

    def record_failure(self):
        self.metrics["total_tasks"] += 1
        self.metrics["failed_tasks"] += 1

    def record_retry(self):
        self.metrics["retries"] += 1

    def get_status(self):
        """Returns the monitoring state for the UI."""
        error_rate = (self.metrics["failed_tasks"] / self.metrics["total_tasks"] * 100) if self.metrics["total_tasks"] > 0 else 0
        return {
            "incidents": self.incidents,
            "metrics": self.metrics,
            "error_rate": f"{error_rate:.1f}%",
            "uptime": f"{int(time.time() - self.metrics['uptime_start'])}s"
        }

# Global Monitor Instance
monitor = ExecutionMonitor()
