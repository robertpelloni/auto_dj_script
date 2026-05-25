"""
Distributed Rendering Cluster Manager for the Auto DJ system (7.0.0).
Coordinates rendering tasks across multiple local or remote nodes.
"""
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

class ClusterNode:
    """
    Represents a rendering node in the cluster.
    Defaults to local CPU processes.
    """
    def __init__(self, node_id, cores=1):
        self.id = node_id
        self.cores = cores
        self.status = "Idle"
        self.failed_tasks = 0
        self.total_tasks = 0

class RenderCluster:
    """
    Manages the distribution of audio rendering tasks.

    Architecture:
    - Segmented Rendering: Each transition/warping task is an independent 'RenderTask'.
    - Load Balancing: Dispatcher distributes tasks to nodes based on availability.
    """
    def __init__(self):
        self.nodes = [ClusterNode("LocalHost", cores=os.cpu_count() or 1)]
        self._executor = None
        self._current_workers = 2
        from .autoscaler import AutoScaler
        self.scaler = AutoScaler(self)

    def get_worker_count(self):
        return self._current_workers

    def scale_up(self):
        if self._current_workers < (os.cpu_count() or 4):
            self._current_workers += 1
            self.refresh_executor()

    def scale_down(self):
        if self._current_workers > 1:
            self._current_workers -= 1
            self.refresh_executor()

    def refresh_executor(self):
        """Restarts the executor with the new worker count."""
        if self._executor:
            self._executor.shutdown(wait=False)
        self._executor = ProcessPoolExecutor(max_workers=self._current_workers)

    def register_node(self, node_id, cores):
        """Registers a new remote node in the cluster."""
        # Check if node already exists
        for n in self.nodes:
            if n.id == node_id:
                n.cores = cores
                n.status = "Idle"
                return
        self.nodes.append(ClusterNode(node_id, cores=cores))

    def get_executor(self, status_obj=None):
        """Returns the persistent ProcessPoolExecutor instance, performing auto-scaling check."""
        # Perform scaling evaluation
        command = self.scaler.evaluate(status_obj)
        if command == "UP":
            self.scale_up()
        elif command == "DOWN":
            self.scale_down()

        if self._executor is None:
            self._executor = ProcessPoolExecutor(max_workers=self._current_workers)
        return self._executor

    def shutdown(self):
        """Gracefully shuts down the cluster."""
        if self._executor:
            self._executor.shutdown()
            self._executor = None

    def reset_node(self, node_id):
        """Resets the failure counter for a node."""
        for n in self.nodes:
            if n.id == node_id:
                n.failed_tasks = 0
                n.status = "Idle"

    def get_status(self):
        """Returns the status of the cluster nodes."""
        return [
            {
                "id": n.id,
                "cores": n.cores,
                "status": n.status,
                "failed_tasks": n.failed_tasks,
                "total_tasks": n.total_tasks
            }
            for n in self.nodes
        ]

# Global Cluster Instance
cluster = RenderCluster()
