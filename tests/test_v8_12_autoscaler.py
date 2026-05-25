import pytest
from autodj.autoscaler import AutoScaler
from unittest.mock import patch

class MockCluster:
    def __init__(self, workers=2):
        self.workers = workers
    def get_worker_count(self):
        return self.workers

def test_autoscaler_downscaling():
    cluster = MockCluster(workers=4)
    scaler = AutoScaler(cluster)
    scaler.scaling_cooldown = 0

    with patch('psutil.cpu_percent', return_value=90.0):
        with patch('psutil.virtual_memory') as mock_mem:
            mock_mem.return_value.percent = 50.0
            decision = scaler.evaluate()
            assert decision == "DOWN"

def test_autoscaler_upscaling():
    cluster = MockCluster(workers=1)
    scaler = AutoScaler(cluster)
    scaler.scaling_cooldown = 0

    status = {"playlist": ["track1.flac", "track2.flac", "track3.flac"], "tracklist": []}

    with patch('psutil.cpu_percent', return_value=20.0):
        with patch('psutil.virtual_memory') as mock_mem:
            mock_mem.return_value.percent = 40.0
            decision = scaler.evaluate(status)
            assert decision == "UP"

def test_autoscaler_cooldown():
    cluster = MockCluster(workers=2)
    scaler = AutoScaler(cluster)
    scaler.scaling_cooldown = 60

    with patch('psutil.cpu_percent', return_value=95.0):
        decision1 = scaler.evaluate()
        assert decision1 == "DOWN"

        decision2 = scaler.evaluate()
        assert decision2 is None
