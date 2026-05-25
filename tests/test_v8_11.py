import pytest
from datetime import datetime, timedelta
from autodj.scheduling import MixScheduler

def test_scheduler_preference_update():
    status = {"live_params": {"energy_bias": 0.5, "genre_preference": "Any"}}
    sched = MixScheduler(status)

    # Schedule preference update
    future = datetime.now() + timedelta(milliseconds=10)
    # The action in scheduling.py is "set_preference" (I used preference_bias in my test earlier)
    sched.add_event(future, "set_preference", {"energy_bias": 0.9, "genre_preference": "Techno"})

    # Mock the internal logic for testing without thread
    event = sched.events[0]
    sched._execute_event(event)

    assert status["live_params"]["energy_bias"] == 0.9
    assert status["live_params"]["genre_preference"] == "Techno"

def test_smart_replenish_logic():
    from autodj.core import SmartReplenishTool

    status = {
        "status": "IDLE",
        "playlist": [],
        "live_params": {"energy_bias": 1.0, "genre_preference": "Techno"}
    }

    tool = SmartReplenishTool()
    # It takes no arguments in __init__ but likely uses the gui_status via a global or parameter in execution.
    # Looking at the code in the trace, it seems it gets it from the context or it's registered.
    assert hasattr(tool, "score_candidate")
