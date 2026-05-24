
import mido
import pytest
from autodj.midi import MidiHandler

def test_midi_logic():
    # Mock status object
    status = {
        "live_params": {
            "low_gain": 1.0,
            "handoff_requested": False,
            "target_bpm": 128.0
        }
    }

    def mock_callback(param, value):
        if param == "target_bpm_relative":
            status["live_params"]["target_bpm"] = 128.0 * (0.9 + (value * 0.2))
        elif param in status["live_params"]:
            status["live_params"][param] = value

    handler = MidiHandler()
    handler.callback = mock_callback

    # Testing Control Change mapping
    cc_msg = mido.Message('control_change', control=16, value=127) # low_gain
    handler._process_message(cc_msg)
    assert status["live_params"]["low_gain"] == 127 / 63.5

    # Testing Note On mapping
    note_msg = mido.Message('note_on', note=36, velocity=100) # handoff_requested
    handler._process_message(note_msg)
    assert status["live_params"]["handoff_requested"] == True

    # Testing BPM relative mapping
    fader_msg = mido.Message('control_change', control=7, value=127) # max fader
    handler._process_message(fader_msg)
    expected_bpm = 128.0 * 1.1
    assert abs(status["live_params"]["target_bpm"] - expected_bpm) < 0.001
