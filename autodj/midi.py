"""
MIDI Hardware Interface | Auto DJ Script (8.8.0)
================================================
This module provides real-time MIDI input handling for tactile control
of the Auto DJ engine, mapping MIDI CCs and Notes to DSP parameters.
"""

import mido
import threading
import time
from .plugins import ToolPlugin, PluginRegistry

class MidiHandler:
    """Handles connection and message processing for MIDI devices."""
    def __init__(self, port_name=None, mapping=None):
        self.port_name = port_name
        self.mapping = mapping or config_default_mapping()
        self.port = None
        self.running = False
        self.thread = None
        self.callback = None
        self.last_activity = 0

    def start(self, callback):
        self.callback = callback
        try:
            available_ports = mido.get_input_names()
            if not available_ports:
                print("[MIDI] No MIDI input devices found.")
                return False

            target_port = self.port_name or available_ports[0]
            self.port = mido.open_input(target_port)
            self.running = True
            self.thread = threading.Thread(target=self._listen, daemon=True)
            self.thread.start()
            print(f"[MIDI] Connected to {target_port}")
            return True
        except Exception as e:
            print(f"[MIDI] Connection failed: {e}")
            return False

    def stop(self):
        self.running = False
        if self.port:
            self.port.close()
        if self.thread:
            self.thread.join(timeout=1.0)

    def _listen(self):
        while self.running:
            for msg in self.port.iter_pending():
                self.last_activity = time.time()
                self._process_message(msg)
            time.sleep(0.01)

    def _process_message(self, msg):
        if not self.callback:
            return

        # Mapping logic
        if msg.type == 'control_change':
            # Map CC to parameter if in mapping
            param = self.mapping.get(f"cc_{msg.control}")
            if param:
                # Normalize 0-127 to 0.0-2.0 (for gains) or 0.0-1.0
                value = msg.value / 63.5 if "gain" in param else msg.value / 127.0
                self.callback(param, value)

        elif msg.type == 'note_on' and msg.velocity > 0:
            action = self.mapping.get(f"note_{msg.note}")
            if action:
                self.callback(action, True)

def config_default_mapping():
    """Default MIDI mapping for common DJ controllers."""
    return {
        "cc_16": "low_gain",     # Knob 1
        "cc_17": "mid_gain",     # Knob 2
        "cc_18": "high_gain",    # Knob 3
        "cc_19": "mastering_intensity",
        "cc_7":  "target_bpm_relative", # Fader
        "note_36": "handoff_requested", # Pad 1
        "note_37": "force_next",        # Pad 2
    }

@PluginRegistry.register_tool
class MidiHardwareTool(ToolPlugin):
    """Bridge between MIDI Hardware and the Auto DJ Engine."""
    name = "midi_hardware"
    display_name = "MIDI Hardware Controller"
    description = "Enables real-time tactile control via MIDI devices."
    version = "1.0.0"

    def __init__(self):
        self.handler = None
        self.status_obj = None

    def pre_mix(self, status_obj=None, **kwargs):
        self.status_obj = status_obj
        if status_obj:
            self.handler = MidiHandler()
            self.handler.start(self.on_midi_event)

    def post_mix(self, status_obj=None, **kwargs):
        if self.handler:
            self.handler.stop()

    def on_midi_event(self, param, value):
        if not self.status_obj:
            return

        live_params = self.status_obj.get("live_params", {})

        if param == "target_bpm_relative":
            # Map 0-1 to +/- 10% of current BPM
            current_bpm = live_params.get("target_bpm", 128.0)
            base_bpm = 128.0 # Should ideally come from initial args
            live_params["target_bpm"] = base_bpm * (0.9 + (value * 0.2))
        elif param in live_params:
            live_params[param] = value

        # Action-based notes
        if param == "force_next":
             # Implementation logic for force next would go here
             # Typically setting a flag that core.py reads
             live_params["handoff_requested"] = True # For now trigger handoff

        print(f"[MIDI] Set {param} to {value}")
