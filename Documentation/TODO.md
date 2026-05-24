# 📝 Immediate Tasks (8.8.0)

## 🚀 Features (Immediate Implementation)
- [ ] **MIDI Hardware Integration**:
    - [ ] Install `mido` and `python-rtmidi`.
    - [ ] Implement `autodj/midi.py` for controller mapping.
    - [ ] Link MIDI CCs to EQ Gains (Low/Mid/High).
    - [ ] Link MIDI Notes to "Execute Handoff" and "Force Next".
- [ ] **Rekordbox Library Plugin**:
    - [ ] Create `RekordboxSourcePlugin` to parse `pioneer.xml`.
    - [ ] Extract Hot Cues and Memory Cues for automated alignment.
- [ ] **Hardware UI Dashboard**:
    - [ ] Add "Hardware" tab to Web Console.
    - [ ] MIDI activity indicator and mapping table.

## 🛠 Improvements & Refactoring
- [ ] **Optimization**: Reduce latency between MIDI input and DSP parameter updates.
- [ ] **Robustness**: Handle MIDI device disconnects/reconnects gracefully.

## 📚 Documentation
- [ ] Update `DEPLOY.md` with MIDI library dependencies.
- [ ] Create `Documentation/MIDI_MAPPING.md` for user configuration.

---
*Keep on goin'! Don't stop the party!*
