# 📝 Immediate Tasks (8.16.0)

## 🚀 Features (Immediate Implementation)

- [ ] **Web Dashboard Bass-Transition Selector**:
  - [ ] Expose `--bass-transition [swap|fade]` as a live dropdown in `templates/index.html` and `autodj/gui.py`.
- [ ] **Per-Track Adaptive Gain Staging**:
  - [ ] Normalize per-track integrated loudness prior to overlap rendering to narrow the ~10 dB inter-track dynamic spread across multi-era compilations.
- [ ] **Rekordbox Hot-Cue Transition Export**:
  - [ ] Emit `POSITION_MARK` cue points at transition start, bass swap/fade midpoint, and outro end in `_rekordbox.xml`.
- [ ] **AI Genre Evolution**:
  - [ ] Transition from spectral-centroid heuristic to CNN-based style classification in `autodj/models.py`.

## 🛠 Improvements & Refactoring

- [ ] **Parallel Transition Worker**: Wire `transition_render_worker` in `autodj/core.py` into `ClusterNode` or remove unused path once benchmarked.
- [ ] **SciPy NumPy 2.5 Deprecation Guard**: Monitor `scipy.signal.sosfilt` / `filtfilt` updates for NumPy 2.5 shape assignment deprecation warnings.

## 📚 Documentation

- [ ] Create `Documentation/SCHEDULING_SPEC.md` for advanced event orchestration.
- [ ] Update `MANUAL.md` with Preference Bias and `--bass-transition` examples.

---
*Keep on goin'! Don't stop the party!*
