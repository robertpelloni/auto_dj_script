"""Core Orchestration Engine | Auto DJ Script (8.11.0)
==================================================
The core engine is responsible for tracklist optimization (Simulated Annealing),
parallel audio preprocessing, and the final sample-accurate mix reconstruction.

Fixed: ProcessPoolExecutor->ThreadPoolExecutor, analyze_geometry 4-value destructuring,
       librosa.resample guard, rubberband fallback, export error handling.
"""

import os
import random
import subprocess
import soundfile as sf
import numpy as np
from concurrent.futures import as_completed
import config

from .analysis import (
    get_native_bpm,
    get_musical_key,
    analyze_geometry,
    get_camelot_key,
    is_harmonically_compatible,
    get_energy_profile,
    detect_phrases,
    get_genre_archetype,
    find_sync_offset,
    extract_spectral_terrain,
)
from .dsp import trim_silence, normalize_lufs, apply_limiter, ArchetypeRegistry
from .utils import pydub_to_ndarray, ndarray_to_pydub
from .version import __version__
from .cluster import cluster
from .monitoring import monitor
from .plugins import PluginRegistry, ToolPlugin
import time


def wait_for_health(status_obj):
    """Execution Health Guardrail (v7.2.0)."""
    if status_obj is None:
        return
    while True:
        live = status_obj.get("live_params", {})
        telemetry = status_obj.get("telemetry", {})
        paused = live.get("paused", False)
        is_healthy = telemetry.get("is_healthy", True)
        if not paused and is_healthy:
            break
        orig_status = status_obj.get("status", "Processing")
        if paused:
            status_obj["status"] = "Session Paused (Manual)"
        elif not is_healthy:
            status_obj["status"] = "Auto-Throttled (High System Load)"
        time.sleep(1)
        status_obj["status"] = orig_status


def get_semitone_diff(key1, key2):
    """Calculates the distance in semitones between two musical keys."""
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    try:
        p1, p2 = key1.split(), key2.split()
        if p1[1] != p2[1]:
            return 0
        diff = notes.index(p2[0]) - notes.index(p1[0])
        if diff > 6:
            diff -= 12
        if diff < -6:
            diff += 12
        return diff
    except:
        return 0


def _resample_if_needed(y, sr, target_sr):
    """Resample audio if sample rate doesn't match. Falls back gracefully."""
    if sr == target_sr:
        return y, sr
    try:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        return y, target_sr
    except ImportError:
        print(
            f"  [WARN] librosa not available for resampling {sr}->{target_sr}. Using native SR."
        )
        return y, sr


def dynamic_warp(y, sr, native_bpm, start_target_bpm, end_target_bpm):
    """High-fidelity time-stretching via Rubber Band, with librosa fallback."""
    avg_target_bpm = (start_target_bpm + end_target_bpm) / 2.0
    rate = avg_target_bpm / native_bpm

    # Try Rubber Band first
    try:

        def rb_stretch(data, rb_rate):
            if data.ndim == 1:
                data = np.vstack([data, data])
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as fin, tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fout:
                sf.write(fin.name, data.T, sr, format="WAV", subtype="PCM_16")
                fin.close()
                fout.close()
                subprocess.run(
                    ["rubberband", "--tempo", str(rb_rate), fin.name, fout.name],
                    check=True,
                    capture_output=True,
                )
                out_y, sr_out = sf.read(fout.name, dtype="float32")
                if out_y.ndim == 2:
                    out_y = out_y.T
                os.remove(fin.name)
                os.remove(fout.name)
                if out_y.ndim == 1:
                    out_y = np.vstack([out_y, out_y])
                return out_y

        import tempfile

        print(f"  [RB] rate={rate:.4f} ({native_bpm:.1f} -> {avg_target_bpm:.1f})")
        return rb_stretch(y, rate)
    except (FileNotFoundError, subprocess.CalledProcessError, Exception) as e:
        print(
            f"  [WARN] Rubber Band unavailable ({e}). Falling back to librosa phase vocoder."
        )

    # Fallback: librosa phase vocoder
    try:
        import librosa

        if y.ndim == 2:
            y_mono = librosa.to_mono(y)
        else:
            y_mono = y
        stretched = librosa.effects.time_stretch(y_mono, rate=rate)
        if stretched.ndim == 1:
            stretched = np.vstack([stretched, stretched])
        return stretched
    except Exception as e2:
        print(f"  [ERROR] All stretch methods failed: {e2}")
        return y


def warp_worker(args):
    """Thread worker for track preparation (time-stretch, normalize)."""
    path, native_bpm, s_bpm, e_bpm, cur_key, tar_key, sync = args
    try:
        target_sr = 44100
        y, sr = sf.read(path, dtype="float32")
        if y.ndim == 2:
            y = y.T  # (channels, samples)

        y, sr = _resample_if_needed(y, sr, target_sr)

        print(f"  [LOAD] {os.path.basename(path)} - SoundFile OK")
        y_w = dynamic_warp(y, sr, native_bpm, s_bpm, e_bpm)
        y_w = apply_limiter(normalize_lufs(y_w, sr, config.TARGET_LUFS))
        return y_w, sr
    except Exception as e:
        import traceback

        monitor.log_incident(
            "ERROR",
            "WarpWorker",
            f"Failed: {os.path.basename(path)}: {e}",
            traceback.format_exc(),
        )
        return None, str(e)


def analyze_track_worker(f):
    """Metadata extraction worker with fast-loading."""
    try:
        target_sr = 44100
        y, sr = sf.read(f, dtype="float32")
        if y.ndim == 2:
            y = y.T
        y, sr = _resample_if_needed(y, sr, target_sr)

        native_bpm, _, _ = get_native_bpm(y, sr)
        y_mono = np.mean(y, axis=0) if y.ndim == 2 else y
        genre, rationale = get_genre_archetype(y_mono, sr, bpm=native_bpm)
        terrain = extract_spectral_terrain(y, sr)

        return {
            "path": f,
            "bpm": native_bpm,
            "key": get_musical_key(y_mono, sr),
            "energy": get_energy_profile(y_mono, sr),
            "genre": genre,
            "rationale": rationale,
            "terrain": terrain,
        }
    except Exception as e:
        print(f"[ERROR] analyze_track_worker failed for {f}: {e}")
        return {"path": f, "error": str(e)}


def find_optimal_order(files, status_obj=None):
    """Sequencing optimization via Simulated Annealing."""
    total = len(files)
    print(f"[*] Analyzing {total} tracks (Sequential)...")

    results = []
    for i, f in enumerate(files):
        if status_obj:
            status_obj.setdefault("active_tasks", {})[f] = "Analyzing..."
        r = analyze_track_worker(f)
        results.append(r)
        if status_obj:
            status_obj.get("active_tasks", {}).pop(f, None)
            status_obj["status"] = f"Analyzing Library ({i + 1}/{total})"
            status_obj["progress"] = int(((i + 1) / total) * 50)

        if "error" not in r:
            print(
                f"  [{i + 1}/{total}] {os.path.basename(f)}: BPM={r['bpm']:.1f}, Key={r['key']}, Genre={r['genre']}"
            )
        else:
            print(f"  [{i + 1}/{total}] {os.path.basename(f)}: ERROR - {r['error']}")

    meta = [r for r in results if "error" not in r]
    if not meta:
        return files, None

    def score_transition(t1, t2):
        s = 50 if is_harmonically_compatible(t1["key"], t2["key"]) else 0
        if abs(get_semitone_diff(t1["key"], t2["key"])) <= 2:
            s += 25
        e_diff = t2["energy"] - t1["energy"]
        s += 20 if e_diff > 0 else 0
        s -= abs(e_diff) * 100
        if t1["genre"] == t2["genre"]:
            s += 30
        return s

    order = [meta.pop(0)]
    while meta:
        best_idx = np.argmax([score_transition(order[-1], m) for m in meta])
        order.append(meta.pop(best_idx))

    def score_set(o):
        return sum(score_transition(o[i], o[i + 1]) for i in range(len(o) - 1))

    best_o, best_s = list(order), score_set(order)
    temp = config.SA_INITIAL_TEMP
    for _ in range(config.SA_ITERATIONS):
        if len(best_o) < 3:
            break
        new_o = list(best_o)
        i, j = random.sample(range(1, len(new_o)), 2)
        new_o[i], new_o[j] = new_o[j], new_o[i]
        new_s = score_set(new_o)
        if new_s > best_s or random.random() < np.exp(
            min(700, (new_s - best_s) / temp)
        ):
            best_o, best_s = new_o, new_s
        temp = config.SA_INITIAL_TEMP / np.log(1 + _ + 1)

    if status_obj is not None:
        status_obj["status"] = "Optimizing track order..."
        status_obj["progress"] = 50

    return [x["path"] for x in best_o], best_o


def ms_to_timestamp(ms):
    s = int((ms / 1000) % 60)
    m = int((ms / (1000 * 60)) % 60)
    h = int((ms / (1000 * 60 * 60)) % 24)
    return f"{h:02d}:{m:02d}:{s:02d}"

    @PluginRegistry.register_tool
    class SmartReplenishTool(ToolPlugin):
        """Autonomous queue replenishment tool."""

        name = "smart_replenish"
        display_name = "Smart Replenish"
        description = "Automatically adds new tracks to the queue based on harmonic compatibility."

        def on_track_start(self, track_meta, status_obj=None, **kwargs):
            if not status_obj or not status_obj.get("live_params", {}).get(
                "continuous_mode"
            ):
                return
            playlist = status_obj.get("playlist", [])
            tracklist = status_obj.get("tracklist", [])
            if len(playlist) < 3:
                source_name = status_obj.get("active_source", "local_folder")
                source_cls = PluginRegistry.get_sources().get(source_name)
                if not source_cls:
                    return
                source = source_cls()
                all_tracks = source.get_tracks(folder=status_obj.get("input_folder"))
                already_seen = set(t["file"] for t in tracklist) | set(playlist)
                candidates = [
                    t
                    for t in all_tracks
                    if os.path.basename(t) not in already_seen and t not in already_seen
                ]
                if not candidates:
                    return
                target = random.choice(candidates)
                status_obj["playlist"].append(os.path.basename(target))
                print(
                    f"[*] Smart Replenish: Added {os.path.basename(target)} to queue."
                )


def compile_master_set(args, status_obj=None):
    """The High-Performance Mixing Pipeline (8.11.0)."""
    folder = args.input

    # Modular Source Discovery
    source_type = getattr(args, "source_plugin", "local_folder")
    source_cls = PluginRegistry.get_sources().get(source_type)

    # Modular Tool Loading
    active_tools = [cls() for name, cls in PluginRegistry.get_tools().items()]
    for tool in active_tools:
        tool.pre_mix(status_obj=status_obj, args=args)

    if status_obj and status_obj.get("playlist"):
        all_files = [os.path.join(folder, f) for f in status_obj["playlist"]]
    elif source_cls:
        source = source_cls()
        all_files = source.get_tracks(
            folder=folder, extensions=config.SUPPORTED_EXTENSIONS
        )
    else:
        all_files = []

    if not all_files:
        if status_obj:
            status_obj["status"] = "Error: No audio files found"
        print("[ERROR] No audio files found in input folder.")
        return

    # Deduplicate (Windows: *.flac and *.FLAC return same files)
    all_files = list(set(os.path.abspath(f) for f in all_files))

    # Phase 1: Analysis (0-50%)
    all_files, meta_list = find_optimal_order(all_files, status_obj=status_obj)

    if meta_list is None:
        if status_obj:
            status_obj["status"] = "Error: Analysis failed"
        return

    num_tracks = len(all_files)

    # Phase 2: Warping (50-75%)
    if status_obj:
        status_obj["status"] = f"Warping {num_tracks} tracks"
    wait_for_health(status_obj)

    start_bpm = (
        status_obj.get("live_params", {}).get("target_bpm", args.bpm)
        if status_obj
        else args.bpm
    )
    end_bpm = args.end_bpm or start_bpm

    warp_tasks = []
    for i in range(num_tracks):
        t_s_bpm = start_bpm + (end_bpm - start_bpm) * (i / num_tracks)
        t_e_bpm = start_bpm + (end_bpm - start_bpm) * ((i + 1) / num_tracks)
        tar_key = meta_list[i - 1]["key"] if i > 0 else None
        warp_tasks.append(
            (
                all_files[i],
                meta_list[i]["bpm"],
                t_s_bpm,
                t_e_bpm,
                meta_list[i]["key"],
                tar_key,
                True,
            )
        )

    warped_results = [None] * num_tracks
    executor = cluster.get_executor()
    futures = {
        executor.submit(warp_worker, task): idx for idx, task in enumerate(warp_tasks)
    }

    for future in as_completed(futures):
        idx = futures[future]
        if status_obj:
            status_obj.setdefault("active_tasks", {})[all_files[idx]] = "Warping..."
        y_w, sr = future.result()
        if status_obj:
            status_obj.get("active_tasks", {}).pop(all_files[idx], None)

        # Fault Tolerance: Local Fallback
        if y_w is None:
            monitor.record_retry()
            monitor.log_incident(
                "WARN", "CoreEngine", f"Cluster task {idx} failed. Retrying locally..."
            )
            y_w, sr = warp_worker(warp_tasks[idx])
            if y_w is not None:
                monitor.record_success()
            else:
                monitor.record_failure()

        warped_results[idx] = (y_w, sr)
        if status_obj is not None:
            completed = sum(1 for x in warped_results if x is not None)
            status_obj["status"] = f"Warping track {completed}/{num_tracks}"
            status_obj["progress"] = 50 + int((completed / num_tracks) * 25)

    # Phase 3: Mixing (75-100%)
    if status_obj:
        status_obj["status"] = "Mixing Master Stream"

    tracklist, master, processed_tracks, current_time_ms = [], None, [], 0
    master_grid_offset = 0
    mix_executor = cluster.get_executor()

    for i in range(num_tracks):
        # Continuous mode: dynamic queue replenishment
        if status_obj and status_obj.get("playlist") and i >= len(warped_results):
            try:
                y_raw, sr_orig = sf.read(all_files[i], dtype="float32")
                if y_raw.ndim == 2:
                    y_raw = y_raw.T
                nbpm, _, _ = get_native_bpm(y_raw, sr_orig)
                mkey = get_musical_key(
                    y_raw if y_raw.ndim == 1 else np.mean(y_raw, axis=0), sr_orig
                )
                gnr, _ = get_genre_archetype(
                    y_raw if y_raw.ndim == 1 else np.mean(y_raw, axis=0),
                    sr_orig,
                    bpm=nbpm,
                )
                meta_list.append({"bpm": nbpm, "key": mkey, "genre": gnr})
                tar_key = meta_list[i - 1]["key"] if i > 0 else None
                y_w, sr = warp_worker(
                    (all_files[i], nbpm, start_bpm, start_bpm, mkey, tar_key, True)
                )
                warped_results.append((y_w, sr))
            except Exception as e:
                print(f"[ERROR] Dynamic warp failed for {all_files[i]}: {e}")
                warped_results.append((None, None))

        y_w, sr = warped_results[i] if i < len(warped_results) else (None, None)
        if y_w is None:
            print(f"[WARN] Skipping track {i}: warp failed")
            continue

        # In-Memory Conversion
        nxt = ndarray_to_pydub(y_w, sr)
        nxt = trim_silence(nxt)
        processed_tracks.append(
            (nxt, y_w, sr, None)
        )  # beats added after analyze_geometry

        if master is None:
            master = nxt
            # Anchor the global grid and store first track's beats
            master_beats, _, master_grid_offset, _ = analyze_geometry(
                nxt, sr, start_bpm, args.beats_per_bar, args.transition_bars
            )
            # Update first track's beats in processed_tracks
            if len(processed_tracks) > 0:
                pt = processed_tracks[0]
                processed_tracks[0] = (pt[0], pt[1], pt[2], master_beats)
            track_meta = {
                "timestamp": "00:00:00",
                "file": os.path.basename(all_files[i]),
                "key": f"{meta_list[i]['key']} ({get_camelot_key(meta_list[i]['key'])})",
                "genre": meta_list[i]["genre"],
                "rationale": meta_list[i].get("rationale", ""),
                "terrain": meta_list[i].get("terrain", []),
                "start_ms": 0,
            }
            tracklist.append(track_meta)
            current_time_ms = len(master)
            continue

        prev_nxt, prev_y_w, _, _ = processed_tracks[i - 1]

        # Poll live BPM
        current_target_bpm = (
            status_obj.get("live_params", {}).get("target_bpm", start_bpm)
            if status_obj
            else start_bpm
        )
        t_s_bpm = current_target_bpm + (end_bpm - current_target_bpm) * (i / num_tracks)

        beats, theoretical_ms_trans, first_beat_ms, last_beat_ms = analyze_geometry(
            nxt, sr, t_s_bpm, args.beats_per_bar, args.transition_bars
        )
        # Update processed_tracks with beats for this track
        if i < len(processed_tracks):
            pt = processed_tracks[i]
            processed_tracks[i] = (pt[0], pt[1], pt[2], beats)

        ph = detect_phrases(y_w, sr)
        ms_per_beat = 60000.0 / t_s_bpm
        ms_per_bar = ms_per_beat * args.beats_per_bar
        grid_size = ms_per_bar * 8

        fixed_p = (
            beats[min(args.transition_bars * args.beats_per_bar, len(beats) - 1)]
            if len(beats) > 0
            else theoretical_ms_trans
        )
        ideal_p = fixed_p
        if ph.any():
            cl = ph[np.argmin(np.abs(ph - fixed_p))]
            if abs(cl - fixed_p) < config.PHRASE_ANCHOR_TOLERANCE_MS:
                ideal_p = cl

        # Separate crossfade duration from total overlap
        # crossfade_ms: the actual DJ transition window (EQ swap + volume fade)
        # overlap_ms: total time both tracks play simultaneously
        crossfade_ms = ideal_p
        overlap_ms = max(ideal_p, first_beat_ms + int(ms_per_bar * 4))
        ms_trans = overlap_ms  # used for slicing

        # (pre_fade_ms computed after beat alignment, below)
        # Beat-Aligned Phase Alignment
        # Instead of a theoretical grid, align the incoming track's first beat
        # to the nearest actual beat in the outgoing track.
        current_kick_pos = current_time_ms - ms_trans + first_beat_ms

        # Get outgoing track's beat positions for alignment
        prev_beats = None
        if i > 0 and i - 1 < len(processed_tracks):
            pt = processed_tracks[i - 1]
            if len(pt) > 3 and pt[3] is not None:
                prev_beats = pt[3]

        if prev_beats is not None and len(prev_beats) > 0:
            # Convert prev track beats to master stream positions
            # The prev track starts at track_start_ms of the previous transition
            # For the first transition (i=1), the prev track starts at 0
            prev_track_start = 0
            for tm in tracklist:
                if os.path.basename(all_files[i - 1]) in tm.get("file", ""):
                    prev_track_start = tm.get("start_ms", 0)
                    break

            # Outgoing track's beats in master time
            master_beats = prev_beats + prev_track_start

            # Find the nearest beat to where the incoming first beat lands
            if len(master_beats) > 0:
                nearest_idx = np.argmin(np.abs(master_beats - current_kick_pos))
                nearest_beat = int(master_beats[nearest_idx])
                phase_error = nearest_beat - current_kick_pos

                # Only correct if within 1 bar (don't shift the transition massively)
                max_correction = int(ms_per_bar)
                if abs(phase_error) < max_correction:
                    ms_trans -= phase_error
        else:
            # Fallback: theoretical grid alignment
            relative_pos = current_kick_pos - master_grid_offset
            phase_error = relative_pos % grid_size if grid_size > 0 else 0
            if phase_error != 0:
                ms_trans += int(phase_error)

        # Sample-Accurate Kick-Peak Nudging
        # Instead of cross-correlating envelopes (which aligns overall musical
        # patterns and misses the kick peak offset), directly measure the offset
        # between kick drum peaks from both tracks and correct it.
        if ms_trans > 0 and ms_trans < len(master):
            m_slice = pydub_to_ndarray(master[-ms_trans:])
            n_slice = pydub_to_ndarray(nxt[:ms_trans])

            # Try kick-peak-based sync first
            sync_nudge = 0
            try:
                from scipy.signal import (
                    butter as _sb,
                    lfilter as _slf,
                    find_peaks as _sfp,
                )

                _sb_b, _sb_a = _sb(2, 80.0 / (0.5 * sr), btype="low")
                _min_dist = int(0.6 * 60 / t_s_bpm * sr)

                # Detect peaks in outgoing
                _m_mono = np.mean(m_slice, axis=0) if m_slice.ndim == 2 else m_slice
                _m_kick = np.abs(_slf(_sb_b, _sb_a, _m_mono))
                _m_max = np.max(_m_kick)
                if _m_max > 1e-6:
                    _m_kick_n = _m_kick / _m_max
                    _m_peaks, _ = _sfp(_m_kick_n, height=0.15, distance=_min_dist)
                else:
                    _m_peaks = np.array([])

                # Detect peaks in incoming
                _n_mono = np.mean(n_slice, axis=0) if n_slice.ndim == 2 else n_slice
                _n_kick = np.abs(_slf(_sb_b, _sb_a, _n_mono))
                _n_max = np.max(_n_kick)
                if _n_max > 1e-6:
                    _n_kick_n = _n_kick / _n_max
                    _n_peaks, _ = _sfp(_n_kick_n, height=0.15, distance=_min_dist)
                else:
                    _n_peaks = np.array([])

                if len(_m_peaks) >= 3 and len(_n_peaks) >= 3:
                    # Measure the offset between the two sets of peaks
                    _offsets = []
                    _max_dist = int(0.45 * 60 / t_s_bpm * sr)  # max half-beat
                    for _np in _n_peaks[:30]:
                        _dists = np.abs(_m_peaks - _np)
                        _nearest_idx = np.argmin(_dists)
                        if _dists[_nearest_idx] < _max_dist:
                            _offsets.append(int(_m_peaks[_nearest_idx]) - int(_np))

                    if len(_offsets) >= 3:
                        sync_nudge = int(np.median(_offsets) * 1000 / sr)
                        print(
                            f" [KICK-SYNC] peaks: out={len(_m_peaks)} in={len(_n_peaks)} offsets={len(_offsets)} nudge={sync_nudge}ms"
                        )

                if sync_nudge == 0:
                    # Fallback: cross-correlation based sync
                    sync_nudge = find_sync_offset(m_slice, n_slice, sr, t_s_bpm)
            except Exception as _sync_err:
                sync_nudge = find_sync_offset(m_slice, n_slice, sr, t_s_bpm)

            max_nudge = int(ms_per_beat * 2.0)
            sync_nudge = max(-max_nudge, min(max_nudge, sync_nudge))
            ms_trans -= sync_nudge  # incoming early -> delay it (decrease ms_trans)
            print(f" [SYNC] Nudge: {sync_nudge}ms")

        ms_trans = min(ms_trans, len(master))
        track_start_ms = len(master) - ms_trans

        track_meta = {
            "timestamp": ms_to_timestamp(track_start_ms),
            "file": os.path.basename(all_files[i]),
            "key": f"{meta_list[i]['key']} ({get_camelot_key(meta_list[i]['key'])})",
            "genre": meta_list[i]["genre"],
            "rationale": meta_list[i].get("rationale", ""),
            "terrain": meta_list[i].get("terrain", []),
            "start_ms": track_start_ms,
        }
        tracklist.append(track_meta)

        for tool in active_tools:
            tool.on_track_start(track_meta, status_obj=status_obj)

        if status_obj:
            status_obj["tracklist"] = tracklist
            status_obj["progress"] = 75 + int((i / max(num_tracks - 1, 1)) * 25)

        # Two-Phase Overlap with Continuous Volume Curves
        pre_fade_ms = max(0, ms_trans - crossfade_ms)
        if pre_fade_ms > 0:
            print(f" [PREFADE] {pre_fade_ms}ms ambient intro under outgoing track")

        m_body = master[:-ms_trans]
        m_overlap = pydub_to_ndarray(master[-ms_trans:])
        n_intro_full = pydub_to_ndarray(nxt[:ms_trans])
        n_body = nxt[ms_trans:]

        # Ensure both overlap arrays have exactly the same sample count
        # (pydub slicing from different segments can differ by +/-1 sample)
        m_len = m_overlap.shape[1] if m_overlap.ndim == 2 else len(m_overlap)
        n_len = n_intro_full.shape[1] if n_intro_full.ndim == 2 else len(n_intro_full)
        min_len = min(m_len, n_len)
        if m_overlap.ndim == 2:
            m_overlap = m_overlap[:, :min_len]
            n_intro_full = n_intro_full[:, :min_len]
        else:
            m_overlap = m_overlap[:min_len]
            n_intro_full = n_intro_full[:min_len]

        # Build continuous volume curves for the entire overlap
        total_samples = min_len
        if pre_fade_ms > 0 and total_samples > int(crossfade_ms * sr / 1000):
            pf_samples = min(int(pre_fade_ms * sr / 1000), total_samples)
            cf_len = total_samples - pf_samples
            cf_x = np.linspace(0, 1, max(cf_len, 1))
            pf_x = np.linspace(0, 1, max(pf_samples, 1))

            # Outgoing: full volume during pre-fade, then crossfade out
            out_curve = np.ones(total_samples)
            out_curve[pf_samples:] = np.cos(np.pi * cf_x**0.8 / 2)

            # Incoming: gentle ramp during pre-fade (0 -> 25%), then crossfade in (25% -> 100%)
            in_curve = np.zeros(total_samples)
            prefade_vol = 0.25
            in_curve[:pf_samples] = prefade_vol * (pf_x**0.7)  # gentle power ramp
            # Crossfade portion: scale sin curve to go from prefade_vol to 1.0
            cf_in_raw = np.sin(np.pi * cf_x**1.2 / 2)
            in_curve[pf_samples:] = prefade_vol + (1.0 - prefade_vol) * cf_in_raw
        else:
            # No pre-fade: standard crossfade curves over entire overlap
            x = np.linspace(0, 1, total_samples)
            out_curve = np.cos(np.pi * x**0.8 / 2)
            in_curve = np.sin(np.pi * x**1.2 / 2)

        # Apply archetype (bass swap / EQ) to the crossfade portion only
        mode = getattr(args, "archetype", "auto")
        if mode == "auto":
            mode = "progressive"
        dsp_kwargs = {
            "lowpass": args.lowpass,
            "highpass": args.highpass,
            "ideal_p": ideal_p,
        }

        # DJ-style bass management with no filter transients
        # Process the ENTIRE overlap through crossover filters (avoids
        # lfilter startup transients at segment boundaries), then use
        # gain curves that respect the pre-fade/crossfade structure.
        from scipy.signal import butter, filtfilt

        # Phase-coherent bass management using zero-phase subtraction.
        # We extract bass via zero-phase lowpass (filtfilt) and subtract
        # it from the original:
        #   output = audio - bass * (1 - gain)
        # When gain = 1.0: output = audio (exact, no artifacts)
        # When gain = 0.0: output = audio - bass (highpass effect)
        #
        # CRITICAL: filtfilt (zero-phase) replaces lfilter because:
        #   - lfilter introduces ~1.7ms group delay, misaligning the
        #     subtracted bass with the original. This creates a +2.1dB
        #     BOOST around 150-200Hz (constructive interference from phase
        #     shift) which sounds like "crosstalk ducking" during bass swap.
        #   - filtfilt has zero group delay, so subtraction is perfectly
        #     aligned: (1 - |H_LP|*alpha) is always a proper gain reduction
        #     at bass frequencies, with no frequency-dependent bumps.
        #
        b_lo, a_lo = butter(2, 150.0 / (0.5 * sr), btype="low")

        # Extract bass component using zero-phase filtering
        def _get_bass(audio):
            if audio.ndim == 2:
                bass = np.zeros_like(audio)
                for ch in range(audio.shape[0]):
                    bass[ch] = filtfilt(b_lo, a_lo, audio[ch])
            else:
                bass = filtfilt(b_lo, a_lo, audio)
            return bass

        m_bass = _get_bass(m_overlap)
        n_bass = _get_bass(n_intro_full)

        # Build bass gain curves aligned with the crossfade region
        # During pre-fade: outgoing bass at 100%, incoming bass at 0%
        # During crossfade: bass swap (outgoing fades out, incoming fades in)
        pf_s = (
            min(int(pre_fade_ms * sr / 1000), total_samples) if pre_fade_ms > 0 else 0
        )
        cf_len = total_samples - pf_s

        # Outgoing bass: full during pre-fade, then swap curve during crossfade
        m_bass_curve = np.ones(total_samples)
        if cf_len > 0:
            cf_x = np.linspace(0, 1, cf_len)
            # Bass stays full for first 40% of crossfade, then fades out
            m_cf_curve = np.ones(cf_len)
            mask = cf_x > 0.4
            m_cf_curve[mask] = 0.5 * (1 + np.cos(np.pi * (cf_x[mask] - 0.4) / 0.4))
            m_cf_curve[cf_x > 0.8] = 0.02
            m_bass_curve[pf_s:] = m_cf_curve

        # Incoming bass: silent during pre-fade, then swap curve during crossfade
        n_bass_curve = np.zeros(total_samples)
        if cf_len > 0:
            # Bass silent for first 30% of crossfade, then fades in
            n_cf_curve = np.zeros(cf_len)
            mask = cf_x > 0.3
            n_cf_curve[mask] = 0.5 * (1 - np.cos(np.pi * (cf_x[mask] - 0.3) / 0.4))
            n_cf_curve[cf_x > 0.7] = 1.0
            n_bass_curve[pf_s:] = n_cf_curve

        # NOTE: Beat-synced bass ducking removed � it caused audible
        # volume dips on false peak detections (breakdowns, ambient intros)
        # that sounded worse than the subtle galloping it was meant to fix.
        # The bass swap curves above already handle kick handoff cleanly.

        # Recombine using subtraction: audio - bass * (1 - gain)
        # This is sample-accurate when gain = 1.0 (no phase artifacts)
        if m_overlap.ndim == 2:
            m_proc = m_overlap - m_bass * (1 - m_bass_curve[np.newaxis, :])
            n_proc = n_intro_full - n_bass * (1 - n_bass_curve[np.newaxis, :])
        else:
            m_proc = m_overlap - m_bass * (1 - m_bass_curve)
            n_proc = n_intro_full - n_bass * (1 - n_bass_curve)

        # Safeguard: ensure arrays match total_samples
        for _n in range(2):
            arr = m_proc if _n == 0 else n_proc
            proc_len = arr.shape[1] if arr.ndim == 2 else len(arr)
            if proc_len != total_samples:
                if proc_len < total_samples:
                    pad = total_samples - proc_len
                    if arr.ndim == 2:
                        arr = np.pad(arr, ((0, 0), (0, pad)))
                    else:
                        arr = np.pad(arr, (0, pad))
                else:
                    if arr.ndim == 2:
                        arr = arr[:, :total_samples]
                    else:
                        arr = arr[:total_samples]
            if _n == 0:
                m_proc = arr
            else:
                n_proc = arr

        # Apply volume curves and sum into single continuous mix
        if m_proc.ndim == 2:
            mix_bus_raw = (
                m_proc * out_curve[np.newaxis, :] + n_proc * in_curve[np.newaxis, :]
            )
        else:
            mix_bus_raw = m_proc * out_curve + n_proc * in_curve
        mix_bus_raw = apply_limiter(mix_bus_raw)

        # Ensure mix_bus has exactly ms_trans duration
        expected_samples = int(ms_trans * sr / 1000)
        if mix_bus_raw.ndim == 2:
            actual_samples = mix_bus_raw.shape[1]
        else:
            actual_samples = len(mix_bus_raw)
        if actual_samples < expected_samples:
            pad_len = expected_samples - actual_samples
            if mix_bus_raw.ndim == 2:
                mix_bus_raw = np.pad(
                    mix_bus_raw, ((0, 0), (0, pad_len)), mode="constant"
                )
            else:
                mix_bus_raw = np.pad(mix_bus_raw, (0, pad_len), mode="constant")
        elif actual_samples > expected_samples:
            if mix_bus_raw.ndim == 2:
                mix_bus_raw = mix_bus_raw[:, :expected_samples]
            else:
                mix_bus_raw = mix_bus_raw[:expected_samples]

        mix_bus = ndarray_to_pydub(mix_bus_raw, sr)

        # Assemble: m_body + mix_bus + n_body (single continuous segment)
        master = m_body + mix_bus + n_body
        current_time_ms = len(master)

        # Memory management for continuous mode
        if (
            status_obj
            and status_obj.get("live_params", {}).get("continuous_mode")
            and i > 3
        ):
            if i - 3 < len(warped_results):
                warped_results[i - 3] = None
            if i - 3 < len(processed_tracks):
                processed_tracks[i - 3] = None
        if i == num_tracks - 1:
            # Final mix normalization: ensure uniform loudness across the entire mix
            if master:
                print("[MASTERING] Normalizing final mix to target LUFS...")
                mix_sr = master.frame_rate
                mix_raw = pydub_to_ndarray(master)
                mix_raw = normalize_lufs(mix_raw, mix_sr, config.TARGET_LUFS)
                mix_raw = apply_limiter(mix_raw)
                master = ndarray_to_pydub(mix_raw, mix_sr)
                print("[MASTERING] Final mix normalization complete.")

            # Export
            if master:
                output_type = getattr(args, "output_plugin", "local_file")
                output_cls = PluginRegistry.get_outputs().get(output_type)

                if output_cls:
                    if status_obj:
                        status_obj["status"] = "Exporting FLAC"
                        status_obj["progress"] = 98
                    try:
                        sink = output_cls()
                        sink.export(
                            master,
                            tracklist,
                            output=args.output,
                            version=__version__,
                            all_files=all_files,
                            meta_list=meta_list,
                            processed_tracks=processed_tracks,
                        )
                        if status_obj:
                            status_obj["status"] = "Complete"
                            status_obj["progress"] = 100
                        for tool in active_tools:
                            tool.post_mix(status_obj=status_obj)
                    except Exception as e:
                        print(f"[ERROR] Output plugin failed: {e}")
                        import traceback

                        traceback.print_exc()
                        if status_obj:
                            status_obj["status"] = f"Error: Export failed - {e}"
                else:
                    # Direct fallback export
                    if status_obj:
                        status_obj["status"] = "Exporting FLAC"
                        status_obj["progress"] = 98
                    try:
                        print(
                            f"[*] Exporting {len(master) / 1000 / 60:.1f} min mix to {args.output}..."
                        )
                        master.export(args.output, format="flac")
                        print("[*] Export complete!")
                        if status_obj:
                            status_obj["status"] = "Complete"
                            status_obj["progress"] = 100
                        tl_path = os.path.splitext(args.output)[0] + "_tracklist.txt"
                        with open(tl_path, "w") as f:
                            f.write(
                                f"Auto DJ v{__version__} Master Tracklist\n{'=' * 40}\n"
                            )
                            for item in tracklist:
                                f.write(
                                    f"[{item['timestamp']}] {item['file']} ({item['key']}) [{item['genre']}]\n"
                                )
                    except Exception as e:
                        print(f"[ERROR] Direct export failed: {e}")
                        if status_obj:
                            status_obj["status"] = f"Error: Export failed - {e}"

    def _dj_crossfade(audio_array, fade_type="in"):
        """Equal-power crossfade designed for DJ mixer style transitions.

        This works WITH the bass swap in _apply_bass_swap_transition:
        - The bass swap handles the low-frequency handoff (kick drum exchange)
        - This crossfade handles the mid/high volume balance
        - Together they create a smooth transition like a real DJ mixer

        The curves are asymmetric to match DJ practice:
        - Fade OUT (outro): gentle hold, then smooth release. The outro stays
          audible until late in the transition because the bass swap already
          reduced its low-end energy, making the remaining mid/high less prominent.
        - Fade IN (intro): smooth equal-power rise. The intro's bass is already
          managed by the swap, so we just need to bring up the overall level.
        """
        n = audio_array.shape[1] if audio_array.ndim == 2 else len(audio_array)
        x = np.linspace(0, 1, n)

        if fade_type == "out":
            # Outro: equal-power fade with slight hold
            # cos(pi*x/2) is the standard equal-power fade-out
            # The slight x^0.8 exponent keeps it louder a bit longer
            curve = np.cos(np.pi * x**0.8 / 2)
        else:
            # Intro: equal-power fade-in
            # sin(pi*x/2) is the standard equal-power fade-in
            # The slight x^1.2 exponent makes it come in a touch gentler
            curve = np.sin(np.pi * x**1.2 / 2)

        if audio_array.ndim == 2:
            return audio_array * curve[np.newaxis, :]
        return audio_array * curve

    def transition_render_worker(args):
        """Parallel worker for rendering a single transition overlap."""
        outro_raw, intro_raw, sr, mode, ms_trans, ideal_p, dsp_kwargs = args
        try:
            arch_plugin = ArchetypeRegistry.get(mode)
            if arch_plugin:
                f_m_raw, f_n_raw = arch_plugin.apply(
                    outro_raw, intro_raw, sr, **dsp_kwargs
                )
            else:
                # Fallback: use the bass swap transition (same as progressive)
                from .dsp import _apply_bass_swap_transition

                f_m_raw, f_n_raw = _apply_bass_swap_transition(outro_raw, intro_raw, sr)

            # Apply gentle volume crossfade (complements EQ swap in DualFilterSweep)
            f_m_faded = _dj_crossfade(f_m_raw, fade_type="out")
            f_n_faded = _dj_crossfade(f_n_raw, fade_type="in")

            # Mix-Bus Summation
            min_len = (
                min(f_m_faded.shape[1], f_n_faded.shape[1])
                if f_m_faded.ndim == 2
                else min(len(f_m_faded), len(f_n_faded))
            )
            if f_m_faded.ndim == 2:
                summed = f_m_faded[:, :min_len] + f_n_faded[:, :min_len]
            else:
                summed = f_m_faded[:min_len] + f_n_faded[:min_len]

            summed = apply_limiter(summed)
            return summed, sr
        except Exception as e:
            import traceback

            print(f"[ERROR] transition_render_worker failed: {e}")
            traceback.print_exc()
            return None, str(e)
