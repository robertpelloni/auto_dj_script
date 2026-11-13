"""Core Orchestration Engine | Auto DJ Script.
==================================================
Responsible for:
1. Track analysis (BPM, Krumhansl-Schmuckler Key/Camelot, Genre Archetype, 3D Spectral Terrain)
2. Optional Simulated Annealing harmonic & energy sequencing (`--reorder`)
3. Parallel time-stretching & LUFS loudness normalization (`ThreadPoolExecutor`)
4. Beat-walked + kick-peak sample-accurate phase alignment & zero-phase DJ bass transitions
5. Multiband dynamics mastering, lossless >4 GB FLAC rendering, Rekordbox XML cues, & live streaming
"""

import os
import random
import subprocess
import time
from collections import Counter
from concurrent.futures import as_completed
from datetime import datetime

import numpy as np
import soundfile as sf

import config
from .analysis import (
    analyze_geometry,
    calculate_dynamic_transition,
    detect_phrases,
    extract_spectral_terrain,
    find_sync_offset,
    get_camelot_key,
    get_energy_profile,
    get_genre_archetype,
    get_musical_key,
    get_native_bpm,
    is_harmonically_compatible,
)
from .cluster import cluster
from .dsp import (
    ArchetypeRegistry,
    apply_limiter,
    apply_multiband_compression,
    calculate_spectral_clash,
    normalize_lufs,
    trim_silence,
)
from .monitoring import monitor
from .plugins import PluginRegistry, ToolPlugin
from .utils import ndarray_to_pydub, pydub_to_ndarray, set_below_normal_priority
from .version import __version__


def _emit_log(status_obj, message: str):
    """Prints `message` to stdout and records it in `status_obj['logs']` for live UI telemetry."""
    print(message)
    if status_obj is not None:
        logs = status_obj.setdefault("logs", [])
        stamp = datetime.now().strftime("%H:%M:%S")
        logs.append(f"[{stamp}] {message}")
        if len(logs) > 250:
            del logs[:-250]


def wait_for_health(status_obj):
    """Execution Health Guardrail (pauses on manual pause or CPU/RAM threshold breach)."""
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
    except Exception:
        return 0


def _resample_if_needed(y, sr, target_sr):
    """Resamples audio if sample rate differs from `target_sr`."""
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
    """High-fidelity time-stretching via Rubber Band CLI, with librosa phase-vocoder fallback."""
    avg_target_bpm = (start_target_bpm + end_target_bpm) / 2.0
    rate = avg_target_bpm / max(float(native_bpm), 1.0)

    if abs(rate - 1.0) < 0.0015:
        return y if y.ndim == 2 else np.vstack([y, y])

    # Try Rubber Band CLI first
    try:
        import tempfile

        def rb_stretch(data, rb_rate):
            if data.ndim == 1:
                data = np.vstack([data, data])
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as fin, tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fout:
                sf.write(fin.name, data.T, sr, format="WAV", subtype="PCM_16")
                fin.close()
                fout.close()
                try:
                    subprocess.run(
                        ["rubberband", "--tempo", str(rb_rate), fin.name, fout.name],
                        check=True,
                        capture_output=True,
                    )
                    out_y, _ = sf.read(fout.name, dtype="float32")
                    if out_y.ndim == 2:
                        out_y = out_y.T
                    if out_y.ndim == 1:
                        out_y = np.vstack([out_y, out_y])
                    return out_y
                finally:
                    for p in (fin.name, fout.name):
                        if os.path.exists(p):
                            try:
                                os.remove(p)
                            except OSError:
                                pass

        print(f"  [RB] rate={rate:.4f} ({native_bpm:.1f} -> {avg_target_bpm:.1f})")
        return rb_stretch(y, rate)
    except Exception as e:
        print(
            f"  [WARN] Rubber Band unavailable ({type(e).__name__}). Falling back to librosa phase vocoder."
        )

    # Fallback: librosa phase vocoder (preserves stereo channels independently when stereo)
    try:
        import librosa

        if y.ndim == 2 and y.shape[0] == 2:
            left = librosa.effects.time_stretch(y[0], rate=rate)
            right = librosa.effects.time_stretch(y[1], rate=rate)
            min_s = min(len(left), len(right))
            return np.vstack([left[:min_s], right[:min_s]])
        y_mono = librosa.to_mono(y) if y.ndim == 2 else y
        stretched = librosa.effects.time_stretch(y_mono, rate=rate)
        return np.vstack([stretched, stretched]) if stretched.ndim == 1 else stretched
    except Exception as e2:
        print(f"  [ERROR] All stretch methods failed: {e2}")
        return y


def warp_worker(args):
    """Thread worker for track preparation (time-stretch + per-track LUFS normalization)."""
    path, native_bpm, s_bpm, e_bpm, cur_key, tar_key, target_lufs = args
    try:
        target_sr = 44100
        y, sr = sf.read(path, dtype="float32")
        if y.ndim == 2:
            y = y.T

        y, sr = _resample_if_needed(y, sr, target_sr)

        print(f"  [LOAD] {os.path.basename(path)} - SoundFile OK")
        y_w = dynamic_warp(y, sr, native_bpm, s_bpm, e_bpm)
        y_w = apply_limiter(normalize_lufs(y_w, sr, target_lufs))
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
    """Metadata extraction worker with fast-loading and 3D spectral terrain extraction."""
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
        duration_sec = round(float(y_mono.shape[0]) / float(sr), 2)

        return {
            "path": f,
            "bpm": native_bpm,
            "key": get_musical_key(y_mono, sr),
            "energy": float(get_energy_profile(y_mono, sr)),
            "genre": genre,
            "rationale": rationale,
            "terrain": terrain,
            "duration_sec": duration_sec,
        }
    except Exception as e:
        print(f"[ERROR] analyze_track_worker failed for {f}: {e}")
        monitor.log_incident("ERROR", "Analyzer", f"{os.path.basename(f)}: {e}")
        return {"path": f, "error": str(e)}


def analyze_tracks_sequential(files, status_obj=None):
    """Extracts BPM, Key, Energy, Genre, and Spectral Terrain for `files` in order."""
    total = len(files)
    _emit_log(status_obj, f"[*] Analyzing {total} tracks...")

    results = []
    for i, f in enumerate(files):
        wait_for_health(status_obj)
        if status_obj:
            status_obj.setdefault("active_tasks", {})[f] = "Analyzing..."
        r = analyze_track_worker(f)
        results.append(r)
        if status_obj:
            status_obj.get("active_tasks", {}).pop(f, None)
            status_obj["status"] = f"Analyzing Library ({i + 1}/{total})"
            status_obj["progress"] = int(((i + 1) / max(total, 1)) * 45)

        if "error" not in r:
            _emit_log(
                status_obj,
                f"  [{i + 1}/{total}] {os.path.basename(f)}: BPM={r['bpm']:.1f}, Key={r['key']} ({get_camelot_key(r['key'])}), Genre={r['genre']}",
            )
        else:
            _emit_log(
                status_obj,
                f"  [{i + 1}/{total}] {os.path.basename(f)}: ERROR - {r['error']}",
            )

    meta = [r for r in results if "error" not in r]
    return meta


def optimize_track_order(meta_list, status_obj=None):
    """Optimizes track sequence via greedy nearest-neighbor + Simulated Annealing."""
    if len(meta_list) <= 2:
        return [x["path"] for x in meta_list], list(meta_list)

    live_params = (status_obj or {}).get("live_params", {})
    energy_bias = float(live_params.get("energy_bias", 0.5))
    preferred_genre = str(live_params.get("genre_preference", "Any") or "Any")

    def score_transition(t1, t2):
        s = 50.0 if is_harmonically_compatible(t1["key"], t2["key"]) else 0.0
        if abs(get_semitone_diff(t1["key"], t2["key"])) <= 2:
            s += 25.0
        e_diff = t2["energy"] - t1["energy"]
        # energy_bias > 0.5 rewards rising energy; < 0.5 rewards steady/cooling energy
        if energy_bias >= 0.5 and e_diff >= 0:
            s += 30.0 * energy_bias
        elif energy_bias < 0.5 and e_diff <= 0:
            s += 30.0 * (1.0 - energy_bias)
        s -= abs(e_diff) * 100.0
        if t1["genre"] == t2["genre"]:
            s += 30.0
        if preferred_genre != "Any" and t2["genre"].lower() == preferred_genre.lower():
            s += 25.0
        return s

    pool = list(meta_list)
    order = [pool.pop(0)]
    while pool:
        best_idx = int(np.argmax([score_transition(order[-1], m) for m in pool]))
        order.append(pool.pop(best_idx))

    def score_set(o):
        return sum(score_transition(o[i], o[i + 1]) for i in range(len(o) - 1))

    best_o, best_s = list(order), score_set(order)
    temp = float(config.SA_INITIAL_TEMP)
    for step in range(config.SA_ITERATIONS):
        if len(best_o) < 3:
            break
        new_o = list(best_o)
        i, j = random.sample(range(1, len(new_o)), 2)
        new_o[i], new_o[j] = new_o[j], new_o[i]
        new_s = score_set(new_o)
        if new_s > best_s or random.random() < np.exp(
            min(700.0, (new_s - best_s) / max(temp, 1e-6))
        ):
            best_o, best_s = new_o, new_s
        temp = config.SA_INITIAL_TEMP / np.log(step + 2.0)

    if status_obj is not None:
        status_obj["status"] = "Optimizing track order..."
        status_obj["progress"] = 50
    _emit_log(status_obj, f"[*] Simulated Annealing sequence score: {best_s:.1f}")
    return [x["path"] for x in best_o], best_o


def find_optimal_order(files, status_obj=None, reorder=True):
    """Analyzes `files` and optionally optimizes their harmonic/energy sequence."""
    meta = analyze_tracks_sequential(files, status_obj=status_obj)
    if not meta:
        return files, None
    if reorder:
        return optimize_track_order(meta, status_obj=status_obj)
    if status_obj is not None:
        status_obj["progress"] = 50
    return [x["path"] for x in meta], meta


def ms_to_timestamp(ms):
    s = int((ms / 1000) % 60)
    m = int((ms / (1000 * 60)) % 60)
    h = int((ms / (1000 * 60 * 60)) % 24)
    return f"{h:02d}:{m:02d}:{s:02d}"


@PluginRegistry.register_tool
class SmartReplenishTool(ToolPlugin):
    """Autonomous queue replenishment tool guided by harmonic/genre preferences."""

    name = "smart_replenish"
    display_name = "Smart Replenish"
    description = "Automatically adds compatible unseen tracks to the active queue."

    def replenish_once(self, status_obj, reference_meta=None) -> str | None:
        if not status_obj:
            return None
        playlist = status_obj.setdefault("playlist", [])
        tracklist = status_obj.get("tracklist", [])
        source_name = status_obj.get("active_source", "local_folder")
        source_cls = PluginRegistry.get_sources().get(source_name)
        if not source_cls:
            return None
        folder = status_obj.get("input_folder") or config.INPUT_FOLDER
        source = source_cls()
        all_tracks = source.get_tracks(
            folder=folder,
            xml_path=status_obj.get("rekordbox_xml_path"),
            extensions=config.SUPPORTED_EXTENSIONS,
        )
        already_seen = {t.get("file") for t in tracklist} | set(playlist)
        candidates = [
            t
            for t in all_tracks
            if os.path.basename(t) not in already_seen and t not in already_seen
        ]
        if not candidates:
            return None

        target = random.choice(candidates)
        chosen_name = os.path.basename(target)
        playlist.append(chosen_name)
        _emit_log(status_obj, f"[*] Smart Replenish: Queued {chosen_name}")
        return chosen_name

    def on_track_start(self, track_meta, status_obj=None, **kwargs):
        if not status_obj or not status_obj.get("live_params", {}).get(
            "continuous_mode"
        ):
            return
        if len(status_obj.get("playlist", [])) < 3:
            self.replenish_once(status_obj, reference_meta=track_meta)


def compile_master_set(args, status_obj=None):
    """The High-Performance Mixing Pipeline."""
    if getattr(args, "below_normal_priority", True):
        prio = set_below_normal_priority()
        if status_obj is not None:
            status_obj.setdefault("telemetry", {})["process_priority"] = prio

    folder = getattr(args, "input", config.INPUT_FOLDER)
    source_type = getattr(args, "source_plugin", "local_folder")
    source_cls = PluginRegistry.get_sources().get(source_type)

    if status_obj is not None:
        status_obj["active_source"] = source_type
        status_obj["input_folder"] = folder
        status_obj["rekordbox_xml_path"] = getattr(args, "rekordbox_xml_path", None)

    active_tools = [cls() for _, cls in PluginRegistry.get_tools().items()]
    for tool in active_tools:
        tool.pre_mix(status_obj=status_obj, args=args)

    if status_obj and status_obj.get("playlist"):
        all_files = [
            f if os.path.isabs(f) else os.path.join(folder, f)
            for f in status_obj["playlist"]
        ]
    elif source_cls:
        source = source_cls()
        all_files = source.get_tracks(
            folder=folder,
            xml_path=getattr(args, "rekordbox_xml_path", None),
            extensions=config.SUPPORTED_EXTENSIONS,
        )
    else:
        all_files = []

    # Deterministic, case-normalized deduplication (preserves explicit playlist order!)
    deduped = {}
    for f in all_files:
        abs_f = os.path.abspath(f)
        if os.path.exists(abs_f):
            norm_k = os.path.normcase(abs_f)
            if norm_k not in deduped:
                deduped[norm_k] = abs_f
    all_files = list(deduped.values())

    if not all_files:
        if status_obj:
            status_obj["status"] = "Error: No audio files found"
        _emit_log(status_obj, "[ERROR] No audio files found in input source.")
        return

    # ── Phase 1: Analysis & Optional Sequencing (0–50%) ──
    t_phase1 = time.perf_counter()
    reorder_flag = bool(getattr(args, "reorder", False))
    all_files, meta_list = find_optimal_order(
        all_files, status_obj=status_obj, reorder=reorder_flag
    )
    monitor.record_task_duration("analysis", round(time.perf_counter() - t_phase1, 3))

    if meta_list is None:
        if status_obj:
            status_obj["status"] = "Error: Analysis failed"
        return

    num_tracks = len(all_files)

    # Dry-run mode: populate tracklist analysis report without time-stretching or rendering audio
    if getattr(args, "dry_run", False):
        preview_tl = []
        cursor_ms = 0
        for idx, m in enumerate(meta_list):
            preview_tl.append(
                {
                    "timestamp": ms_to_timestamp(cursor_ms),
                    "file": os.path.basename(m["path"]),
                    "bpm": round(float(m["bpm"]), 1),
                    "key": f"{m['key']} ({get_camelot_key(m['key'])})",
                    "genre": m["genre"],
                    "rationale": m.get("rationale", ""),
                    "terrain": m.get("terrain", []),
                    "start_ms": cursor_ms,
                    "overlap_ms": 0,
                }
            )
            cursor_ms += int(m.get("duration_sec", 0) * 1000)
        _emit_log(
            status_obj,
            f"[DRY-RUN] Analyzed {num_tracks} tracks cleanly (audio render skipped).",
        )
        if status_obj is not None:
            status_obj["tracklist"] = preview_tl
            status_obj["status"] = "Dry Run Complete"
            status_obj["progress"] = 100
        for tool in active_tools:
            tool.post_mix(status_obj=status_obj)
        return

    # ── Phase 2: Parallel Time-Stretching & Normalization (50–75%) ──
    t_phase2 = time.perf_counter()
    if status_obj:
        status_obj["status"] = f"Warping {num_tracks} tracks"
    wait_for_health(status_obj)

    start_bpm = (
        status_obj.get("live_params", {}).get("target_bpm", args.bpm)
        if status_obj
        else args.bpm
    )
    end_bpm = getattr(args, "end_bpm", None) or start_bpm
    target_lufs = float(getattr(args, "target_lufs", config.TARGET_LUFS))

    warp_tasks = []
    for i in range(num_tracks):
        t_s_bpm = start_bpm + (end_bpm - start_bpm) * (i / max(num_tracks, 1))
        t_e_bpm = start_bpm + (end_bpm - start_bpm) * ((i + 1) / max(num_tracks, 1))
        tar_key = meta_list[i - 1]["key"] if i > 0 else None
        warp_tasks.append(
            (
                all_files[i],
                meta_list[i]["bpm"],
                t_s_bpm,
                t_e_bpm,
                meta_list[i]["key"],
                tar_key,
                target_lufs,
            )
        )

    warped_results = [None] * num_tracks
    if cluster.nodes:
        cluster.nodes[0].status = "Busy (Warping)"
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

        if y_w is None:
            monitor.record_retry()
            if cluster.nodes:
                cluster.nodes[0].failed_tasks += 1
            monitor.log_incident(
                "WARN", "CoreEngine", f"Cluster task {idx} failed. Retrying locally..."
            )
            y_w, sr = warp_worker(warp_tasks[idx])
            if y_w is not None:
                monitor.record_success()
            else:
                monitor.record_failure()
        else:
            monitor.record_success()

        if cluster.nodes:
            cluster.nodes[0].total_tasks += 1

        warped_results[idx] = (y_w, sr)
        if status_obj is not None:
            completed = sum(1 for x in warped_results if x is not None)
            status_obj["status"] = f"Warping track {completed}/{num_tracks}"
            status_obj["progress"] = 50 + int((completed / max(num_tracks, 1)) * 25)

    if cluster.nodes:
        cluster.nodes[0].status = "Idle"
    monitor.record_task_duration("warping", round(time.perf_counter() - t_phase2, 3))

    # ── Phase 3: Beat-Aligned Sample-Accurate Mixing (75–100%) ──
    t_phase3 = time.perf_counter()
    if status_obj:
        status_obj["status"] = "Mixing Master Stream"

    tracklist, master, processed_tracks, current_time_ms = [], None, [], 0
    master_grid_offset = 0

    for i in range(num_tracks):
        wait_for_health(status_obj)

        # Continuous mode: dynamic queue replenishment
        if status_obj and status_obj.get("playlist") and i >= len(warped_results):
            try:
                y_raw, sr_orig = sf.read(all_files[i], dtype="float32")
                if y_raw.ndim == 2:
                    y_raw = y_raw.T
                nbpm, _, _ = get_native_bpm(y_raw, sr_orig)
                mono_raw = y_raw if y_raw.ndim == 1 else np.mean(y_raw, axis=0)
                mkey = get_musical_key(mono_raw, sr_orig)
                gnr, rat = get_genre_archetype(mono_raw, sr_orig, bpm=nbpm)
                meta_list.append(
                    {
                        "path": all_files[i],
                        "bpm": nbpm,
                        "key": mkey,
                        "genre": gnr,
                        "rationale": rat,
                        "terrain": extract_spectral_terrain(y_raw, sr_orig),
                    }
                )
                tar_key = meta_list[i - 1]["key"] if i > 0 else None
                y_w, sr = warp_worker(
                    (
                        all_files[i],
                        nbpm,
                        start_bpm,
                        start_bpm,
                        mkey,
                        tar_key,
                        target_lufs,
                    )
                )
                warped_results.append((y_w, sr))
            except Exception as e:
                _emit_log(
                    status_obj, f"[ERROR] Dynamic warp failed for {all_files[i]}: {e}"
                )
                warped_results.append((None, None))

        y_w, sr = warped_results[i] if i < len(warped_results) else (None, None)
        if y_w is None:
            _emit_log(status_obj, f"[WARN] Skipping track {i}: warp failed")
            processed_tracks.append((None, None, 44100, None))
            continue

        nxt = ndarray_to_pydub(y_w, sr)
        nxt = trim_silence(nxt)
        processed_tracks.append((nxt, y_w, sr, None))

        if master is None:
            master = nxt
            master_beats, _, master_grid_offset, _ = analyze_geometry(
                nxt, sr, start_bpm, args.beats_per_bar, args.transition_bars
            )
            if len(processed_tracks) > 0:
                pt = processed_tracks[0]
                processed_tracks[0] = (pt[0], pt[1], pt[2], master_beats)
            track_meta = {
                "timestamp": "00:00:00",
                "file": os.path.basename(all_files[i]),
                "bpm": round(float(meta_list[i]["bpm"]), 1),
                "key": f"{meta_list[i]['key']} ({get_camelot_key(meta_list[i]['key'])})",
                "genre": meta_list[i]["genre"],
                "rationale": meta_list[i].get("rationale", ""),
                "terrain": meta_list[i].get("terrain", []),
                "start_ms": 0,
                "overlap_ms": 0,
            }
            tracklist.append(track_meta)
            if status_obj:
                status_obj["tracklist"] = tracklist
            for tool in active_tools:
                tool.on_track_start(track_meta, status_obj=status_obj)
            current_time_ms = len(master)
            continue

        prev_entry = processed_tracks[i - 1] if i - 1 < len(processed_tracks) else None
        prev_y_w = prev_entry[1] if prev_entry else None

        current_target_bpm = (
            status_obj.get("live_params", {}).get("target_bpm", start_bpm)
            if status_obj
            else start_bpm
        )
        t_s_bpm = current_target_bpm + (end_bpm - current_target_bpm) * (
            i / max(num_tracks, 1)
        )

        effective_bars = int(args.transition_bars)
        if getattr(args, "dynamic_transitions", False) and prev_y_w is not None:
            # Adapt phrase bar count (8 / 16 / 32) only when tracks are long enough (>90s)
            if len(nxt) > 90_000 and len(master) > 90_000:
                effective_bars = calculate_dynamic_transition(
                    prev_y_w, y_w, sr, t_s_bpm, args.beats_per_bar
                )

        beats, theoretical_ms_trans, first_beat_ms, last_beat_ms = analyze_geometry(
            nxt, sr, t_s_bpm, args.beats_per_bar, effective_bars
        )
        if i < len(processed_tracks):
            pt = processed_tracks[i]
            processed_tracks[i] = (pt[0], pt[1], pt[2], beats)

        ph = detect_phrases(y_w, sr)
        ms_per_beat = 60000.0 / t_s_bpm
        ms_per_bar = ms_per_beat * args.beats_per_bar
        grid_size = ms_per_bar * 8

        fixed_p = (
            beats[min(effective_bars * args.beats_per_bar, len(beats) - 1)]
            if len(beats) > 0
            else theoretical_ms_trans
        )
        ideal_p = fixed_p
        if ph.any():
            cl = ph[np.argmin(np.abs(ph - fixed_p))]
            if abs(cl - fixed_p) < config.PHRASE_ANCHOR_TOLERANCE_MS:
                ideal_p = cl

        crossfade_ms = ideal_p
        overlap_ms = max(ideal_p, first_beat_ms + int(ms_per_bar * 4))
        ms_trans = min(overlap_ms, int(len(nxt) * 0.75), int(len(master) * 0.75))
        ms_trans = max(ms_trans, min(2000, len(nxt) // 2, len(master) // 2))

        current_kick_pos = current_time_ms - ms_trans + first_beat_ms

        prev_beats = None
        if prev_entry is not None and len(prev_entry) > 3 and prev_entry[3] is not None:
            prev_beats = prev_entry[3]

        if prev_beats is not None and len(prev_beats) > 0:
            prev_track_start = 0
            for tm in tracklist:
                if os.path.basename(all_files[i - 1]) in tm.get("file", ""):
                    prev_track_start = tm.get("start_ms", 0)
                    break

            master_beats = prev_beats + prev_track_start
            if len(master_beats) > 0:
                nearest_idx = np.argmin(np.abs(master_beats - current_kick_pos))
                nearest_beat = int(master_beats[nearest_idx])
                phase_error = nearest_beat - current_kick_pos
                max_correction = int(ms_per_bar)
                if abs(phase_error) < max_correction:
                    ms_trans -= phase_error
        else:
            relative_pos = current_kick_pos - master_grid_offset
            phase_error = relative_pos % grid_size if grid_size > 0 else 0
            if phase_error != 0:
                ms_trans += int(phase_error)

        # Sample-Accurate Kick-Peak Nudging
        if 0 < ms_trans < len(master) and ms_trans <= len(nxt):
            m_slice = pydub_to_ndarray(master[-ms_trans:])
            n_slice = pydub_to_ndarray(nxt[:ms_trans])

            sync_nudge = 0
            try:
                from scipy.signal import (
                    butter as _sb,
                    find_peaks as _sfp,
                    lfilter as _slf,
                )

                _sb_b, _sb_a = _sb(2, 80.0 / (0.5 * sr), btype="low")
                _min_dist = int(0.6 * 60 / t_s_bpm * sr)

                _m_mono = np.mean(m_slice, axis=0) if m_slice.ndim == 2 else m_slice
                _m_kick = np.abs(_slf(_sb_b, _sb_a, _m_mono))
                _m_max = np.max(_m_kick)
                _m_peaks = (
                    _sfp(_m_kick / _m_max, height=0.15, distance=_min_dist)[0]
                    if _m_max > 1e-6
                    else np.array([])
                )

                _n_mono = np.mean(n_slice, axis=0) if n_slice.ndim == 2 else n_slice
                _n_kick = np.abs(_slf(_sb_b, _sb_a, _n_mono))
                _n_max = np.max(_n_kick)
                _n_peaks = (
                    _sfp(_n_kick / _n_max, height=0.15, distance=_min_dist)[0]
                    if _n_max > 1e-6
                    else np.array([])
                )

                if len(_m_peaks) >= 3 and len(_n_peaks) >= 3:
                    _offsets = []
                    _max_dist = int(0.45 * 60 / t_s_bpm * sr)
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
                    sync_nudge = find_sync_offset(m_slice, n_slice, sr, t_s_bpm)
            except Exception:
                sync_nudge = find_sync_offset(m_slice, n_slice, sr, t_s_bpm)

            max_nudge = int(ms_per_beat * 2.0)
            sync_nudge = max(-max_nudge, min(max_nudge, sync_nudge))
            ms_trans -= sync_nudge
            print(f" [SYNC] Nudge: {sync_nudge}ms")

        ms_trans = max(500, min(ms_trans, len(master) - 1, len(nxt) - 1))
        track_start_ms = len(master) - ms_trans

        track_meta = {
            "timestamp": ms_to_timestamp(track_start_ms),
            "file": os.path.basename(all_files[i]),
            "bpm": round(float(meta_list[i]["bpm"]), 1),
            "key": f"{meta_list[i]['key']} ({get_camelot_key(meta_list[i]['key'])})",
            "genre": meta_list[i]["genre"],
            "rationale": meta_list[i].get("rationale", ""),
            "terrain": meta_list[i].get("terrain", []),
            "start_ms": track_start_ms,
            "overlap_ms": int(ms_trans),
        }
        tracklist.append(track_meta)
        _emit_log(
            status_obj,
            f"[MIX] #{i + 1} @ {track_meta['timestamp']} -> {track_meta['file']} ({track_meta['key']})",
        )

        for tool in active_tools:
            tool.on_track_start(track_meta, status_obj=status_obj)

        if status_obj:
            status_obj["tracklist"] = tracklist
            status_obj["progress"] = 75 + int((i / max(num_tracks - 1, 1)) * 22)

        # Two-Phase Overlap with Continuous Equal-Power Volume Curves
        pre_fade_ms = max(0, ms_trans - crossfade_ms)
        if pre_fade_ms > 0:
            print(f" [PREFADE] {pre_fade_ms}ms ambient intro under outgoing track")

        m_body = master[:-ms_trans]
        m_overlap = pydub_to_ndarray(master[-ms_trans:])
        n_intro_full = pydub_to_ndarray(nxt[:ms_trans])
        n_body = nxt[ms_trans:]

        m_len = m_overlap.shape[1] if m_overlap.ndim == 2 else len(m_overlap)
        n_len = n_intro_full.shape[1] if n_intro_full.ndim == 2 else len(n_intro_full)
        min_len = min(m_len, n_len)
        if m_overlap.ndim == 2:
            m_overlap = m_overlap[:, :min_len]
            n_intro_full = n_intro_full[:, :min_len]
        else:
            m_overlap = m_overlap[:min_len]
            n_intro_full = n_intro_full[:min_len]

        total_samples = min_len
        if pre_fade_ms > 0 and total_samples > int(crossfade_ms * sr / 1000):
            pf_samples = min(int(pre_fade_ms * sr / 1000), total_samples)
            cf_len = total_samples - pf_samples
            cf_x = np.linspace(0, 1, max(cf_len, 1))
            pf_x = np.linspace(0, 1, max(pf_samples, 1))

            out_curve = np.ones(total_samples)
            out_curve[pf_samples:] = np.cos(np.pi * cf_x / 2)

            in_curve = np.zeros(total_samples)
            prefade_vol = 0.25
            in_curve[:pf_samples] = prefade_vol * pf_x
            cf_in_raw = np.sin(np.pi * cf_x / 2)
            in_curve[pf_samples:] = prefade_vol + (1.0 - prefade_vol) * cf_in_raw
        else:
            x = np.linspace(0, 1, total_samples)
            out_curve = np.cos(np.pi * x / 2)
            in_curve = np.sin(np.pi * x / 2)

        mode = getattr(args, "archetype", "auto") or "auto"
        bass_style = getattr(
            args,
            "bass_transition",
            getattr(config, "BASS_TRANSITION_STYLE", "swap"),
        )
        highpass_hz = float(getattr(args, "highpass", config.HIGHPASS_CUTOFF) or 150.0)
        lowpass_hz = float(getattr(args, "lowpass", config.LOWPASS_CUTOFF) or 200.0)

        pf_s = (
            min(int(pre_fade_ms * sr / 1000), total_samples) if pre_fade_ms > 0 else 0
        )
        cf_len = total_samples - pf_s

        # Explicit specialized archetype plugin path vs zero-phase progressive bass management
        if mode not in ("auto", "progressive") and ArchetypeRegistry.get(mode):
            arch_cls = ArchetypeRegistry.get(mode)
            m_proc = np.copy(m_overlap)
            n_proc = np.copy(n_intro_full)
            if cf_len > 0:
                m_cf = m_overlap[:, pf_s:] if m_overlap.ndim == 2 else m_overlap[pf_s:]
                n_cf = (
                    n_intro_full[:, pf_s:]
                    if n_intro_full.ndim == 2
                    else n_intro_full[pf_s:]
                )
                fm_cf, fn_cf = arch_cls.apply(
                    m_cf,
                    n_cf,
                    sr,
                    bass_transition=bass_style,
                    highpass=highpass_hz,
                    lowpass=lowpass_hz,
                    bpm=t_s_bpm,
                    ideal_p=ideal_p,
                )
                if m_proc.ndim == 2:
                    m_proc[:, pf_s:] = fm_cf[:, :cf_len]
                    n_proc[:, pf_s:] = fn_cf[:, :cf_len]
                else:
                    m_proc[pf_s:] = fm_cf[:cf_len]
                    n_proc[pf_s:] = fn_cf[:cf_len]
        else:
            from scipy.signal import butter, filtfilt

            b_lo, a_lo = butter(2, highpass_hz / (0.5 * sr), btype="low")

            def _get_bass(audio):
                if audio.ndim == 2:
                    bass = np.zeros_like(audio)
                    for ch in range(audio.shape[0]):
                        bass[ch] = filtfilt(b_lo, a_lo, audio[ch])
                    return bass
                return filtfilt(b_lo, a_lo, audio)

            m_bass = _get_bass(m_overlap)
            n_bass = _get_bass(n_intro_full)

            m_bass_curve = np.ones(total_samples)
            n_bass_curve = np.zeros(total_samples)
            if cf_len > 0:
                cf_x = np.linspace(0, 1, cf_len)
                if bass_style == "fade":
                    m_bass_curve[pf_s:] = np.cos(np.pi * cf_x / 2)
                    n_bass_curve[pf_s:] = np.sin(np.pi * cf_x / 2)
                else:
                    m_cf_curve = np.ones(cf_len)
                    mask_out = cf_x > 0.4
                    m_cf_curve[mask_out] = 0.5 * (
                        1 + np.cos(np.pi * (cf_x[mask_out] - 0.4) / 0.4)
                    )
                    m_cf_curve[cf_x > 0.8] = 0.02
                    m_bass_curve[pf_s:] = m_cf_curve

                    n_cf_curve = np.zeros(cf_len)
                    mask_in = cf_x > 0.3
                    n_cf_curve[mask_in] = 0.5 * (
                        1 - np.cos(np.pi * (cf_x[mask_in] - 0.3) / 0.4)
                    )
                    n_cf_curve[cf_x > 0.7] = 1.0
                    n_bass_curve[pf_s:] = n_cf_curve

            if m_overlap.ndim == 2:
                m_proc = m_overlap - m_bass * (1 - m_bass_curve[np.newaxis, :])
                n_proc = n_intro_full - n_bass * (1 - n_bass_curve[np.newaxis, :])
            else:
                m_proc = m_overlap - m_bass * (1 - m_bass_curve)
                n_proc = n_intro_full - n_bass * (1 - n_bass_curve)

            # Optional Adaptive Spectral Balancing high-band anti-harshness check
            if getattr(args, "adaptive_spectral_balancing", False) and cf_len > 64:
                clashes = calculate_spectral_clash(m_proc, n_proc, sr)
                if clashes["high"] > 1.75:
                    b_hi, a_hi = butter(2, min(6500.0, 0.45 * sr) / (0.5 * sr), btype="low")
                    if m_proc.ndim == 2:
                        for ch in range(m_proc.shape[0]):
                            m_proc[ch, pf_s:] = 0.7 * m_proc[ch, pf_s:] + 0.3 * filtfilt(
                                b_hi, a_hi, m_proc[ch, pf_s:]
                            )
                    else:
                        m_proc[pf_s:] = 0.7 * m_proc[pf_s:] + 0.3 * filtfilt(
                            b_hi, a_hi, m_proc[pf_s:]
                        )

        # Safeguard sample count match
        for _n in range(2):
            arr = m_proc if _n == 0 else n_proc
            proc_len = arr.shape[1] if arr.ndim == 2 else len(arr)
            if proc_len != total_samples:
                if proc_len < total_samples:
                    pad = total_samples - proc_len
                    arr = (
                        np.pad(arr, ((0, 0), (0, pad)))
                        if arr.ndim == 2
                        else np.pad(arr, (0, pad))
                    )
                else:
                    arr = (
                        arr[:, :total_samples] if arr.ndim == 2 else arr[:total_samples]
                    )
            if _n == 0:
                m_proc = arr
            else:
                n_proc = arr

        if m_proc.ndim == 2:
            mix_bus_raw = (
                m_proc * out_curve[np.newaxis, :] + n_proc * in_curve[np.newaxis, :]
            )
        else:
            mix_bus_raw = m_proc * out_curve + n_proc * in_curve
        mix_bus_raw = apply_limiter(mix_bus_raw)

        expected_samples = int(ms_trans * sr / 1000)
        actual_samples = (
            mix_bus_raw.shape[1] if mix_bus_raw.ndim == 2 else len(mix_bus_raw)
        )
        if actual_samples < expected_samples:
            pad_len = expected_samples - actual_samples
            mix_bus_raw = (
                np.pad(mix_bus_raw, ((0, 0), (0, pad_len)), mode="constant")
                if mix_bus_raw.ndim == 2
                else np.pad(mix_bus_raw, (0, pad_len), mode="constant")
            )
        elif actual_samples > expected_samples:
            mix_bus_raw = (
                mix_bus_raw[:, :expected_samples]
                if mix_bus_raw.ndim == 2
                else mix_bus_raw[:expected_samples]
            )

        mix_bus = ndarray_to_pydub(mix_bus_raw, sr)
        master = m_body + mix_bus + n_body
        current_time_ms = len(master)

        if (
            status_obj
            and status_obj.get("live_params", {}).get("continuous_mode")
            and i > 3
        ):
            if i - 3 < len(warped_results):
                warped_results[i - 3] = None
            if i - 3 < len(processed_tracks):
                processed_tracks[i - 3] = None

    monitor.record_task_duration("mixing", round(time.perf_counter() - t_phase3, 3))

    # ── Final Mastering & Sink Export ──
    if master is not None:
        _emit_log(status_obj, "[MASTERING] Finalizing multiband dynamics & LUFS loudness...")
        mix_sr = master.frame_rate
        mix_raw = pydub_to_ndarray(master)

        mast_intensity = float(
            (status_obj or {})
            .get("live_params", {})
            .get("mastering_intensity", getattr(args, "mastering_intensity", 0.0))
        )
        if mast_intensity > 0.01:
            dominant_genre = None
            if getattr(args, "genre_aware_mastering", True) and tracklist:
                counts = Counter(t.get("genre", "Progressive") for t in tracklist)
                dominant_genre = counts.most_common(1)[0][0]
            if getattr(args, "dynamic_energy_mastering", True):
                rms = float(np.sqrt(np.mean(mix_raw**2)) + 1e-9)
                # Slightly gentler compression on already dense mixes
                mast_intensity = float(np.clip(mast_intensity * (0.85 if rms > 0.22 else 1.0), 0.0, 1.0))
            mix_raw = apply_multiband_compression(
                mix_raw, mix_sr, intensity=mast_intensity, genre_profile=dominant_genre
            )

        mix_raw = normalize_lufs(mix_raw, mix_sr, target_lufs)
        mix_raw = apply_limiter(mix_raw)
        master = ndarray_to_pydub(mix_raw, mix_sr)
        _emit_log(status_obj, f"[MASTERING] Normalized to {target_lufs:.1f} LUFS.")

        output_type = getattr(args, "output_plugin", "local_file")
        if getattr(args, "broadcast_mode", False) and getattr(args, "stream_url", None):
            output_type = "icecast_stream"
        output_cls = PluginRegistry.get_outputs().get(output_type)

        if status_obj:
            status_obj["status"] = "Exporting FLAC"
            status_obj["progress"] = 98

        try:
            if output_cls:
                sink = output_cls()
                sink.export(
                    master,
                    tracklist,
                    output=args.output,
                    version=__version__,
                    all_files=all_files,
                    meta_list=meta_list,
                    processed_tracks=processed_tracks,
                    export_mp3=bool(getattr(args, "export_mp3", False)),
                    stream_url=getattr(args, "stream_url", None),
                )
            else:
                master.export(args.output, format="flac")

            if status_obj:
                status_obj["tracklist"] = tracklist
                status_obj["status"] = "Complete"
                status_obj["progress"] = 100
            _emit_log(status_obj, f"[*] Master mix complete -> {args.output}")
            for tool in active_tools:
                tool.post_mix(status_obj=status_obj)
        except Exception as e:
            import traceback

            traceback.print_exc()
            monitor.log_incident(
                "ERROR", "Export", f"Export failed: {e}", traceback.format_exc()
            )
            if status_obj:
                status_obj["status"] = f"Error: Export failed - {e}"
