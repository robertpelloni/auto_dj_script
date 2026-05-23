""" Core Orchestration Engine | Auto DJ Script (7.6.0)
==================================================
The core engine is responsible for tracklist optimization (Simulated Annealing),
parallel audio preprocessing, and the final sample-accurate mix reconstruction.

Version 7.6.0 features: The Visual Era (Spectral Terrain 3D).
"""

import os, glob, re, librosa, random, json, subprocess, io
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import config

from .analysis import (
    get_native_bpm, get_musical_key, analyze_geometry,
    get_camelot_key, is_harmonically_compatible,
    get_energy_profile, detect_phrases, get_genre_archetype,
    find_sync_offset, identify_loopable_phrase, extract_spectral_terrain
)
from .dsp import (
    apply_dsp_filter, trim_silence, normalize_lufs,
    apply_bass_swap, apply_echo_out, apply_hpf_sweep,
    apply_limiter, apply_multiband_compression,
    apply_log_fade, ArchetypeRegistry
)
from .utils import pydub_to_ndarray, ndarray_to_pydub, export_rekordbox_xml
from .version import __version__
from .cluster import cluster
from .monitoring import monitor
from .plugins import PluginRegistry
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

        # Update status message to reflect guardrail state
        orig_status = status_obj.get("status", "Processing")
        if paused:
            status_obj["status"] = "Session Paused (Manual)"
        elif not is_healthy:
            status_obj["status"] = "Auto-Throttled (High System Load)"

        time.sleep(1)
        # Restore status after waiting
        status_obj["status"] = orig_status


def get_semitone_diff(key1, key2):
    """Calculates the distance in semitones between two musical keys."""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
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


def dynamic_warp(y, sr, native_bpm, start_target_bpm, end_target_bpm):
    """Rubber Band based high-fidelity time-stretching."""
    def rb_stretch(data, rate):
        import tempfile, subprocess
        # Strict Stereo Check
        if data.ndim == 1:
            data = np.vstack([data, data])
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fin, \
             tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fout:
            sf.write(fin.name, data.T, sr, format='WAV', subtype='PCM_16')
            fin.close(); fout.close()
            # Correct Ratio: rate is target/native. 
            # If target > native, rate > 1.0 (speed up).
            # Rubber Band --tempo X: X > 1.0 is faster.
            print(f"    [RB] rate={rate:.4f} in={fin.name}")
            subprocess.run(["rubberband", "--tempo", str(rate), fin.name, fout.name], check=True)
            out_y, _ = librosa.load(fout.name, sr=sr, mono=False)
            os.remove(fin.name); os.remove(fout.name)
            
            if out_y.ndim == 1:
                out_y = np.vstack([out_y, out_y])
            return out_y

    # If the BPM difference is small, process the entire track in one pass to avoid "muddiness"
    # muddiness is often caused by splitting audio into chunks without crossfading.
    avg_target_bpm = (start_target_bpm + end_target_bpm) / 2.0
    rate = avg_target_bpm / native_bpm
    print(f"  [WARP] Channels: {y.shape[0]}, Rate: {rate:.4f} ({native_bpm:.1f} -> {avg_target_bpm:.1f})")
    return rb_stretch(y, rate)


def warp_worker(args):
    """Thread worker for track preparation (time-stretch, pitch-shift, normalize)."""
    path, native_bpm, s_bpm, e_bpm, cur_key, tar_key, sync = args
    try:
        # Use soundfile.read for much faster loading
        target_sr = 44100
        y, sr = sf.read(path, dtype='float32')
        # Soundfile returns (samples, channels). We need (channels, samples)
        if y.ndim == 2:
            y = y.T
        
        # Ensure sample rate matches or resample if necessary
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
            
        print(f"  [LOAD] {os.path.basename(path)} - Fast-Loaded via SoundFile")
        
        y_w = dynamic_warp(y, sr, native_bpm, s_bpm, e_bpm)
        
        y_w = apply_limiter(normalize_lufs(y_w, sr, config.TARGET_LUFS))
        return y_w, sr
    except Exception as e:
        import traceback
        monitor.log_incident("ERROR", "WarpWorker", f"Failed to warp {os.path.basename(path)}: {e}", traceback.format_exc())
        return None, str(e)


def analyze_track_worker(f):
    """Metadata extraction worker with fast-loading."""
    try:
        target_sr = 44100
        y, sr = sf.read(f, dtype='float32')
        if y.ndim == 2:
            y = y.T
            
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
            
        native_bpm, _, _ = get_native_bpm(y, sr)
        
        genre, rationale = get_genre_archetype(y if y.ndim == 1 else librosa.to_mono(y), sr, bpm=native_bpm)
        terrain = extract_spectral_terrain(y, sr)

        return {
            'path': f,
            'bpm': native_bpm,
            'key': get_musical_key(y if y.ndim == 1 else librosa.to_mono(y), sr),
            'energy': get_energy_profile(y if y.ndim == 1 else librosa.to_mono(y), sr),
            'genre': get_genre_archetype(y if y.ndim == 1 else librosa.to_mono(y), sr)
        }
    except Exception as e:
        print(f"[ERROR] analyze_track_worker failed for {f}: {e}")
        return {'path': f, 'error': str(e)}


def find_optimal_order(files, status_obj=None):
    """Sequencing optimization via Simulated Annealing."""
    total = len(files)
    print(f"[*] Analyzing {total} tracks (Sequential)...")

    results = []
    for i, f in enumerate(files):
        if status_obj:
            status_obj["active_tasks"][f] = "Analyzing..."
        
        r = analyze_track_worker(f)
        results.append(r)
        
        if status_obj:
            status_obj["active_tasks"].pop(f, None)
            status_obj["status"] = f"Analyzing Library ({i+1}/{total})"
            status_obj["progress"] = int(((i + 1) / total) * 50)
            
        if 'error' not in r:
            print(f"  [{i+1}/{total}] {os.path.basename(f)}: BPM={r['bpm']:.1f}, Key={r['key']}, Genre={r['genre']}")
        else:
            print(f"  [{i+1}/{total}] {os.path.basename(f)}: ERROR - {r['error']}")

    meta = [r for r in results if 'error' not in r]
    if not meta:
        return files, None

    def score_transition(t1, t2):
        s = 50 if is_harmonically_compatible(t1['key'], t2['key']) else 0
        if abs(get_semitone_diff(t1['key'], t2['key'])) <= 2:
            s += 25
        e_diff = t2['energy'] - t1['energy']
        s += 20 if e_diff > 0 else 0
        s -= abs(e_diff) * 100
        if t1['genre'] == t2['genre']:
            s += 30
        return s

    order = [meta.pop(0)]
    while meta:
        best_idx = np.argmax([score_transition(order[-1], m) for m in meta])
        order.append(meta.pop(best_idx))

    def score_set(o):
        return sum(score_transition(o[i], o[i+1]) for i in range(len(o)-1))

    best_o, best_s = list(order), score_set(order)
    temp = config.SA_INITIAL_TEMP
    for _ in range(config.SA_ITERATIONS):
        if len(best_o) < 3:
            break
        new_o = list(best_o)
        i, j = random.sample(range(1, len(new_o)), 2)
        new_o[i], new_o[j] = new_o[j], new_o[i]
        new_s = score_set(new_o)
        if new_s > best_s or random.random() < np.exp(min(700, (new_s - best_s)/temp)):
            best_o, best_s = new_o, new_s
        temp = config.SA_INITIAL_TEMP / np.log(1 + _ + 1)

    if status_obj is not None:
        status_obj["status"] = "Optimizing track order..."
        status_obj["progress"] = 50

    return [x['path'] for x in best_o], best_o


def compile_master_set(args, status_obj=None):
    """The High-Performance Mixing Pipeline (7.7.0)."""
    folder = args.input

    # Modular Source Discovery (v7.7.0)
    source_type = getattr(args, 'source_plugin', 'local_folder')
    source_cls = PluginRegistry.get_sources().get(source_type)

    # Modular Tool Loading
    active_tools = [cls() for name, cls in PluginRegistry.get_tools().items()]
    for tool in active_tools:
        tool.pre_mix(status_obj=status_obj, args=args)

    if status_obj and status_obj.get("playlist"):
        all_files = [os.path.join(folder, f) for f in status_obj["playlist"]]
    elif source_cls:
        source = source_cls()
        all_files = source.get_tracks(folder=folder, extensions=config.SUPPORTED_EXTENSIONS)
    else:
        all_files = []

    if not all_files:
        if status_obj:
            status_obj["status"] = "Error: No audio files found"
        print("[ERROR] No audio files found in input folder.")
        return

    # Deduplicate files (On Windows, globbing *.flac and *.FLAC returns same files)
    all_files = list(set(os.path.abspath(f) for f in all_files))
    
    # Phase 1: Analysis (0-50%)
    all_files, meta_list = find_optimal_order(all_files, status_obj=status_obj)

    if meta_list is None:
        if status_obj:
            status_obj["status"] = "Error: Analysis failed"
        return

    num_tracks = len(all_files)

    # Phase 2: Cluster-Accelerated Warping (50-75%)
    if status_obj:
        status_obj["status"] = f"Warping {num_tracks} tracks (Cluster: {cluster.nodes[0].id})"

    wait_for_health(status_obj)

    # Check for live parameter overrides (v7.1.0)
    start_bpm = status_obj.get("live_params", {}).get("target_bpm", args.bpm) if status_obj else args.bpm
    end_bpm = (args.end_bpm or start_bpm)

    warp_tasks = []
    for i in range(num_tracks):
        t_s_bpm = start_bpm + (end_bpm - start_bpm) * (i / num_tracks)
        t_e_bpm = start_bpm + (end_bpm - start_bpm) * ((i + 1) / num_tracks)
        tar_key = meta_list[i-1]['key'] if i > 0 else None
        warp_tasks.append((all_files[i], meta_list[i]['bpm'], t_s_bpm, t_e_bpm, meta_list[i]['key'], tar_key, True))

    warped_results = [None] * num_tracks
    executor = cluster.get_executor()
    futures = {executor.submit(warp_worker, task): i for i, task in enumerate(warp_tasks)}
    for future in as_completed(futures):
        idx = futures[future]
        if status_obj:
            status_obj["active_tasks"][all_files[idx]] = "Warping..."

        y_w, sr = future.result()

        if status_obj:
            status_obj["active_tasks"].pop(all_files[idx], None)

        # Fault Tolerance: Local Fallback (v7.4.0)
        if y_w is None:
            monitor.record_retry()
            monitor.log_incident("WARN", "CoreEngine", f"Cluster task {idx} failed. Retrying locally...")
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

    # Phase 3: Segmented Cluster Mixing (75-100%)
    if status_obj:
        status_obj["status"] = "Mixing Master Stream (Cluster)"

    tracklist, master, processed_tracks, current_time_ms = [], None, [], 0
    master_grid_offset = 0

    for i in range(num_tracks):
        y_w, sr = warped_results[i]
        if y_w is None:
            print(f"[WARN] Skipping track {i}: warp failed")
            continue

        # In-Memory Buffer Conversion (v7.0.0 Optimized)
        # Avoids disk I/O for temporary track preparation
        buf = io.BytesIO()
        sf.write(buf, y_w.T, sr, format='WAV', subtype='PCM_16')
        buf.seek(0)
        nxt = trim_silence(AudioSegment.from_file(buf, format="wav"))
        processed_tracks.append((nxt, y_w, sr))

        if master is None:
            master = nxt
            # Anchor the global grid to the first beat of the first track
            _, _, master_grid_offset = analyze_geometry(nxt, sr, start_bpm, args.beats_per_bar, args.transition_bars)
            tracklist.append({'timestamp': "00:00:00", 'file': os.path.basename(all_files[i]),
                               'key': f"{meta_list[i]['key']} ({get_camelot_key(meta_list[i]['key'])})",
                               'genre': meta_list[i]['genre'],
                               'rationale': meta_list[i].get('rationale', ''),
                               'terrain': meta_list[i].get('terrain', []),
                               'start_ms': 0})
            current_time_ms = len(master)
            i += 1
            continue

        prev_nxt, prev_y_w, _ = processed_tracks[i-1]

        # Poll live BPM for dynamic ramp adjustments (v7.1.0)
        current_target_bpm = status_obj.get("live_params", {}).get("target_bpm", start_bpm) if status_obj else start_bpm
        t_s_bpm = current_target_bpm + (end_bpm - current_target_bpm) * (i / num_tracks)

        beats, theoretical_ms_trans, first_beat_ms = analyze_geometry(nxt, sr, t_s_bpm, args.beats_per_bar, args.transition_bars)
        ph = detect_phrases(y_w, sr)

        # 1. Theoretical Transition Prep (v7.2.0)
        ms_per_beat = 60000.0 / t_s_bpm
        ms_per_bar = ms_per_beat * args.beats_per_bar
        grid_size = ms_per_bar * 8 # Switch to 8-bar phrasing (Standard Psy)
        
        fixed_p = beats[min(args.transition_bars * args.beats_per_bar, len(beats)-1)] if len(beats) > 0 else theoretical_ms_trans
        ideal_p = fixed_p
        if ph.any():
            cl = ph[np.argmin(np.abs(ph - fixed_p))]
            if abs(cl - fixed_p) < config.PHRASE_ANCHOR_TOLERANCE_MS:
                ideal_p = cl

        # Initial overlap
        ms_trans = max(ideal_p, first_beat_ms + int(ms_per_bar * 4))

        # 2. Intelligent Tail Extension
        if ms_trans > (len(master) - tracklist[-1]['start_ms']):
            loop_bar = identify_loopable_phrase(prev_y_w, sr, t_s_bpm, args.beats_per_bar)
            needed_ms = ms_trans - (len(master) - tracklist[-1]['start_ms'])
            num_loops = int(np.ceil(needed_ms / (len(loop_bar) / sr * 1000))) + 1
            ext_segment = np.tile(loop_bar, num_loops)
            master += ndarray_to_pydub(ext_segment, sr)
            current_time_ms = len(master)

        # 3. Final Precise Phase Alignment (Relative to First Kick of Mix)
        # Expected Kick = master_grid_offset + (N * grid_size)
        current_kick_pos = (current_time_ms - ms_trans + first_beat_ms)
        relative_pos = current_kick_pos - master_grid_offset
        phase_error = relative_pos % grid_size
        
        if phase_error != 0:
             ms_trans += int(phase_error)

        # 4. Sample-Accurate Nudging (High-Res Kick Alignment)
        m_slice = pydub_to_ndarray(master[-ms_trans:])
        n_slice = pydub_to_ndarray(nxt[:ms_trans])
        sync_nudge = find_sync_offset(m_slice, n_slice, sr, t_s_bpm)
        
        # Increase nudge limit to +/- 2 beats (approx 826ms @ 145BPM)
        # This allows the engine to jump over a kick if the initial phrasing was off.
        max_nudge = int(ms_per_beat * 2.0)
        sync_nudge = max(-max_nudge, min(max_nudge, sync_nudge))
        
        # CORRECT DIRECTION: Subtract nudge to align transients
        ms_trans -= sync_nudge

        print(f"  [SYNC] 8-Bar Phrase Locked. Offset: {master_grid_offset}ms. Nudge: {sync_nudge}ms. Final Overlap: {ms_trans/1000:.1f}s")

        ms_trans = min(ms_trans, len(master))
        track_start_ms = len(master) - ms_trans

        track_meta = {'timestamp': ms_to_timestamp(track_start_ms), 'file': os.path.basename(all_files[i]),
                       'key': f"{meta_list[i]['key']} ({get_camelot_key(meta_list[i]['key'])})",
                       'genre': meta_list[i]['genre'],
                       'rationale': meta_list[i].get('rationale', ''),
                       'terrain': meta_list[i].get('terrain', []),
                       'start_ms': track_start_ms}
        tracklist.append(track_meta)

        for tool in active_tools:
            tool.on_track_start(track_meta, status_obj=status_obj)

        if status_obj:
            status_obj["tracklist"] = tracklist
            status_obj["progress"] = 75 + int((i / (num_tracks-1)) * 25)

        # Gapless Slicing: Both tracks must be sliced using the EXACT same ms_trans
        m_body, m_outro = master[:-ms_trans], master[-ms_trans:]
        n_intro, n_body = nxt[:ms_trans], nxt[ms_trans:]

        # Archetype Selection Logic
        mode = getattr(args, 'archetype', 'auto')
        if mode == 'auto' and meta_list[i]['genre'] == 'High-Energy':
            mode = 'progressive' # Professional default for Psytrance

        # Parallel Transition Rendering (7.0.0)
        dsp_kwargs = {'lowpass': args.lowpass, 'highpass': args.highpass, 'ideal_p': ideal_p}
        render_args = (pydub_to_ndarray(m_outro), pydub_to_ndarray(n_intro), sr, mode, ms_trans, ideal_p, dsp_kwargs)

        # Using the cluster executor
        if status_obj:
            status_obj["active_tasks"][f"Transition {i-1}->{i}"] = "Mixing..."

        future = mix_executor.submit(transition_render_worker, render_args)
        mix_bus_raw, _ = future.result()

        if status_obj:
             status_obj["active_tasks"].pop(f"Transition {i-1}->{i}", None)

        # Fault Tolerance: Fallback to Sequential Render (v7.4.0)
        if mix_bus_raw is None:
            monitor.log_incident("WARN", "CoreEngine", f"Transition render {i} failed in cluster. Falling back...")
            mix_bus_raw, _ = transition_render_worker(render_args)

        if mix_bus_raw is not None:
            mix_bus = ndarray_to_pydub(mix_bus_raw, sr)
            monitor.record_success()
        else:
             monitor.record_failure()
             # Classic Fallback if parallel render fails
             f_m = ndarray_to_pydub(apply_dsp_filter(pydub_to_ndarray(m_outro), sr, 'lowpass', args.lowpass), sr)
             f_n = ndarray_to_pydub(apply_dsp_filter(pydub_to_ndarray(n_intro), sr, 'highpass', args.highpass), sr)
             mix_bus = f_m.fade_out(ms_trans).overlay(f_n.fade_in(ms_trans))

        master = m_body + mix_bus + n_body
        current_time_ms = len(master)
        i += 1

    if master:
        # Modular Output Export (v7.7.0)
        output_type = getattr(args, 'output_plugin', 'local_file')
        output_cls = PluginRegistry.get_outputs().get(output_type)

        if output_cls:
            if status_obj:
                status_obj["status"] = "Exporting via Modular Sink"
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
                    processed_tracks=processed_tracks
                )

                if status_obj:
                    status_obj["status"] = "Complete"
                    status_obj["progress"] = 100

                for tool in active_tools:
                    tool.post_mix(status_obj=status_obj)
            except Exception as e:
                print(f"[ERROR] Output plugin failed: {e}")
                if status_obj:
                    status_obj["status"] = f"Error: Output failed - {e}"
        else:
            print(f"[ERROR] Output plugin '{output_type}' not found.")
            if status_obj:
                status_obj["status"] = "Error: Output plugin not found"


def transition_render_worker(args):
    """Parallel worker for rendering a single transition overlap (7.0.0)."""
    outro_raw, intro_raw, sr, mode, ms_trans, ideal_p, dsp_kwargs = args
    try:
        arch_plugin = ArchetypeRegistry.get(mode)
        if arch_plugin:
            f_m_raw, f_n_raw = arch_plugin.apply(
                outro_raw,
                intro_raw,
                sr,
                **dsp_kwargs
            )
        else:
            # Classic Fallback Filters
            f_m_raw = apply_dsp_filter(outro_raw, sr, 'lowpass', dsp_kwargs.get('lowpass', 200.0))
            f_n_raw = apply_dsp_filter(intro_raw, sr, 'highpass', dsp_kwargs.get('highpass', 150.0))

        # Apply Professional Logarithmic Fades with Dip (7.0.0)
        f_m_faded = apply_log_fade(f_m_raw, fade_type='out')
        f_n_faded = apply_log_fade(f_n_raw, fade_type='in')

        # Precise Mix-Bus Summation
        summed = f_m_faded + f_n_faded

        # Safety: Apply Limiter to prevent digital clipping in the mix-bus
        summed = apply_limiter(summed)

        # Ensure correct shape (stereo) and duration
        return summed, sr
    except Exception as e:
        import traceback
        print(f"[ERROR] transition_render_worker failed: {e}")
        traceback.print_exc()
        return None, str(e)

def ms_to_timestamp(ms):
    s = int((ms / 1000) % 60)
    m = int((ms / (1000 * 60)) % 60)
    h = int((ms / (1000 * 60 * 60)) % 24)
    return f"{h:02d}:{m:02d}:{s:02d}"
