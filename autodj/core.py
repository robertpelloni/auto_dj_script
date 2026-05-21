""" Core Orchestration Engine | Auto DJ Script (v5.5.0)
==================================================
The core engine is responsible for tracklist optimization (Simulated Annealing),
parallel audio preprocessing, and the final sample-accurate mix reconstruction.

Version 5.8.x features: 3-band mastering and enhanced genre detection.
"""

import os, glob, re, librosa, random, json, subprocess
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import config

from .analysis import (
    get_native_bpm, get_musical_key, analyze_geometry,
    get_camelot_key, is_harmonically_compatible,
    get_energy_profile, detect_phrases, get_genre_archetype
)
from .dsp import (
    apply_dsp_filter, trim_silence, normalize_lufs,
    apply_bass_swap, apply_echo_out, apply_hpf_sweep,
    apply_limiter, apply_multiband_compression
)
from .utils import pydub_to_ndarray, ndarray_to_pydub
from .version import __version__


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
    """Phase Vocoder based time-stretching with intra-track BPM ramping."""
    if abs(start_target_bpm - end_target_bpm) < 0.01:
        return librosa.effects.time_stretch(y, rate=start_target_bpm / native_bpm)
    duration_sec = len(y) / sr
    num_chunks = max(1, int(duration_sec))
    chunks = np.array_split(y, num_chunks)
    warped_chunks = []
    for i, chunk in enumerate(chunks):
        target_bpm = start_target_bpm + (end_target_bpm - start_bpm) * (i / num_chunks)
        rate = target_bpm / native_bpm
        warped_chunks.append(librosa.effects.time_stretch(chunk, rate=rate))
    return np.concatenate(warped_chunks)


def warp_worker(args):
    """Thread worker for track preparation (time-stretch, pitch-shift, normalize)."""
    path, native_bpm, s_bpm, e_bpm, cur_key, tar_key, sync = args
    try:
        y, sr = librosa.load(path, sr=None)
        y_w = dynamic_warp(y, sr, native_bpm, s_bpm, e_bpm)
        if sync and cur_key and tar_key:
            diff = get_semitone_diff(cur_key, tar_key)
            if 0 < abs(diff) <= 2:
                y_w = librosa.effects.pitch_shift(y_w, sr=sr, n_steps=diff)
        y_w = apply_limiter(normalize_lufs(y_w, sr, config.TARGET_LUFS))
        return y_w, sr
    except Exception as e:
        print(f"[ERROR] warp_worker failed for {path}: {e}")
        return None, str(e)


def analyze_track_worker(f):
    """Metadata extraction worker."""
    try:
        native_bpm, y, sr = get_native_bpm(f)
        return {
            'path': f,
            'bpm': native_bpm,
            'key': get_musical_key(y, sr),
            'energy': get_energy_profile(y, sr),
            'genre': get_genre_archetype(y, sr)
        }
    except Exception as e:
        print(f"[ERROR] analyze_track_worker failed for {f}: {e}")
        return {'path': f, 'error': str(e)}


def find_optimal_order(files, status_obj=None):
    """Sequencing optimization via Simulated Annealing."""
    total = len(files)
    print(f"[*] Analyzing {total} tracks...")

    # Run analysis sequentially (librosa is GIL-bound, threading doesn't help)
    results = []
    for i, f in enumerate(files):
        r = analyze_track_worker(f)
        results.append(r)
        if status_obj is not None:
            status_obj["status"] = f"Analyzing Library ({i+1}/{total})"
            status_obj["progress"] = int((i / total) * 50)  # Analysis = 0-50%
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
    """The High-Performance Mixing Pipeline (v5.8.0)."""
    folder = args.input
    all_files = []
    for ext in config.SUPPORTED_EXTENSIONS:
        all_files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        all_files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))

    if not all_files:
        if status_obj:
            status_obj["status"] = "Error: No audio files found"
        print("[ERROR] No audio files found in input folder.")
        return

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

    start_bpm, end_bpm = args.bpm, (args.end_bpm or args.bpm)

    warp_tasks = []
    for i in range(num_tracks):
        t_s_bpm = start_bpm + (end_bpm - start_bpm) * (i / num_tracks)
        t_e_bpm = start_bpm + (end_bpm - start_bpm) * ((i + 1) / num_tracks)
        tar_key = meta_list[i-1]['key'] if i > 0 else None
        warp_tasks.append((all_files[i], meta_list[i]['bpm'], t_s_bpm, t_e_bpm, meta_list[i]['key'], tar_key, True))

    warped_results = []
    for i, task in enumerate(warp_tasks):
        result = warp_worker(task)
        warped_results.append(result)
        if status_obj is not None:
            status_obj["status"] = f"Warping track {i+1}/{num_tracks}"
            status_obj["progress"] = 50 + int((i / num_tracks) * 25)

    # Phase 3: Mixing (75-100%)
    if status_obj:
        status_obj["status"] = "Mixing Master Stream"

    tracklist, master, current_time_ms = [], None, 0
    for i in range(num_tracks):
        y_w, sr = warped_results[i]
        if y_w is None:
            print(f"[WARN] Skipping track {i}: warp failed")
            continue

        path = f"scratch_{i}_{os.getpid()}.wav"
        sf.write(path, y_w.T, sr, format='WAV', subtype='PCM_16')
        nxt = trim_silence(AudioSegment.from_wav(path))
        os.remove(path)
        processed_tracks.append((nxt, y_w, sr))

        if master is None:
            master = nxt
            tracklist.append({'timestamp': "00:00:00", 'file': os.path.basename(all_files[i]),
                               'key': f"{meta_list[i]['key']} ({get_camelot_key(meta_list[i]['key'])})",
                               'genre': meta_list[i]['genre'], 'start_ms': 0})
            current_time_ms = len(master)
            continue

        prev_nxt, prev_y_w, _ = processed_tracks[i-1]
        t_s_bpm = start_bpm + (end_bpm - start_bpm) * (i / num_tracks)
        beats, ms_trans = analyze_geometry(nxt, sr, t_s_bpm, args.beats_per_bar, args.transition_bars)
        ph = detect_phrases(y_w, sr)

        fixed_p = beats[min(args.transition_bars * args.beats_per_bar, len(beats)-1)]
        ideal_p = fixed_p
        if ph.any():
            cl = ph[np.argmin(np.abs(ph - fixed_p))]
            if abs(cl - fixed_p) < config.PHRASE_ANCHOR_TOLERANCE_MS:
                ideal_p = cl

        # Intelligent Tail Extension (v6.6.0)
        ext_rationale = ""
        if ms_trans > len(prev_nxt):
            loop_bar = identify_loopable_phrase(prev_y_w, sr, t_s_bpm, args.beats_per_bar)
            num_loops = int(np.ceil((ms_trans - len(prev_nxt)) / (len(loop_bar) / sr * 1000))) + 1
            ext_segment = np.tile(loop_bar, num_loops)
            prev_nxt += ndarray_to_pydub(ext_segment, sr)
            ext_rationale = " [Loop-Extended]"

        ms_trans = min(ms_trans, len(prev_nxt))
        track_start_ms = current_time_ms - ms_trans

        tracklist.append({'timestamp': ms_to_timestamp(track_start_ms), 'file': os.path.basename(all_files[i]),
                           'key': f"{meta_list[i]['key']} ({get_camelot_key(meta_list[i]['key'])})",
                           'genre': meta_list[i]['genre'], 'start_ms': track_start_ms})

        if status_obj:
            status_obj["tracklist"] = tracklist
            status_obj["progress"] = 75 + int((i / (num_tracks-1)) * 25)

        m_body, m_outro = master[:-ms_trans], master[-ms_trans:]
        n_intro, n_body = nxt[:ideal_p], nxt[ideal_p:]

        # Archetype Selection Logic
        mode = getattr(args, 'archetype', 'auto')
        if mode == 'bass_swap' or (mode == 'auto' and meta_list[i]['genre'] == 'High-Energy'):
            f_m_h, f_n_l = apply_bass_swap(pydub_to_ndarray(m_outro), pydub_to_ndarray(n_intro), sr)
            f_m, f_n = ndarray_to_pydub(f_m_h, sr), ndarray_to_pydub(f_n_l, sr)
        elif mode == 'echo_out':
            f_m = ndarray_to_pydub(apply_echo_out(pydub_to_ndarray(m_outro), sr), sr)
            f_n = ndarray_to_pydub(pydub_to_ndarray(n_intro), sr)
        elif mode == 'hpf_sweep':
            f_m = ndarray_to_pydub(apply_hpf_sweep(pydub_to_ndarray(m_outro), sr), sr)
            f_n = ndarray_to_pydub(pydub_to_ndarray(n_intro), sr)
        else:  # Classic Fade
            f_m = ndarray_to_pydub(apply_dsp_filter(pydub_to_ndarray(m_outro), sr, 'lowpass', args.lowpass), sr)
            f_n = ndarray_to_pydub(apply_dsp_filter(pydub_to_ndarray(n_intro), sr, 'highpass', args.highpass), sr)

        master = m_body + f_m.fade_out(ms_trans).overlay(f_n.fade_in(ms_trans)) + n_body
        current_time_ms = len(master)

    if master:
        try:
            # Export directly — skip multiband compression on the full mix
            # (too memory/CPU intensive for large sets; per-track limiting already applied)
            if status_obj:
                status_obj["status"] = "Exporting FLAC"
                status_obj["progress"] = 98
            print(f"[*] Exporting {len(master)/1000/60:.1f} min mix to {args.output}...")
            master.export(args.output, format="flac")
            if status_obj:
                status_obj["status"] = "Complete"
                status_obj["progress"] = 100
            print(f"[*] Export complete!")
        except Exception as e:
            print(f"[ERROR] Export failed: {e}")
            if status_obj:
                status_obj["status"] = f"Error: Export failed - {e}"
            return

        tl_path = os.path.splitext(args.output)[0] + "_tracklist.txt"
        with open(tl_path, "w") as f:
            f.write(f"Auto DJ v{__version__} Master Tracklist\n{'='*40}\n")
            for item in tracklist:
                f.write(f"[{item['timestamp']}] {item['file']} ({item['key']}) [{item['genre']}]\n")


def transition_render_worker(args):
    """Parallel worker for rendering a single transition overlap."""
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
            f_m, f_n = ndarray_to_pydub(f_m_raw, sr), ndarray_to_pydub(f_n_raw, sr)
        else:
            f_m, f_n = ndarray_to_pydub(outro_raw, sr), ndarray_to_pydub(intro_raw, sr)

        # Apply fades and overlay
        rendered = f_m.fade_out(ms_trans).overlay(f_n.fade_in(ms_trans))
        return pydub_to_ndarray(rendered), sr
    except Exception as e:
        return None, str(e)

def ms_to_timestamp(ms):
    s = int((ms / 1000) % 60)
    m = int((ms / (1000 * 60)) % 60)
    h = int((ms / (1000 * 60 * 60)) % 24)
    return f"{h:02d}:{m:02d}:{s:02d}"
