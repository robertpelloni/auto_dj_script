"""
Core Orchestration Engine | Auto DJ Script (v5.5.0)
==================================================

The core engine is responsible for tracklist optimization (Simulated Annealing),
parallel audio preprocessing, and the final sample-accurate mix reconstruction.

Version 5.5.x features: Manual Archetype Overrides.
"""

import os, glob, re, librosa, random, json
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import config
from .analysis import (get_native_bpm, get_musical_key, analyze_geometry,
                       get_camelot_key, is_harmonically_compatible,
                       get_energy_profile, detect_phrases, get_genre_archetype)
from .dsp import (apply_dsp_filter, trim_silence, normalize_lufs,
                  apply_bass_swap, apply_echo_out, apply_hpf_sweep, apply_limiter)
from .utils import pydub_to_ndarray, ndarray_to_pydub
from .version import __version__

def get_semitone_diff(key1, key2):
    """Calculates the distance in semitones between two musical keys."""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    try:
        p1, p2 = key1.split(), key2.split()
        if p1[1] != p2[1]: return 0
        diff = notes.index(p2[0]) - notes.index(p1[0])
        if diff > 6: diff -= 12
        if diff < -6: diff += 12
        return diff
    except: return 0

def dynamic_warp(y, sr, native_bpm, start_target_bpm, end_target_bpm):
    """Phase Vocoder based time-stretching with intra-track BPM ramping."""
    if abs(start_target_bpm - end_target_bpm) < 0.01:
        return librosa.effects.time_stretch(y, rate=start_target_bpm / native_bpm)

    duration_sec = len(y) / sr
    num_chunks = max(1, int(duration_sec))
    chunks = np.array_split(y, num_chunks)

    warped_chunks = []
    for i, chunk in enumerate(chunks):
        target_bpm = start_target_bpm + (end_target_bpm - start_target_bpm) * (i / num_chunks)
        rate = target_bpm / native_bpm
        warped_chunks.append(librosa.effects.time_stretch(chunk, rate=rate))

    return np.concatenate(warped_chunks)

def warp_worker(args):
    """Parallel worker process for track preparation."""
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
        return None, str(e)

def find_optimal_order(files):
    """Sequencing optimization via Simulated Annealing."""
    print(f"[*] Analyzing {len(files)} tracks in parallel...")
    with ProcessPoolExecutor() as executor:
        from .core import analyze_track_worker
        results = list(tqdm(executor.map(analyze_track_worker, files), total=len(files), desc="Analyzing", unit="track"))

    meta = [r for r in results if 'error' not in r]
    if not meta: return files, None

    def score_transition(t1, t2):
        s = 50 if is_harmonically_compatible(t1['key'], t2['key']) else 0
        if abs(get_semitone_diff(t1['key'], t2['key'])) <= 2: s += 25
        e_diff = t2['energy'] - t1['energy']
        s += 20 if e_diff > 0 else 0
        s -= abs(e_diff) * 100
        if t1['genre'] == t2['genre']: s += 30
        return s

    order = [meta.pop(0)]
    while meta:
        best_idx = np.argmax([score_transition(order[-1], m) for m in meta])
        order.append(meta.pop(best_idx))

    def score_set(o): return sum(score_transition(o[i], o[i+1]) for i in range(len(o)-1))

    best_o, best_s = list(order), score_set(order)
    temp = config.SA_INITIAL_TEMP
    for _ in range(config.SA_ITERATIONS):
        if len(best_o) < 3: break
        new_o = list(best_o)
        i, j = random.sample(range(1, len(new_o)), 2)
        new_o[i], new_o[j] = new_o[j], new_o[i]
        new_s = score_set(new_o)
        if new_s > best_s or random.random() < np.exp(min(700, (new_s - best_s)/temp)):
            best_o, best_s = new_o, new_s
        temp = config.SA_INITIAL_TEMP / np.log(1 + _ + 1)

    return [x['path'] for x in best_o], best_o

def analyze_track_worker(f):
    """Metadata extraction worker."""
    try:
        y, sr = librosa.load(f, sr=None)
        return {
            'path': f,
            'bpm': get_native_bpm(y, sr)[0],
            'key': get_musical_key(y, sr),
            'energy': get_energy_profile(y, sr),
            'genre': get_genre_archetype(y, sr)
        }
    except Exception as e: return {'path': f, 'error': str(e)}

def compile_master_set(args, status_obj=None):
    """The High-Performance Mixing Pipeline (v5.5.0)."""
    folder = args.input
    all_files = []
    for ext in config.SUPPORTED_EXTENSIONS:
        all_files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        all_files.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    if not all_files: return

    if status_obj: status_obj["status"] = "Analyzing Library"
    all_files, meta_list = find_optimal_order(all_files)
    num_tracks = len(all_files)

    if status_obj: status_obj["status"] = f"Warping {num_tracks} tracks"
    start_bpm, end_bpm = args.bpm, (args.end_bpm or args.bpm)
    warp_tasks = []
    for i in range(num_tracks):
        t_s_bpm = start_bpm + (end_bpm - start_bpm) * (i / num_tracks)
        t_e_bpm = start_bpm + (end_bpm - start_bpm) * ((i + 1) / num_tracks)
        tar_key = meta_list[i-1]['key'] if i > 0 else None
        warp_tasks.append((all_files[i], meta_list[i]['bpm'], t_s_bpm, t_e_bpm, meta_list[i]['key'], tar_key, True))

    with ProcessPoolExecutor() as executor:
        warped_results = list(tqdm(executor.map(warp_worker, warp_tasks), total=num_tracks, desc="Warping", unit="track"))

    if status_obj: status_obj["status"] = "Mixing Master Stream"
    tracklist, master, current_time_ms = [], None, 0

    for i in range(num_tracks):
        y_w, sr = warped_results[i]
        if y_w is None: continue

        path = f"scratch_{i}_{os.getpid()}.wav"
        sf.write(path, y_w.T, sr, format='WAV', subtype='PCM_16')
        nxt = trim_silence(AudioSegment.from_wav(path))
        os.remove(path)

        if master is None:
            master = nxt
            tracklist.append({'timestamp': "00:00:00", 'file': os.path.basename(all_files[i]), 'key': f"{meta_list[i]['key']} ({get_camelot_key(meta_list[i]['key'])})", 'genre': meta_list[i]['genre'], 'start_ms': 0})
            current_time_ms = len(master)
            continue

        t_s_bpm = start_bpm + (end_bpm - start_bpm) * (i / num_tracks)
        beats, ms_trans = analyze_geometry(nxt, sr, t_s_bpm, args.beats_per_bar, args.transition_bars)

        ph = detect_phrases(y_w, sr)
        fixed_p = beats[min(args.transition_bars * args.beats_per_bar, len(beats)-1)]
        ideal_p = fixed_p
        if ph.any():
            cl = ph[np.argmin(np.abs(ph - fixed_p))]
            if abs(cl - fixed_p) < config.PHRASE_ANCHOR_TOLERANCE_MS: ideal_p = cl

        ms_trans = min(ms_trans, len(master))
        track_start_ms = current_time_ms - ms_trans

        tracklist.append({'timestamp': ms_to_timestamp(track_start_ms), 'file': os.path.basename(all_files[i]), 'key': f"{meta_list[i]['key']} ({get_camelot_key(meta_list[i]['key'])})", 'genre': meta_list[i]['genre'], 'start_ms': track_start_ms})

        if status_obj:
            status_obj["tracklist"] = tracklist
            status_obj["progress"] = int((i / (num_tracks-1)) * 100)

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
        else: # Classic Fade
            f_m = ndarray_to_pydub(apply_dsp_filter(pydub_to_ndarray(m_outro), sr, 'lowpass', args.lowpass), sr)
            f_n = ndarray_to_pydub(apply_dsp_filter(pydub_to_ndarray(n_intro), sr, 'highpass', args.highpass), sr)

        master = m_body + f_m.fade_out(ms_trans).overlay(f_n.fade_in(ms_trans)) + n_body
        current_time_ms = len(master)

    if master:
        master.export(args.output, format="flac")
        if status_obj: status_obj["status"] = "Complete"
        tl_path = os.path.splitext(args.output)[0] + "_tracklist.txt"
        with open(tl_path, "w") as f:
            f.write(f"Auto DJ v{__version__} Master Tracklist\n{'='*40}\n")
            for item in tracklist: f.write(f"[{item['timestamp']}] {item['file']} ({item['key']}) [{item['genre']}]\n")

def ms_to_timestamp(ms):
    s = int((ms / 1000) % 60)
    m = int((ms / (1000 * 60)) % 60)
    h = int((ms / (1000 * 60 * 60)) % 24)
    return f"{h:02d}:{m:02d}:{s:02d}"
