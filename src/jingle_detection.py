"""
jingle_detection.py
Detects occurrences of a reference jingle in audio chunks.
"""
import os
import librosa
import numpy as np
from typing import List, Dict, Any


def load_audio(path: str, sr: int = 22050):
    try:
        y, _ = librosa.load(path, sr=sr)
        return y
    except Exception as e:
        print(f"Error loading audio {path}: {e}")
        return None


def extract_mfcc(y, sr=22050, n_mfcc=20):
    if y is None:
        return None
    try:
        return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    except Exception as e:
        print(f"Error extracting MFCC: {e}")
        return None


def match_jingle_dtw(chunk_path: str, jingle_mfcc: np.ndarray, sr: int = 22050) -> float:
    """
    Compute similarity between chunk and jingle using DTW on MFCCs.
    Returns a similarity score (the lower, the more similar).
    """
    y = load_audio(chunk_path, sr)
    chunk_mfcc = extract_mfcc(y, sr)
    if chunk_mfcc is None or jingle_mfcc is None:
        return float('inf')
    try:
        # Use librosa's DTW implementation
        D, wp = librosa.sequence.dtw(jingle_mfcc, chunk_mfcc, metric='cosine')
        # Normalize by path length
        score = D[-1, -1] / len(wp)
        return float(score)
    except Exception as e:
        print(f"DTW error for {chunk_path}: {e}")
        return float('inf')


def find_jingles_in_audio(audio_path: str, jingle_templates: Dict[str, np.ndarray], sr: int = 22050, threshold: float = 0.8, window_hop: float = 0.25) -> List[Dict[str, Any]]:
    """
    Scan audio for all occurrences of each jingle template using sliding window + DTW (MFCC).
    Returns list of detections: [{jingle, start_sec, end_sec, score}]
    """
    y = load_audio(audio_path, sr)
    if y is None:
        return []
    detections = []
    audio_len = librosa.get_duration(y=y, sr=sr)
    for jingle_name, jingle_mfcc in jingle_templates.items():
        jingle_len = jingle_mfcc.shape[1] * 512 / sr  # frame count * hop / sr
        win_size = int(jingle_len * sr)
        hop_size = int(window_hop * sr)
        for start in range(0, len(y) - win_size + 1, hop_size):
            y_win = y[start:start+win_size]
            mfcc_win = extract_mfcc(y_win, sr)
            if mfcc_win is None or mfcc_win.shape[1] < jingle_mfcc.shape[1] // 2:
                continue
            try:
                D, wp = librosa.sequence.dtw(
                    jingle_mfcc, mfcc_win, metric='cosine')
                score = D[-1, -1] / len(wp)
                if score < threshold:
                    detections.append({
                        'jingle': jingle_name,
                        'start_sec': start / sr,
                        'end_sec': (start + win_size) / sr,
                        'score': float(score)
                    })
            except Exception as e:
                print(f"DTW error in window: {e}")
                continue
    return detections


def detect_jingles(audio_dir: str, jingle_dir: str, sr: int = 22050, threshold: float = 0.8) -> List[Dict[str, Any]]:
    """
    Detect all occurrences of all jingles in each audio file in audio_dir.
    Returns a list of dicts: {chunk, jingle_occurrences: [ {jingle, start_sec, end_sec, score} ], detected_jingles: [jingle_filenames], jingle_scores: {jingle: best_score}}
    """
    # Load all reference jingles
    jingle_files = [f for f in os.listdir(
        jingle_dir) if f.endswith('.mp3') or f.endswith('.wav')]
    jingle_mfccs = {}
    jingle_duration_sec = 3  # Only use first 3 seconds for matching
    for jf in jingle_files:
        y = load_audio(os.path.join(jingle_dir, jf), sr)
        if y is not None:
            # Truncate to first N seconds
            y = y[:int(jingle_duration_sec * sr)]
        mfcc = extract_mfcc(y, sr)
        if mfcc is not None:
            jingle_mfccs[jf] = mfcc

    results = []
    for fname in sorted(os.listdir(audio_dir)):
        if not fname.endswith('.mp3') and not fname.endswith('.wav'):
            continue
        audio_path = os.path.join(audio_dir, fname)
        occurrences = find_jingles_in_audio(
            audio_path, jingle_mfccs, sr, threshold)
        detected = list(set([occ['jingle'] for occ in occurrences]))
        # For each jingle, best (lowest) score
        jingle_scores = {}
        for occ in occurrences:
            j = occ['jingle']
            if j not in jingle_scores or occ['score'] < jingle_scores[j]:
                jingle_scores[j] = occ['score']
        results.append({
            'chunk': fname,
            'jingle_occurrences': occurrences,
            'detected_jingles': detected,
            'jingle_scores': jingle_scores
        })
    return results
