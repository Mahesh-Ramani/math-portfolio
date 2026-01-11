#!/usr/bin/env python3

#This code generates the dataset which I used for writing Ramani_Chord_Data.pdf

import numpy as np
import math
import csv
from itertools import combinations
from tqdm import tqdm
from collections import defaultdict

#these are the standard coefficients established by Plomp-Levelt
A = 3.5
B = 5.75
d_star = 0.24
s1 = 0.0207
s2 = 18.96

MIDI_MIN = 21
MIDI_MAX = 108
N_KEYS = MIDI_MAX - MIDI_MIN + 1

HAND_SPAN_SEMITONES = 18
HAND_WINDOW_KEYS = HAND_SPAN_SEMITONES + 1
HAND_STARTS = range(MIDI_MIN, MIDI_MAX - HAND_SPAN_SEMITONES + 1)


def midi_to_freq(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def plomp_levelt_pair(f1, f2):
    if f1 > f2:
        f1, f2 = f2, f1
    s = d_star / (s1 * f1 + s2)
    x = s * (f2 - f1)
    d = math.exp(-A * x) - math.exp(-B * x)
    return max(0.0, d)


def build_pairwise_matrix():
    midis = np.arange(MIDI_MIN, MIDI_MAX + 1)
    freqs = np.array([midi_to_freq(m) for m in midis])
    N = len(freqs)
    M = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        fi = freqs[i]
        for j in range(i + 1, N):
            fj = freqs[j]
            M[i, j] = plomp_levelt_pair(fi, fj)
            M[j, i] = M[i, j]
    return midis, freqs, M


def chord_harmonicity(freqs, max_div=12):
    if len(freqs) == 0:
        return 0.0
    best_err = 1.0
    for fi in freqs:
        for n in range(1, max_div + 1):
            f0 = fi / n
            if f0 <= 0.0:
                continue
            qs = freqs / f0
            errs = np.abs(qs - np.round(qs))
            mean_err = float(np.mean(errs))
            if mean_err < best_err:
                best_err = mean_err
                if best_err < 1e-6:
                    return 1.0
    best_err = min(max(best_err, 0.0), 0.5)
    harmonicity = 1.0 - 2.0 * best_err
    return float(max(0.0, min(1.0, harmonicity)))


def interval_class_vector(midi_list):
    ic_vec = np.zeros(12, dtype=int)
    sorted_midi = sorted(midi_list)
    for i in range(len(sorted_midi)):
        for j in range(i + 1, len(sorted_midi)):
            interval = (sorted_midi[j] - sorted_midi[i]) % 12
            ic_vec[interval] += 1
    return ic_vec


def chord_metrics(midi_list, pairwise_matrix, midi_to_idx_offset):
    mids = np.array(midi_list, dtype=float)
    n = len(mids)
    centroid = float(np.mean(mids)) if n > 0 else 0.0
    spread = float(np.ptp(mids)) if n > 0 else 0.0
    std = float(np.std(mids, ddof=0)) if n > 1 else 0.0
    if n > 1 and std > 1e-12:
        skew = float(np.mean(((mids - centroid) / std) ** 3))
        kurt = float(np.mean(((mids - centroid) / std) ** 4) - 3.0)
    else:
        skew = 0.0
        kurt = -3.0

    idx = [int(m - midi_to_idx_offset) for m in midi_list]
    s = 0.0
    pair_vals = []
    for a in range(n):
        for b in range(a + 1, n):
            v = pairwise_matrix[idx[a], idx[b]]
            pair_vals.append(v)
            s += v
    mean_pair = float(np.mean(pair_vals)) if pair_vals else 0.0

    freqs = np.array([midi_to_freq(int(m)) for m in midi_list], dtype=float)
    harm = chord_harmonicity(freqs, max_div=12)

    sorted_midi = sorted(midi_list)
    bass_note = int(sorted_midi[0]) if n > 0 else 0
    treble_note = int(sorted_midi[-1]) if n > 0 else 0
    
    ic_vec = interval_class_vector(midi_list)
    ic_vector_str = " ".join(str(x) for x in ic_vec)
    
    if n > 1:
        intervals = [sorted_midi[i+1] - sorted_midi[i] for i in range(n-1)]
        avg_spacing = float(np.mean(intervals))
        spacing_std = float(np.std(intervals))
    else:
        avg_spacing = 0.0
        spacing_std = 0.0

    return {
        "n_notes": n,
        "centroid_midi": centroid,
        "spread_semitones": spread,
        "skew": skew,
        "kurtosis": kurt,
        "dissonance_sum": s,
        "dissonance_mean": mean_pair,
        "harmonicity": harm,
        "bass_note": bass_note,
        "treble_note": treble_note,
        "ic_vector": ic_vector_str,
        "avg_spacing": avg_spacing,
        "spacing_std": spacing_std
    }


def generate_twohand_playable_chords_ordered(min_notes=2, max_notes=6, max_per_hand=None):
    if max_per_hand is None:
        max_per_hand = max_notes - 1

    window_cache = {}

    valid_starts = list(HAND_STARTS)
    for start in valid_starts:
        keys = list(range(start, start + HAND_WINDOW_KEYS))
        window_cache[start] = defaultdict(list)
        for k in range(1, min(max_per_hand, len(keys)) + 1):
            for comb in combinations(keys, k):
                window_cache[start][k].append(tuple(comb))

    for target_notes in range(min_notes, max_notes + 1):
        seen = set()
        print(f"\nGenerating chords with {target_notes} notes...")

        for i_ls, ls in enumerate(valid_starts):
            for rs in valid_starts[i_ls:]:
                for k_l in range(1, min(max_per_hand, HAND_WINDOW_KEYS) + 1):
                    k_r = target_notes - k_l
                    if not (1 <= k_r <= min(max_per_hand, HAND_WINDOW_KEYS)):
                        continue
                    list_l = window_cache[ls].get(k_l, ())
                    if not list_l:
                        continue
                    list_r = window_cache[rs].get(k_r, ())
                    if not list_r:
                        continue
                    for left_subset in list_l:
                        for right_subset in list_r:
                            chord_set = set(left_subset) | set(right_subset)
                            final_n = len(chord_set)
                            if final_n != target_notes:
                                continue
                            chord_tuple = tuple(sorted(chord_set))
                            if chord_tuple in seen:
                                continue
                            seen.add(chord_tuple)
                            yield chord_tuple


def main(output_csv="twohand_chords.csv", min_notes=2, max_notes=6, batch_size=100000):
    print("Building Plomp-Levelt pairwise matrix for 88 piano keys...")
    midis, freqs_array, pairwise_matrix = build_pairwise_matrix()
    print("Done matrix build.")

    header = [
        "midi_list", "n_notes",
        "centroid_midi", "spread_semitones", "skew", "kurtosis",
        "dissonance_sum", "dissonance_mean", "harmonicity",
        "bass_note", "treble_note", "ic_vector", 
        "avg_spacing", "spacing_std"
    ]

    writer = None
    buf = []
    total_written = 0

    gen = generate_twohand_playable_chords_ordered(min_notes=min_notes, max_notes=max_notes)

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for chord in tqdm(gen, desc="Chords generated"):
            metrics = chord_metrics(chord, pairwise_matrix, MIDI_MIN)
            row = [
                " ".join(str(int(m)) for m in chord),
                metrics["n_notes"],
                metrics["centroid_midi"],
                metrics["spread_semitones"],
                metrics["skew"],
                metrics["kurtosis"],
                metrics["dissonance_sum"],
                metrics["dissonance_mean"],
                metrics["harmonicity"],
                metrics["bass_note"],
                metrics["treble_note"],
                metrics["ic_vector"],
                metrics["avg_spacing"],
                metrics["spacing_std"]
            ]
            buf.append(row)
            if len(buf) >= batch_size:
                writer.writerows(buf)
                total_written += len(buf)
                buf = []
        if buf:
            writer.writerows(buf)
            total_written += len(buf)

    print(f"Finished. Rows written: {total_written}. CSV saved to: {output_csv}")


if __name__ == "__main__":
    OUTPUT_FILE = "twohand_chords.csv"
    MIN_NOTES = 2
    MAX_NOTES = 6
    BATCH_SIZE = 100000
    
    main(output_csv=OUTPUT_FILE, min_notes=MIN_NOTES, max_notes=MAX_NOTES, batch_size=BATCH_SIZE)
