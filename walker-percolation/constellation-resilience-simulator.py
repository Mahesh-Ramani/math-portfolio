import os
# env
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time as time_module
import math
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigsh
import warnings
from datetime import datetime
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.stats import t

plt.style.use('seaborn-v0_8-whitegrid')

# output
OUTPUT_DIR = r"C:\Coding\aero\results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = os.path.join(OUTPUT_DIR, f"run_log_{RUN_TIMESTAMP}.txt")

# logger
class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()

# config
SIM_CONFIG = {
    'N_SATS': 5000,
    'PLANES': 71,
    'PHASING': 1,
    'INC_DEG': 53.0,
    'ALT_KM': 550.0,
    'D_MAX_KM': 1000.0,
    'MAX_DEGREE': 4,
    'ATMOSPHERE_KM': 80.0,
    'PHASE_TRANSITION_MAX_N': 5000,
    'PHASE_TRANSITION_STEP': 150,
    'SAMPLE_PATH_FRACTION': 0.7,
    'MAX_WORKERS': max(1, min(10, mp.cpu_count())),
    'REPLICATES': 12,
    'KESSLER_FRACTIONS_NUM': 70,
    'CONFIDENCE_LEVEL': 0.95
}

# physics
MU_EARTH = 398600.4418
R_EARTH_KM = 6371.0
DEG2RAD = np.pi / 180.0

def rotation_matrix_x(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rotation_matrix_z(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def walker_delta_dynamic(t_seconds, total_sats, planes, phasing, inc_deg, alt_km=SIM_CONFIG['ALT_KM']):
    if total_sats == 0:
        return np.empty((0, 3)), np.zeros((0,), dtype=int)
    planes = max(1, planes)
    base = total_sats // planes
    rem = total_sats % planes
    sats_per_plane_list = [base + (1 if p < rem else 0) for p in range(planes)]
    total_sats_sym = sum(sats_per_plane_list)

    a = R_EARTH_KM + alt_km
    n = np.sqrt(MU_EARTH / (a**3))
    i_rad = inc_deg * DEG2RAD

    positions = np.zeros((total_sats_sym, 3))
    plane_ids = np.zeros(total_sats_sym, dtype=int)

    rot_x_inc = rotation_matrix_x(i_rad)

    idx = 0
    for j in range(planes):
        sats_in_this_plane = sats_per_plane_list[j]
        RAAN_j = 2 * np.pi * j / planes
        rot_z_raan = rotation_matrix_z(RAAN_j)
        matrix_op = rot_z_raan @ rot_x_inc

        phase_offset = (2 * np.pi / sats_in_this_plane) * phasing * j if sats_in_this_plane > 0 else 0.0

        for k in range(sats_in_this_plane):
            M0 = 2 * np.pi * k / sats_in_this_plane + phase_offset
            M_t = M0 + n * t_seconds
            r_pf = np.array([a * np.cos(M_t), a * np.sin(M_t), 0.0])
            positions[idx] = matrix_op @ r_pf
            plane_ids[idx] = j
            idx += 1

    return positions, plane_ids

def stochastic_constellation_dynamic(t_seconds, N, alt_km=SIM_CONFIG['ALT_KM'], seed=42):
    if N == 0: return np.empty((0, 3))
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1.0, 1.0, N)
    theta = rng.uniform(0, 2 * np.pi, N)
    r_xy = np.sqrt(np.clip(1 - z**2, 0.0, 1.0))
    x = r_xy * np.cos(theta)
    y = r_xy * np.sin(theta)
    R = R_EARTH_KM + alt_km
    pos_0 = R * np.vstack((x, y, z)).T

    omega = np.sqrt(MU_EARTH / (R**3))
    angle = omega * t_seconds
    c, s = np.cos(angle), np.sin(angle)
    rot_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return (rot_matrix @ pos_0.T).T

# graph
def check_los_occultation(r1, r2, R_body=R_EARTH_KM, atm_layer=80.0):
    d = r2 - r1
    dist_sq = np.sum(d**2, axis=1)
    dist_sq[dist_sq == 0] = 1e-9
    t = -np.sum(r1 * d, axis=1) / dist_sq
    segment_mask = (t > 0) & (t < 1)
    if not np.any(segment_mask):
        return np.zeros(len(r1), dtype=bool)
    closest = r1 + d * t[:, np.newaxis]
    closest_dist = np.linalg.norm(closest, axis=1)
    occluded = segment_mask & (closest_dist < (R_body + atm_layer))
    return occluded

def build_strict_adjacency(positions, D_max_km=2000.0, max_degree=4, atmosphere_km=80.0, seed=None):
    N = len(positions)
    if N < 2:
        return csr_matrix(([], ([], [])), shape=(N, N), dtype=int)

    tree = cKDTree(positions)
    current_degrees = np.zeros(N, dtype=int)
    row_ind, col_ind = [], []

    k_search = min(N, max(50, max_degree * 8))
    dists, indices = tree.query(positions, k=k_search, distance_upper_bound=D_max_km)

    processing_order = np.arange(N)

    for i in processing_order:
        if current_degrees[i] >= max_degree: continue
        curr_dists = dists[i]
        curr_inds = indices[i]
        sorted_order = np.argsort(curr_dists)

        candidates = []
        candidate_indices = []
        for idx in sorted_order:
            j_idx = int(curr_inds[idx])
            if j_idx == N or j_idx == i or np.isinf(curr_dists[idx]): continue
            if current_degrees[j_idx] >= max_degree: continue
            candidates.append(positions[j_idx])
            candidate_indices.append(j_idx)
            if len(candidates) >= (max_degree - current_degrees[i]) + 2: break

        if not candidates: continue
        candidates = np.array(candidates)
        p1 = np.tile(positions[i], (len(candidates), 1))
        blocked = check_los_occultation(p1, candidates, atm_layer=atmosphere_km)

        for k_idx, is_blocked in enumerate(blocked):
            if current_degrees[i] >= max_degree: break
            target_idx = candidate_indices[k_idx]
            if not is_blocked and current_degrees[target_idx] < max_degree:
                row_ind.extend([i, target_idx])
                col_ind.extend([target_idx, i])
                current_degrees[i] += 1
                current_degrees[target_idx] += 1

    data = np.ones(len(row_ind), dtype=int)
    A = csr_matrix((data, (row_ind, col_ind)), shape=(N, N))
    A.data[:] = 1
    return A

# metrics
def calculate_fiedler_value(A):
    N = A.shape[0]
    if N < 2: return 0.0
    try:
        degrees = np.asarray(A.sum(axis=1)).ravel()
        L = diags(degrees) - A
        if N < 500:
            eigs = np.linalg.eigvalsh(L.toarray())
            eigs = np.sort(eigs)
            if len(eigs) < 2 or abs(eigs[0]) > 1e-8:
                return 0.0
            return float(eigs[1])
        else:
            vals = eigsh(L, k=3, which='SM', tol=1e-6, return_eigenvectors=False)
            vals = np.sort(vals)
            vals = vals[vals > 1e-9]
            if len(vals) == 0:
                return 0.0
            return float(vals[0])
    except Exception as e:
        warnings.warn(f"Fiedler computation failed: {e}")
        return 0.0

def calculate_clustering_coefficient(A):
    if A.shape[0] < 3: return 0.0
    A_copy = A.copy().astype(int)
    A_copy.data[:] = 1
    degrees = np.array(A_copy.sum(axis=1)).flatten()
    triplets = np.sum(degrees * (degrees - 1)) / 2.0
    if triplets == 0: return 0.0
    A2 = A_copy.dot(A_copy)
    tri_sum = A2.multiply(A_copy).sum()
    return (3.0 * (tri_sum / 6.0)) / triplets

def calculate_average_path_length(A):
    N = A.shape[0]
    if N < 2: return 0.0
    k = max(1, int(N * SIM_CONFIG['SAMPLE_PATH_FRACTION']))
    rng = np.random.default_rng(42)
    sources = rng.choice(N, size=k, replace=False)
    try:
        dist_matrix = shortest_path(A, directed=False, unweighted=True, indices=sources)
        finite_dists = dist_matrix[np.isfinite(dist_matrix) & (dist_matrix > 0)]
        return float(np.mean(finite_dists)) if finite_dists.size > 0 else 0.0
    except:
        return 0.0

def calculate_metrics(A, N_total_original):
    metrics = {'gcc': 0.0, 'chi': 0.0, 'fiedler': 0.0, 'clustering': 0.0, 'path_length': 0.0}
    n_comp, labels = connected_components(A, directed=False)
    unique, counts = np.unique(labels, return_counts=True)
    if len(counts) == 0: return metrics

    gcc_size = np.max(counts)
    label_of_gcc = unique[np.argmax(counts)]
    metrics['gcc'] = gcc_size / N_total_original

    if len(counts) > 1:
        finite_clusters = counts[counts != gcc_size]
        metrics['chi'] = np.sum(finite_clusters.astype(float) ** 2) / N_total_original

    if gcc_size > 2:
        gcc_mask = (labels == label_of_gcc)
        A_gcc = A[gcc_mask][:, gcc_mask]
        metrics['fiedler'] = calculate_fiedler_value(A_gcc)
        metrics['clustering'] = calculate_clustering_coefficient(A_gcc)
        metrics['path_length'] = calculate_average_path_length(A_gcc)
    return metrics

def perform_attack(A_base, positions, plane_ids, mode='random', fraction=0.0, seed=None):
    N_total = A_base.shape[0]
    rng = np.random.default_rng(seed)
    keep_mask = np.ones(N_total, dtype=bool)

    if mode == 'random':
        n_remove = int(N_total * fraction)
        if n_remove > 0:
            dead = rng.choice(N_total, n_remove, replace=False)
            keep_mask[dead] = False
    elif mode == 'plane':
        unique_planes = np.unique(plane_ids)
        n_kill = int(len(unique_planes) * fraction)
        if n_kill > 0:
            dead_planes = rng.choice(unique_planes, n_kill, replace=False)
            keep_mask[np.isin(plane_ids, dead_planes)] = False

    A_surviving = A_base[keep_mask][:, keep_mask]
    n_comp, labels = connected_components(A_surviving, directed=False)
    gcc_size = np.max(np.bincount(labels)) if n_comp > 0 else 0
    return gcc_size / N_total

# workers
def _phase_worker(task):
    N_req, seed_main, seed_adj = task
    planes = max(1, int(np.sqrt(N_req)))
    pos_w, _ = walker_delta_dynamic(0, N_req, planes=planes, phasing=1, inc_deg=SIM_CONFIG['INC_DEG'])
    actual_N = len(pos_w)
    A_w = build_strict_adjacency(pos_w, D_max_km=SIM_CONFIG['D_MAX_KM'], max_degree=SIM_CONFIG['MAX_DEGREE'],
                                 atmosphere_km=SIM_CONFIG['ATMOSPHERE_KM'])
    w_metrics = calculate_metrics(A_w, actual_N)

    pos_s = stochastic_constellation_dynamic(0, actual_N, alt_km=SIM_CONFIG['ALT_KM'], seed=seed_adj)
    A_s = build_strict_adjacency(pos_s, D_max_km=SIM_CONFIG['D_MAX_KM'], max_degree=SIM_CONFIG['MAX_DEGREE'],
                                 atmosphere_km=SIM_CONFIG['ATMOSPHERE_KM'])
    s_metrics = calculate_metrics(A_s, actual_N)
    return (actual_N, w_metrics, s_metrics)

_shared_kessler = {}
def _kessler_init(A_w, pos_w, p_ids_w, A_s, pos_s):
    _shared_kessler['A_w'] = A_w
    _shared_kessler['pos_w'] = pos_w
    _shared_kessler['p_ids_w'] = p_ids_w
    _shared_kessler['A_s'] = A_s
    _shared_kessler['pos_s'] = pos_s

def _kessler_worker(args):
    frac, seed = args
    A_w = _shared_kessler['A_w']
    pos_w = _shared_kessler['pos_w']
    p_ids_w = _shared_kessler['p_ids_w']
    A_s = _shared_kessler['A_s']
    pos_s = _shared_kessler['pos_s']
    w_plane = perform_attack(A_w, pos_w, p_ids_w, 'plane', frac, seed)
    w_rand = perform_attack(A_w, pos_w, p_ids_w, 'random', frac, seed)
    s_rand = perform_attack(A_s, pos_s, None, 'random', frac, seed)
    return (frac, w_plane, w_rand, s_rand)

# experiments
def run_phase_transition():
    max_n = SIM_CONFIG['PHASE_TRANSITION_MAX_N']
    step = SIM_CONFIG['PHASE_TRANSITION_STEP']
    print(f"Running Phase Transition (N=50 to {max_n}, step={step})...")
    tasks = []
    rng = np.random.default_rng(42)
    for n in range(50, max_n + 1, step):
        tasks.append((n, 42, rng.integers(0, 1e9)))

    results = {'N': [], 'w': {k: [] for k in ['gcc', 'chi', 'fiedler', 'clustering', 'path_length']},
               's': {k: [] for k in ['gcc', 'chi', 'fiedler', 'clustering', 'path_length']}}

    with mp.Pool(processes=SIM_CONFIG['MAX_WORKERS']) as pool:
        for res in pool.imap_unordered(_phase_worker, tasks):
            act_N, w, s = res
            print(f"  Processed N={act_N} | Walker GCC={w['gcc']:.2f}")
            results['N'].append(act_N)
            for k in results['w']:
                results['w'][k].append(w[k])
                results['s'][k].append(s[k])

    sorted_idx = np.argsort(results['N'])
    results['N'] = np.array(results['N'])[sorted_idx]
    for k in results['w']:
        results['w'][k] = np.array(results['w'][k])[sorted_idx]
        results['s'][k] = np.array(results['s'][k])[sorted_idx]
    return results

def run_kessler():
    print(f"Running Kessler Syndrome (replicates={SIM_CONFIG['REPLICATES']}, fractions={SIM_CONFIG['KESSLER_FRACTIONS_NUM']})...")
    print(f"  Config: N={SIM_CONFIG['N_SATS']}, Planes={SIM_CONFIG['PLANES']}, D_max={SIM_CONFIG['D_MAX_KM']} km")

    pos_w, p_ids_w = walker_delta_dynamic(0, SIM_CONFIG['N_SATS'], SIM_CONFIG['PLANES'], SIM_CONFIG['PHASING'],
                                          SIM_CONFIG['INC_DEG'], SIM_CONFIG['ALT_KM'])
    actual_N = len(pos_w)
    A_w = build_strict_adjacency(pos_w, D_max_km=SIM_CONFIG['D_MAX_KM'], max_degree=SIM_CONFIG['MAX_DEGREE'],
                                 atmosphere_km=SIM_CONFIG['ATMOSPHERE_KM'])
    pos_s = stochastic_constellation_dynamic(0, actual_N, alt_km=SIM_CONFIG['ALT_KM'], seed=999)
    A_s = build_strict_adjacency(pos_s, D_max_km=SIM_CONFIG['D_MAX_KM'], max_degree=SIM_CONFIG['MAX_DEGREE'],
                                 atmosphere_km=SIM_CONFIG['ATMOSPHERE_KM'])

    fractions = np.linspace(0, 0.95, SIM_CONFIG['KESSLER_FRACTIONS_NUM'])
    replicates = SIM_CONFIG['REPLICATES']

    w_plane_data = np.zeros((replicates, len(fractions)))
    w_rand_data = np.zeros((replicates, len(fractions)))
    s_rand_data = np.zeros((replicates, len(fractions)))

    for rep in range(replicates):
        seed_base = 999 + rep * 123456
        tasks = [(f, seed_base + int(f * 100000)) for f in fractions]
        rep_results = []
        with mp.Pool(processes=SIM_CONFIG['MAX_WORKERS'], initializer=_kessler_init,
                     initargs=(A_w, pos_w, p_ids_w, A_s, pos_s)) as pool:
            for res in pool.imap_unordered(_kessler_worker, tasks):
                rep_results.append(res)
        rep_results.sort(key=lambda x: x[0])
        for i, (_, wp, wr, sr) in enumerate(rep_results):
            w_plane_data[rep, i] = wp
            w_rand_data[rep, i] = wr
            s_rand_data[rep, i] = sr
        print(f"  Completed replicate {rep+1}/{replicates}")

    data = {
        'frac': fractions,
        'w_plane_mean': np.mean(w_plane_data, axis=0),
        'w_plane_std': np.std(w_plane_data, axis=0),
        'w_rand_mean': np.mean(w_rand_data, axis=0),
        'w_rand_std': np.std(w_rand_data, axis=0),
        's_rand_mean': np.mean(s_rand_data, axis=0),
        's_rand_std': np.std(s_rand_data, axis=0)
    }

    if replicates > 1:
        t_val = t.ppf(1 - (1 - SIM_CONFIG['CONFIDENCE_LEVEL']) / 2, df=replicates - 1)
        sem = data['w_plane_std'] / np.sqrt(replicates)
        data['w_plane_ci_low'] = data['w_plane_mean'] - t_val * sem
        data['w_plane_ci_high'] = data['w_plane_mean'] + t_val * sem

        sem = data['w_rand_std'] / np.sqrt(replicates)
        data['w_rand_ci_low'] = data['w_rand_mean'] - t_val * sem
        data['w_rand_ci_high'] = data['w_rand_mean'] + t_val * sem

        sem = data['s_rand_std'] / np.sqrt(replicates)
        data['s_rand_ci_low'] = data['s_rand_mean'] - t_val * sem
        data['s_rand_ci_high'] = data['s_rand_mean'] + t_val * sem
    else:
        data['w_plane_ci_low'] = data['w_plane_ci_high'] = data['w_plane_mean']
        data['w_rand_ci_low'] = data['w_rand_ci_high'] = data['w_rand_mean']
        data['s_rand_ci_low'] = data['s_rand_ci_high'] = data['s_rand_mean']

    return data

# plotting
def plot_results(pt, ks):
    fig = plt.figure(figsize=(18, 16))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.3], hspace=0.35, wspace=0.3)

    config_str = (f"Walker Delta: N={SIM_CONFIG['N_SATS']}, Planes={SIM_CONFIG['PLANES']}, "
                  f"Phasing={SIM_CONFIG['PHASING']}, Inc={SIM_CONFIG['INC_DEG']}°, "
                  f"Alt={SIM_CONFIG['ALT_KM']} km, D_max={SIM_CONFIG['D_MAX_KM']} km, "
                  f"Max Deg={SIM_CONFIG['MAX_DEGREE']}, Replicates={SIM_CONFIG['REPLICATES']}, "
                  f"CI={int(SIM_CONFIG['CONFIDENCE_LEVEL']*100)}%")

    fig.suptitle(config_str, fontsize=14, y=0.98)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(pt['N'], pt['w']['gcc'], 'b-o', ms=5, label='Walker Delta')
    ax1.plot(pt['N'], pt['s']['gcc'], 'g--s', ms=5, label='Stochastic')
    ax1.set_ylabel('Giant Component Fraction (GCC)')
    ax1.set_title('Percolation Phase Transition')
    ax1.legend(frameon=True)
    ax1b = ax1.twinx()
    ax1b.fill_between(pt['N'], pt['w']['chi'], color='blue', alpha=0.12)
    ax1b.fill_between(pt['N'], pt['s']['chi'], color='green', alpha=0.12)
    ax1b.set_ylabel('Susceptibility χ', color='gray')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(pt['N'], pt['w']['fiedler'], 'b-o', ms=5, label='Walker λ₂')
    ax2.plot(pt['N'], pt['s']['fiedler'], 'g--s', ms=5, label='Stochastic')
    ax2.set_yscale('log')
    ax2.set_ylabel('Algebraic Connectivity λ₂')
    ax2.set_title('Algebraic Connectivity')
    ax2.legend(frameon=True)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(pt['N'], pt['w']['clustering'], 'b-o', ms=5, label='Walker')
    ax3.plot(pt['N'], pt['s']['clustering'], 'g--s', ms=5, label='Stochastic')
    ax3.set_ylabel('Global Clustering Coefficient')
    ax3.set_title('Global Clustering')
    ax3.legend(frameon=True)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(pt['N'], pt['w']['path_length'], 'b-o', ms=5, label='Walker')
    ax4.plot(pt['N'], pt['s']['path_length'], 'g--s', ms=5, label='Stochastic')
    ax4.set_ylabel('Average Path Length (sampled)')
    ax4.set_title('Routing Efficiency')
    ax4.legend(frameon=True)

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(ks['frac'], ks['w_plane_mean'], 'r-X', lw=2.5, ms=8, label='Plane Attack (mean)')
    ax5.fill_between(ks['frac'], ks['w_plane_ci_low'], ks['w_plane_ci_high'], color='red', alpha=0.18, label='95% CI')
    ax5.plot(ks['frac'], ks['w_rand_mean'], 'b--D', lw=2, ms=6, label='Random Failure (mean)')
    ax5.fill_between(ks['frac'], ks['w_rand_ci_low'], ks['w_rand_ci_high'], color='blue', alpha=0.12, label='95% CI')
    ax5.plot(ks['frac'], ks['s_rand_mean'], 'g-o', lw=1.8, ms=6, alpha=0.8, label='Stochastic Ref (mean)')
    ax5.fill_between(ks['frac'], ks['s_rand_ci_low'], ks['s_rand_ci_high'], color='green', alpha=0.1, label='95% CI')
    ax5.set_xlabel('Fraction Removed')
    ax5.set_ylabel('Normalized GCC')
    ax5.set_title('Network Robustness under Attacks')
    ax5.legend(loc='upper right', frameon=True, fontsize=10)

    ax6 = fig.add_subplot(gs[2, 1])
    masking_delta = ks['w_rand_mean'] - ks['w_plane_mean']
    ax6.plot(ks['frac'], masking_delta, 'k-', lw=2.5, label='ΔGCC (Random - Plane)')
    ax6.fill_between(ks['frac'], masking_delta, 0, where=(masking_delta > 0), color='red', alpha=0.25, label='Masking Zone')
    ax6.fill_between(ks['frac'], masking_delta, 0, where=(masking_delta <= 0), color='green', alpha=0.15, label='Collapse Region')
    max_idx = np.argmax(masking_delta)
    ax6.axvline(ks['frac'][max_idx], color='purple', linestyle='--', lw=1.8, label=f'Peak Masking (f={ks["frac"][max_idx]:.3f})')
    ax6.set_xlabel('Fraction Removed')
    ax6.set_ylabel('ΔGCC (Masking Effect)')
    ax6.set_title('Masking Effect Quantification')
    ax6.set_ylim(np.min(masking_delta)-0.05, np.max(masking_delta)+0.1)
    ax6.legend(loc='upper left', frameon=True, fontsize=10)

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.grid(True, alpha=0.4, linestyle='--')

    plot_file = os.path.join(OUTPUT_DIR, f"results_plot_{RUN_TIMESTAMP}.png")
    plt.savefig(plot_file, dpi=400, bbox_inches='tight')
    print(f"Main results plot saved to: {plot_file}")
    plt.close(fig)

# sweep & analysis
default_config = SIM_CONFIG.copy()

sweep_vars = {
    'ALT_KM': np.arange(300, 1300, 200),
    'D_MAX_KM': np.arange(500, 3250, 250),
    'MAX_DEGREE': np.arange(2, 6, 1),
    'N_SATS': np.arange(1000, 9000, 1000),
    'PHASING': np.arange(0, 101, 10),
    'PLANES': np.arange(10, 140, 20),
    'INC_DEG': np.arange(30, 100, 10),
}

def extract_metrics(pt, ks):
    gcc_final = pt['w']['gcc'][-1] if len(pt['w']['gcc']) > 0 else 0.0
    fiedler_final = pt['w']['fiedler'][-1] if len(pt['w']['fiedler']) > 0 else 0.0

    if len(ks['frac']) == 0:
        return {'gcc_final': 0.0, 'fiedler_final': 0.0, 'max_masking': 0.0, 'area_masking': 0.0,
                'resilience_plane': 0.0, 'resilience_rand': 0.0}

    masking_delta = ks['w_rand_mean'] - ks['w_plane_mean']
    max_masking = np.max(masking_delta)
    area_masking = np.trapezoid(np.maximum(masking_delta, 0), ks['frac'])

    def get_resilience(gcc_series):
        if np.min(gcc_series) >= 0.5: return 1.0
        if np.max(gcc_series) <= 0.5: return 0.0
        interp = interp1d(gcc_series[::-1], ks['frac'][::-1], fill_value="extrapolate", bounds_error=False)
        return interp(0.5)

    resilience_plane = get_resilience(ks['w_plane_mean'])
    resilience_rand = get_resilience(ks['w_rand_mean'])

    return {
        'gcc_final': gcc_final,
        'fiedler_final': fiedler_final,
        'max_masking': max_masking,
        'area_masking': area_masking,
        'resilience_plane': resilience_plane,
        'resilience_rand': resilience_rand
    }

def run_oat_sweep():
    results = []
    orig_config = SIM_CONFIG.copy()

    print("Running default configuration...")
    pt_data = run_phase_transition()
    ks_data = run_kessler()
    metrics = extract_metrics(pt_data, ks_data)
    results.append({'variable': 'default', 'value': np.nan, **metrics})
    plot_results(pt_data, ks_data)

    for var, values in sweep_vars.items():
        for val in values:
            config = default_config.copy()
            config[var] = val
            if var == 'N_SATS':
                config['PLANES'] = max(1, int(np.sqrt(val)))
                config['PHASE_TRANSITION_MAX_N'] = val
            print(f"Running sweep for {var} = {val}")
            SIM_CONFIG.update(config)
            pt_data = run_phase_transition()
            ks_data = run_kessler()
            metrics = extract_metrics(pt_data, ks_data)
            results.append({'variable': var, 'value': val, **metrics})

    SIM_CONFIG.update(orig_config)
    return pd.DataFrame(results)

def plot_sweep_results(df):
    unique_vars = [v for v in sweep_vars if v in df['variable'].unique()]
    metrics_to_plot = ['gcc_final', 'fiedler_final', 'max_masking', 'area_masking', 'resilience_plane', 'resilience_rand']

    if unique_vars:
        fig, axs = plt.subplots(len(unique_vars), 1, figsize=(12, 4 * len(unique_vars)), squeeze=False)
        for i, var in enumerate(unique_vars):
            subdf = df[df['variable'] == var].sort_values('value')
            ax = axs[i, 0]
            for met in metrics_to_plot:
                ax.plot(subdf['value'], subdf[met], marker='o', label=met.replace('_', ' ').title())
            ax.set_title(f'One-at-a-Time Sweep: {var}')
            ax.set_xlabel(var)
            ax.set_ylabel('Metric Value')
            ax.legend(frameon=True, fontsize=9)
            ax.grid(True, alpha=0.5)
        plt.tight_layout()
        sweep_file = os.path.join(OUTPUT_DIR, f"sweep_line_plots_{RUN_TIMESTAMP}.png")
        plt.savefig(sweep_file, dpi=300, bbox_inches='tight')
        print(f"Sweep line plots saved to: {sweep_file}")
        plt.close(fig)

    corr_data = {}
    for var in unique_vars:
        subdf = df[df['variable'] == var]
        corr_series = subdf.corr(numeric_only=True)['value'].drop('value', errors='ignore')
        corr_data[var] = corr_series.reindex(metrics_to_plot).fillna(0)

    corr_df = pd.DataFrame(corr_data).T
    if not corr_df.empty:
        fig = plt.figure(figsize=(14, 9))
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=0.5)
        plt.title('Correlation Heatmap: Sweep Parameters vs Key Metrics')
        heat_file = os.path.join(OUTPUT_DIR, f"corr_heatmap_{RUN_TIMESTAMP}.png")
        plt.savefig(heat_file, dpi=300, bbox_inches='tight')
        print(f"Correlation heatmap saved to: {heat_file}")
        plt.close(fig)

    return corr_df

def generate_recommendations(df, corr_df):
    print("\n=== Optimal Configurations & Recommendations ===")
    df['score'] = df['resilience_plane'] - 2 * df['area_masking']
    unique_vars = [v for v in sweep_vars if v in df['variable'].unique()]

    for var in unique_vars:
        sub = df[df['variable'] == var]
        if sub.empty: continue
        opt_row = sub.loc[sub['score'].idxmax()]
        print(f"Optimal {var}: {opt_row['value']} (score: {opt_row['score']:.3f})")
        print(f"  GCC Final: {opt_row['gcc_final']:.3f} | Fiedler: {opt_row['fiedler_final']:.4f}")
        print(f"  Max Masking: {opt_row['max_masking']:.3f} | Area Masking: {opt_row['area_masking']:.3f}")
        print(f"  Resilience Plane: {opt_row['resilience_plane']:.3f} | Random: {opt_row['resilience_rand']:.3f}\n")

    if 'area_masking' in corr_df.columns:
        corrs = corr_df['area_masking']
        best = corrs.idxmin()
        val = corrs[best]
        if val < -0.1:
            print(f"Strongest lever to reduce masking: {best} (corr = {val:.2f})")
            print("  Adjusting this parameter can minimize hidden structural vulnerabilities.")
        else:
            print("No strong negative correlations found for reducing masking area.")

    default_score = df[df['variable'] == 'default']['score'].values[0]
    print(f"Default configuration score: {default_score:.3f}")

if __name__ == "__main__":
    logger = TeeLogger(LOG_FILE)
    sys.stdout = logger
    sys.stderr = logger
    try:
        print("=" * 90)
        print(f"Satellite Constellation Robustness Simulation - Started {RUN_TIMESTAMP}")
        print(f"Output: {OUTPUT_DIR}")
        print(f"Log: {LOG_FILE}")
        print("\nConfiguration:")
        for k, v in SIM_CONFIG.items():
            print(f"  {k: <25}: {v}")
        print("=" * 90)

        t0 = time_module.time()
        df = run_oat_sweep()
        corr_df = plot_sweep_results(df)
        generate_recommendations(df, corr_df)

        runtime = time_module.time() - t0
        print(f"\nTotal Runtime: {runtime:.1f} s")
        print("=" * 90)
        print("Simulation Complete. Results saved.")
        print("=" * 90)
    finally:
        sys.stdout = logger.terminal
        logger.close()
        print(f"Log saved: {LOG_FILE}")
