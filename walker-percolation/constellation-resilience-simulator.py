import os
# --- Set BLAS / threading env vars early to avoid oversubscription ---
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
from scipy.sparse import csr_matrix, issparse, diags
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial import cKDTree
from scipy.sparse.linalg import eigsh, lobpcg
import warnings
from datetime import datetime

# ==========================================
# OUTPUT DIRECTORY CONFIGURATION
# ==========================================
OUTPUT_DIR = r"C:\Coding\aero\Results"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate timestamp for this run
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Setup log file for terminal output
LOG_FILE = os.path.join(OUTPUT_DIR, f"run_log_{RUN_TIMESTAMP}.txt")

# ==========================================
# LOGGING CLASS (Captures Terminal Output)
# ==========================================
class TeeLogger:
    """Captures stdout and writes to both console and file"""
    def isatty(self):
        return self.terminal.isatty() if hasattr(self.terminal, "isatty") else False
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

# ==========================================
# 0. MASTER CONFIGURATION (EDIT THIS)
# ==========================================
SIM_CONFIG = {
    # --- General Walker Parameters (Square Geometry) ---

    'N_SATS': 10000,          
    'PLANES': 100,           
    'PHASING': 1,          
    'INC_DEG': 53.0,         
    'ALT_KM': 550.0,        
    
    # --- Network Physics Constraints ---
    'D_MAX_KM': 1000.0,     
    'MAX_DEGREE': 4,         
    'ATMOSPHERE_KM': 80.0,   
    
    # --- Simulation Settings ---
    'PHASE_TRANSITION_MAX_N': 10000, 
    'PHASE_TRANSITION_STEP': 150,
    'SAMPLE_PATH_FRACTION': 0.7,   #Sample of satellites to calculate path for
    'MAX_WORKERS': max(1, min(10, mp.cpu_count()))
}

# ==========================================
# 1. PHYSICS & MATH HELPERS
# ==========================================
MU_EARTH = 398600.4418  # km^3 / s^2
R_EARTH_KM = 6371.0     # Mean Earth Radius
DEG2RAD = np.pi / 180.0

def rotation_matrix_x(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rotation_matrix_z(angle_rad):
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def walker_delta_dynamic(t_seconds, total_sats, planes, phasing, inc_deg,alt_km=SIM_CONFIG['ALT_KM']):
    if total_sats == 0:
        return np.empty((0, 3)), np.zeros((0,), dtype=int)

    if planes <= 0: planes = 1
    
    # Distribute satellites as evenly as possible among planes
    base = total_sats // planes
    rem = total_sats % planes
    sats_per_plane_list = [base + (1 if p < rem else 0) for p in range(planes)]
    total_sats_sym = sum(sats_per_plane_list)

    a = R_EARTH_KM + alt_km
    n = np.sqrt(MU_EARTH / (a**3)) # Mean motion
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
        
        # Walker Phasing Parameter (f)
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

    # Sphere Point Picking (Archimedes theorem)
    z = rng.uniform(-1.0, 1.0, N)
    theta = rng.uniform(0, 2 * np.pi, N)

    r_xy = np.sqrt(np.clip(1 - z**2, 0.0, 1.0))
    x = r_xy * np.cos(theta)
    y = r_xy * np.sin(theta)

    R = R_EARTH_KM + alt_km
    pos_0 = R * np.vstack((x, y, z)).T

    # Rotate the whole sphere to simulate time passing (optional for snapshot, but good for consistency)
    omega = np.sqrt(MU_EARTH / (R**3))
    angle = omega * t_seconds
    c, s = np.cos(angle), np.sin(angle)
    rot_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    return (rot_matrix @ pos_0.T).T

# ==========================================
# 2. GRAPH ENGINE 
# ==========================================
def check_los_occultation(r1, r2, R_body=R_EARTH_KM, atm_layer=80.0):
    """ Vectorized Line-of-Sight check. """
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
    """ Constructs the graph with physical constraints. """
    N = len(positions)
    if N < 2:
        return csr_matrix(([], ([], [])), shape=(N, N), dtype=int)

    tree = cKDTree(positions)
    current_degrees = np.zeros(N, dtype=int)
    row_ind = []
    col_ind = []

    k_search = min(N, max(50, max_degree * 6))
    dists, indices = tree.query(positions, k=k_search, distance_upper_bound=D_max_km)  
    
    rng = np.random.default_rng(seed)
    processing_order = rng.permutation(N)

    for i in processing_order:
        if current_degrees[i] >= max_degree: continue

        curr_dists = dists[i]
        curr_inds = indices[i]
        
        candidates = []
        candidate_indices = []
        sorted_order = np.argsort(curr_dists)

        for idx in sorted_order:
            j_idx = int(curr_inds[idx])
            dist = float(curr_dists[idx])

            if j_idx == N or j_idx == i: continue 
            if np.isinf(dist): continue
            if current_degrees[j_idx] >= max_degree: continue

            candidates.append(positions[j_idx])
            candidate_indices.append(j_idx)
            
            needed = max_degree - current_degrees[i]
            if len(candidates) >= needed + 2: break

        if not candidates: continue

        candidates = np.array(candidates)
        p1 = np.tile(positions[i], (len(candidates), 1))
        blocked = check_los_occultation(p1, candidates, atm_layer=atmosphere_km)

        for k_idx, is_blocked in enumerate(blocked):
            if current_degrees[i] >= max_degree: break
            target_idx = candidate_indices[k_idx]
            if (not is_blocked) and (current_degrees[target_idx] < max_degree):
                row_ind.extend([i, target_idx])
                col_ind.extend([target_idx, i])
                current_degrees[i] += 1
                current_degrees[target_idx] += 1

    data = np.ones(len(row_ind), dtype=int)
    A = csr_matrix((data, (row_ind, col_ind)), shape=(N, N))
    A.data[:] = 1 
    return A

# ==========================================
# 3. METRICS
# ==========================================
def calculate_fiedler_value(A):
    N = A.shape[0]
    if N < 2: return 0.0
    try:
        degrees = np.asarray(A.sum(axis=1)).ravel()
        L = diags(degrees) - A
        if N < 500:
            eigs = np.linalg.eigvalsh(L.toarray())
            eigs = np.sort(eigs)
            return float(eigs[1]) if len(eigs) > 1 else 0.0
        else:
            vals = eigsh(L, k=3, which='SM', return_eigenvectors=False)
            vals = np.sort(vals)
            vals = vals[vals > 1e-9]
            if len(vals) == 0: return 0.0
            return float(vals[0])
    except:
        return 0.0

def calculate_clustering_coefficient(A):
    if A.shape[0] < 3:
        return 0.0
    # Work on a copy to avoid side-effects
    A_copy = A.copy().astype(int)
    A_copy.data[:] = 1
    degrees = np.array(A_copy.sum(axis=1)).flatten()
    triplets = np.sum(degrees * (degrees - 1)) / 2.0
    if triplets == 0:
        return 0.0
    A2 = A_copy.dot(A_copy)
    tri_sum = A2.multiply(A_copy).sum()  # trace(A^3)
    # Global clustering coefficient: 3 * (#triangles) / (#triplets)
    return (3.0 * (tri_sum / 6.0)) / triplets


def calculate_average_path_length(A):
    N = A.shape[0]
    if N < 2: return 0.0
    k = max(1, int(N * SIM_CONFIG['SAMPLE_PATH_FRACTION']))
    rng = np.random.default_rng()
    sources = rng.choice(N, size=k, replace=False)
    try:
        dist_matrix = shortest_path(A, directed=False, unweighted=True, indices=sources)
        finite_dists = dist_matrix[np.isfinite(dist_matrix) & (dist_matrix > 0)]
        if finite_dists.size == 0: return 0.0
        return float(np.mean(finite_dists))
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

# ==========================================
# 4. WORKERS
# ==========================================
def _phase_worker(task):
    N_req, seed_main, seed_adj = task
    

    planes = max(1, int(np.sqrt(N_req)))
    pos_w, _ = walker_delta_dynamic(0, N_req, planes=planes, phasing=1, inc_deg=SIM_CONFIG['INC_DEG'])
    actual_N = len(pos_w)
    
    A_w = build_strict_adjacency(
        pos_w, 
        D_max_km=SIM_CONFIG['D_MAX_KM'], 
        max_degree=SIM_CONFIG['MAX_DEGREE'], 
        atmosphere_km=SIM_CONFIG['ATMOSPHERE_KM'],
        seed=seed_adj
    )
    w_metrics = calculate_metrics(A_w, actual_N)

    pos_s = stochastic_constellation_dynamic(0, actual_N, alt_km=SIM_CONFIG['ALT_KM'], seed=seed_adj)
    A_s = build_strict_adjacency(
        pos_s, 
        D_max_km=SIM_CONFIG['D_MAX_KM'], 
        max_degree=SIM_CONFIG['MAX_DEGREE'], 
        atmosphere_km=SIM_CONFIG['ATMOSPHERE_KM'],
        seed=seed_adj+1
    )
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

# ==========================================
# 5. EXPERIMENT CONTROLLERS
# ==========================================
def run_phase_transition():
    max_n = SIM_CONFIG['PHASE_TRANSITION_MAX_N']
    step = SIM_CONFIG['PHASE_TRANSITION_STEP']
    print(f"Running Phase Transition (N=50 to {max_n})...")
    
    tasks = []
    rng = np.random.default_rng(42)
    for n in range(50, max_n + 1, step):
        tasks.append((n, 42, rng.integers(0, 1e9)))
        
    results = {'N':[], 'w':{}, 's':{}}
    keys = ['gcc', 'chi', 'fiedler', 'clustering', 'path_length']
    for k in keys: results['w'][k] = []; results['s'][k] = []

    with mp.Pool(processes=SIM_CONFIG['MAX_WORKERS']) as pool:
        for res in pool.imap_unordered(_phase_worker, tasks):
            act_N, w, s = res
            print(f"  Processed N={act_N} | Walker GCC={w['gcc']:.2f}")
            results['N'].append(act_N)
            for k in keys:
                results['w'][k].append(w[k])
                results['s'][k].append(s[k])

    sorted_indices = np.argsort(results['N'])
    results['N'] = np.array(results['N'])[sorted_indices]
    for k in keys:
        results['w'][k] = np.array(results['w'][k])[sorted_indices]
        results['s'][k] = np.array(results['s'][k])[sorted_indices]
    return results

def run_kessler():
    print(f"Running Kessler Syndrome...")
    print(f"  Config: {SIM_CONFIG['N_SATS']} Sats, {SIM_CONFIG['PLANES']} Planes, D_max={SIM_CONFIG['D_MAX_KM']}km")
    
    # 1. Build Exact Starlink Graph
    pos_w, p_ids_w = walker_delta_dynamic(
        t_seconds=0, 
        total_sats=SIM_CONFIG['N_SATS'], 
        planes=SIM_CONFIG['PLANES'], 
        phasing=SIM_CONFIG['PHASING'], 
        inc_deg=SIM_CONFIG['INC_DEG'],
        alt_km=SIM_CONFIG['ALT_KM']
    )
    actual_N = len(pos_w)
    
    A_w = build_strict_adjacency(
        pos_w, 
        D_max_km=SIM_CONFIG['D_MAX_KM'], 
        max_degree=SIM_CONFIG['MAX_DEGREE'], 
        atmosphere_km=SIM_CONFIG['ATMOSPHERE_KM'],
        seed=999
    )
    
    # 2. Build Control Stochastic
    pos_s = stochastic_constellation_dynamic(0, actual_N, alt_km=SIM_CONFIG['ALT_KM'], seed=999) 
    A_s = build_strict_adjacency(
        pos_s, 
        D_max_km=SIM_CONFIG['D_MAX_KM'], 
        max_degree=SIM_CONFIG['MAX_DEGREE'], 
        atmosphere_km=SIM_CONFIG['ATMOSPHERE_KM'],
        seed=1000
    )
    
    fractions = np.linspace(0, 0.95, 30) # Increased resolution
    tasks = [(f, int(f*1000)) for f in fractions]
    
    data = {'frac': [], 'w_plane': [], 'w_rand': [], 's_rand': []}
    
    with mp.Pool(processes=SIM_CONFIG['MAX_WORKERS'], initializer=_kessler_init, initargs=(A_w, pos_w, p_ids_w, A_s, pos_s)) as pool:
        for res in pool.imap_unordered(_kessler_worker, tasks):
            f, wp, wr, sr = res
            data['frac'].append(f)
            data['w_plane'].append(wp)
            data['w_rand'].append(wr)
            data['s_rand'].append(sr)
            print(f"  Attack f={f:.2f} | Walker(Plane)={wp:.2f} | Walker(Rand)={wr:.2f}")

    idx = np.argsort(data['frac'])
    for k in data: data[k] = np.array(data[k])[idx]
    return data



def plot_results(pt, ks):
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1.2], hspace=0.4, wspace=0.3)

    # TOP ROW
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(pt['N'], pt['w']['gcc'], 'b-o', markersize=4, label='Walker')
    ax1.plot(pt['N'], pt['s']['gcc'], 'g--^', markersize=4, label='Stochastic')
    ax1.set_ylabel("Giant Component Fraction")
    ax1.set_title("Percolation Phase Transition")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax1b = ax1.twinx()
    ax1b.fill_between(pt['N'], pt['w']['chi'], color='blue', alpha=0.1)
    ax1b.fill_between(pt['N'], pt['s']['chi'], color='green', alpha=0.1)
    ax1b.set_ylabel("Susceptibility $\chi$", color='gray')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(pt['N'], pt['w']['fiedler'], 'b-o', markersize=4, label='Walker')
    ax2.plot(pt['N'], pt['s']['fiedler'], 'g--^', markersize=4, label='Stochastic')
    ax2.set_yscale('log')
    ax2.set_title("Algebraic Connectivity ($\lambda_2$)")
    ax2.grid(True, alpha=0.3)

    # MIDDLE ROW
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(pt['N'], pt['w']['clustering'], 'b-o', markersize=4, label='Walker')
    ax3.plot(pt['N'], pt['s']['clustering'], 'g--^', markersize=4, label='Stochastic')
    ax3.set_title("Global Clustering Coefficient")
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(pt['N'], pt['w']['path_length'], 'b-o', markersize=4, label='Walker')
    ax4.plot(pt['N'], pt['s']['path_length'], 'g--^', markersize=4, label='Stochastic')
    ax4.set_title("Average Path Length")
    ax4.grid(True, alpha=0.3)

    # BOTTOM ROW
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(ks['frac'], ks['w_plane'], 'r-X', linewidth=2.5, markersize=8, label='Walker: Plane Attack')
    ax5.plot(ks['frac'], ks['w_rand'], 'b--', alpha=0.6, label='Walker: Random Failure')
    ax5.plot(ks['frac'], ks['s_rand'], 'g-o', linewidth=1.5, alpha=0.4, label='Stochastic: Random (Ref)')
    
    if len(ks['frac']) > 0:
        diffs = ks['w_rand'] - ks['w_plane']
        div_indices = np.where(diffs > 0.1)[0]
        if len(div_indices) > 0:
            idx = div_indices[0]
            ax5.axvline(x=ks['frac'][idx], color='k', linestyle=':', alpha=0.5)
            ax5.text(ks['frac'][idx] + 0.02, 0.5, "Structural Failure\nRevealed", fontsize=9)

    ax5.set_title(f"Robustness: (D={int(SIM_CONFIG['D_MAX_KM'])})")
    ax5.set_xlabel("Fraction Removed")
    ax5.set_ylabel("GCC Size")
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(gs[2, 1])
    masking_delta = ks['w_rand'] - ks['w_plane']
    
    ax6.plot(ks['frac'], masking_delta, 'k-', linewidth=2)
    ax6.fill_between(ks['frac'], masking_delta, 0, where=(masking_delta > 0), color='red', alpha=0.3)
    ax6.fill_between(ks['frac'], masking_delta, 0, where=(masking_delta <= 0), color='green', alpha=0.1)
    
    ax6.set_title("The Masking Effect ($\Delta_{GCC}$)")
    ax6.set_xlabel("Fraction Removed")
    ax6.set_ylabel("Difference (Random - Plane)")
    ax6.set_ylim(-0.1, 0.6)
    
    ax6.text(0.1, 0.05, "Masking Zone\n", fontsize=10, color='green', fontweight='bold', ha='center')
    ax6.text(0.6, 0.3, "Structural Cost\n(Collapse)", fontsize=10, color='darkred', fontweight='bold', ha='center')
    ax6.grid(True, alpha=0.3)

    # Save the figure
    plot_file = os.path.join(OUTPUT_DIR, f"results_plot_{RUN_TIMESTAMP}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {plot_file}")

if __name__ == "__main__":
    # Setup logging to capture terminal output
    logger = TeeLogger(LOG_FILE)
    sys.stdout = logger
    sys.stderr = logger 
    try:
        print("=" * 80)
        print(f"Satellite Network Simulation - Run Started at {RUN_TIMESTAMP}")
        print("=" * 80)
        print(f"Output Directory: {OUTPUT_DIR}")
        print(f"Log File: {LOG_FILE}")
        print("\nSimulation Configuration:")
        for key, value in SIM_CONFIG.items():
            print(f"  {key}: {value}")
        print("=" * 80)
        print()
        
        t0 = time_module.time()
        
        # Run Experiments with Global Config
        pt_data = run_phase_transition()
        ks_data = run_kessler()
        
        runtime = time_module.time() - t0
        print(f"\nTotal Runtime: {runtime:.1f}s")
        print("=" * 80)
        
        # Plot and save results
        plot_results(pt_data, ks_data)
        
        print("\n" + "=" * 80)
        print("Simulation Complete!")
        print(f"All results saved to: {OUTPUT_DIR}")
        print("=" * 80)
        
    finally:
        # Close the logger
        sys.stdout = logger.terminal
        logger.close()
        print(f"\nLog file saved to: {LOG_FILE}")
