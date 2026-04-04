
from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Known optima and best-known feasible solutions
# ---------------------------------------------------------------------------
KNOWN_OPTIMA: dict[int, dict] = {
    1: {"queens": 1,  "status": "optimal",  "placement":[(0, 0, 0)]},
    2: {"queens": 1,  "status": "optimal",  "placement": [(1, 0, 0)]},
    3: {"queens": 1,  "status": "optimal",  "placement":[(1, 1, 1)]},
    4: {"queens": 4,  "status": "optimal",
        "placement":[(1, 0, 3), (1, 1, 0), (1, 2, 0), (1, 3, 3)]},
    5: {"queens": 6,  "status": "optimal",
        "placement":[(1, 0, 3), (1, 1, 0), (1, 3, 4), (1, 4, 1),
                      (2, 2, 2), (3, 2, 2)]},
    6: {"queens": 8,  "status": "optimal",
        "placement":[(2, 2, 2), (2, 2, 3), (2, 3, 2), (2, 3, 3),
                      (3, 2, 2), (3, 2, 3), (3, 3, 2), (3, 3, 3)]},
    7: {"queens": 12, "status": "feasible",
        "placement":[(0, 4, 3), (0, 6, 6), (1, 1, 5), (2, 3, 0),
                      (2, 4, 0), (3, 0, 2), (3, 6, 4), (4, 3, 6),
                      (4, 6, 4), (5, 0, 1), (6, 2, 3), (6, 5, 1)]},
    8: {"queens": 16, "status": "feasible",
        "placement":[(0, 2, 3), (0, 3, 4), (1, 0, 6), (1, 4, 6),
                      (1, 7, 6), (2, 4, 0), (3, 4, 3), (3, 7, 2),
                      (4, 0, 4), (4, 1, 1), (4, 3, 5), (5, 6, 7),
                      (6, 2, 5), (6, 3, 0), (7, 5, 3), (7, 7, 2)]},
}


# ---------------------------------------------------------------------------
# Gurobi environment
# ---------------------------------------------------------------------------

def make_gurobi_env(verbose: bool = False) -> gp.Env:
    env = gp.Env(empty=True)
    # Keeping credentials hardcoded as requested
    env.setParam("WLSACCESSID", "x")
    env.setParam("WLSSECRET",   "y")
    env.setParam("LICENSEID",   123456)          
    env.setParam("OutputFlag",  1 if verbose else 0)
    env.start()
    return env


# ---------------------------------------------------------------------------
# Geometry — O(N^4) ray-cast
# ---------------------------------------------------------------------------

_QUEEN_DIRS: list[tuple[int, int, int]] =[
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) > (0, 0, 0)
]


def precompute_attackers(N: int) -> dict[tuple, list[tuple]]:
    cells = list(itertools.product(range(N), repeat=3))
    attackers: dict[tuple, set[tuple]] = {c: {c} for c in cells}

    for (x, y, z) in cells:
        for (dx, dy, dz) in _QUEEN_DIRS:
            nx, ny, nz = x + dx, y + dy, z + dz
            while 0 <= nx < N and 0 <= ny < N and 0 <= nz < N:
                attackers[(x, y, z)].add((nx, ny, nz))
                attackers[(nx, ny, nz)].add((x, y, z))
                nx += dx; ny += dy; nz += dz

    return {c: list(v) for c, v in attackers.items()}


# ---------------------------------------------------------------------------
# Octahedral symmetry
# ---------------------------------------------------------------------------

_IDENTITY_SYM: tuple = ((0, 1, 2), (1, 1, 1))


def apply_symmetry(perm, signs, c, N):
    return tuple(
        c[perm[j]] if signs[j] == 1 else N - 1 - c[perm[j]]
        for j in range(3)
    )


def octahedral_symmetries(N):
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            yield (perm, signs)


def fundamental_domain(N: int) -> list[tuple]:
    mid = (N - 1) // 2
    return[
        (x, y, z)
        for x in range(mid + 1)
        for y in range(x, mid + 1)
        for z in range(y, mid + 1)
    ]


def canonicalize_placement(queens: list[tuple], N: int) -> list[tuple]:
    best = tuple(sorted(queens))
    for perm, signs in octahedral_symmetries(N):
        if (perm, signs) == _IDENTITY_SYM:
            continue
        image = tuple(sorted(
            apply_symmetry(perm, signs, q, N) for q in queens
        ))
        if image < best:
            best = image
    return list(best)


def get_hint_for_n(N: int, canonicalize: bool = False) -> list[tuple] | None:
    entry = KNOWN_OPTIMA.get(N)
    if entry is None:
        return None
    placement = list(entry["placement"])
    if canonicalize:
        placement = canonicalize_placement(placement, N)
    return placement


def trim_hint_by_coverage(
    hint: list[tuple],
    attackers: dict[tuple, list[tuple]],
    k: int,
) -> list[tuple]:
    covered: set[tuple] = set()
    selected: list[tuple] =[]
    remaining = list(hint)
    for _ in range(k):
        if not remaining:
            break
        best = max(remaining, key=lambda q: len(set(attackers[q]) - covered))
        selected.append(best)
        covered.update(attackers[best])
        remaining.remove(best)
    return selected


def add_fundamental_domain_pin(
    model: gp.Model,
    board: dict,
    cells: list[tuple],
    N: int,
) -> None:
    fd = set(fundamental_domain(N))
    lex_order = sorted(cells)

    for i, c in enumerate(lex_order):
        if c in fd:
            continue
        earlier = lex_order[:i]
        if not earlier:
            continue
        model.addConstr(
            board[c] <= gp.quicksum(board[d] for d in earlier),
            name=f"fdpin_{c[0]}_{c[1]}_{c[2]}",
        )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_solution(queens: list[tuple], N: int) -> bool:
    for q in queens:
        if not all(0 <= c < N for c in q):
            print(f"[INVALID] Queen {q} out of bounds for N={N}.")
            return False
    queen_set = set(queens)
    uncovered =[
        cell for cell in itertools.product(range(N), repeat=3)
        if not any(
            cell == q or (
                len({abs(a-b) for a, b in zip(cell, q) if a != b}) == 1
            )
            for q in queen_set
        )
    ]
    if uncovered:
        print(f"[INVALID] {len(uncovered)}/{N**3} cells not dominated.")
        return False
    print(f"[VALID]   {len(queens)} queens dominate all {N**3} cells of {N}³.")
    return True


# ---------------------------------------------------------------------------
# Result type & I/O
# ---------------------------------------------------------------------------

@dataclass
class SolveResult:
    status: str
    placement: list[tuple] | None
    elapsed: float
    N: int

    def __repr__(self) -> str:
        if self.placement is not None:
            return (f"SolveResult(N={self.N}, {self.status}, "
                    f"{len(self.placement)} queens, {self.elapsed:.2f}s)")
        return f"SolveResult(N={self.N}, {self.status}, {self.elapsed:.2f}s)"


def save_result_to_txt(
    result: SolveResult,
    filepath: str = "3d_queen_domination_results.txt",
    append: bool = False,
) -> None:
    file_exists = os.path.exists(filepath)
    mode = "a" if (append and file_exists) else "w"
    with open(filepath, mode, encoding="utf-8") as f:
        if not file_exists or not append:
            f.write("3D Queen Domination Results\n===========================\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"N = {result.N}\n")
        f.write(f"Status          : {result.status.upper()}\n")
        f.write(f"Time            : {result.elapsed:.3f} seconds\n")
        if result.placement is not None:
            f.write(f"Queens used     : {len(result.placement)}\n")
            f.write("Placement (sorted lex):\n")
            for q in sorted(result.placement):
                f.write(f"    {q}\n")
        else:
            f.write("Placement       : None\n")
        f.write("\n" + "-" * 60 + "\n\n")
    print(f"[INFO] Results {'appended to' if append else 'saved to'} → {filepath}")


def _save_checkpoint(path: str, queens: list[tuple], N: int) -> None:
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(f"# N={N} checkpoint -- {len(queens)} queens\n")
        f.write(f"# Saved: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"N={N}\n")
        for q in sorted(queens):
            f.write(f"{q[0]},{q[1]},{q[2]}\n")
    os.replace(tmp, path)


def load_checkpoint(path: str) -> tuple[int, list[tuple]] | None:
    if not os.path.exists(path):
        return None
    queens, N =[], None
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            if line.startswith('N='):
                N = int(line.split('=')[1])
            else:
                parts = line.split(',')
                if len(parts) == 3:
                    queens.append(tuple(int(x) for x in parts))
    if N is None or not queens:
        return None
    return (N, queens)


class SolutionLogger:
    def __init__(self, board, cells, N, checkpoint_path=None):
        self._cells           = cells
        self._N               = N
        self._checkpoint_path = checkpoint_path
        self._count           = 0
        self._start           = time.time()
        self._vars            = [board[c] for c in cells]
        self._last_ckpt_time  = 0.0
        self._last_ckpt_obj   = float('inf')
        self.best_solution: list[tuple] | None = None

    def __call__(self, model: gp.Model, where: int) -> None:
        if where != GRB.Callback.MIPSOL:
            return
        self._count += 1
        vals    = model.cbGetSolution(self._vars)
        queens  =[c for c, v in zip(self._cells, vals) if v > 0.5]
        obj     = int(round(model.cbGet(GRB.Callback.MIPSOL_OBJ)))
        elapsed = time.time() - self._start
        self.best_solution = queens
        print(f"  [IMPROVING #{self._count}] {obj} queens @ {elapsed:.1f}s")

        if self._checkpoint_path:
            now = time.time()
            if (self._last_ckpt_obj - obj >= 2
                    or now - self._last_ckpt_time >= 60.0):
                try:
                    _save_checkpoint(self._checkpoint_path, queens, self._N)
                    self._last_ckpt_time = now
                    self._last_ckpt_obj  = obj
                except OSError as e:
                    print(f"[CHECKPOINT] Write failed: {e}")


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(
    N: int,
    attackers: dict[tuple, list[tuple]],
    *,
    env: gp.Env | None = None,
    hard_upper_bound: int | None = None,
    hard_lower_bound: int | None = None,
    hint_queens: list[tuple] | None = None,
    use_symmetry_breaking: bool = True,
    fixed_positions: set[tuple] | None = None,
    forbidden_positions: set[tuple] | None = None,
) -> tuple[gp.Model, dict, gp.Var]:
    model = gp.Model(f"queen3d_N{N}", env=env)
    model.Params.OutputFlag = 0

    cells = list(itertools.product(range(N), repeat=3))
    board = {
        cell: model.addVar(vtype=GRB.BINARY, name=f"q_{cell[0]}_{cell[1]}_{cell[2]}")
        for cell in cells
    }

    # EXPLICIT INTEGER OBJECTIVE: Forces Gurobi to strictly consider whole-number bounds.
    total_queens = model.addVar(vtype=GRB.INTEGER, name="TotalQueens")
    model.addConstr(total_queens == gp.quicksum(board.values()), name="link_obj")

    for cell in (fixed_positions or set()):
        board[cell].LB = 1
        board[cell].UB = 1
    for cell in (forbidden_positions or set()):
        board[cell].LB = 0
        board[cell].UB = 0

    for target in cells:
        model.addConstr(
            gp.quicksum(board[src] for src in attackers[target]) >= 1,
            name=f"dom_{target[0]}_{target[1]}_{target[2]}",
        )

    if hard_upper_bound is not None:
        total_queens.UB = hard_upper_bound
    if hard_lower_bound is not None:
        total_queens.LB = hard_lower_bound

    if hint_queens is not None:
        hint_set = set(hint_queens)
        for cell in cells:
            board[cell].Start = 1.0 if cell in hint_set else 0.0

    if use_symmetry_breaking and N > 2 and not fixed_positions:
        add_fundamental_domain_pin(model, board, cells, N)
        fd = set(fundamental_domain(N))
        for cell in cells:
            board[cell].BranchPriority = 1 if cell in fd else 0

    return model, board, total_queens


# ---------------------------------------------------------------------------
# Gurobi solver parameters
# ---------------------------------------------------------------------------

def _apply_solver_params(
    model: gp.Model,
    *,
    num_workers: int,
    time_limit: float | None,
    verbose: bool,
    mode: str,
) -> None:
    model.Params.Threads    = num_workers
    model.Params.MIPGap     = 0.0
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.Presolve       = 2   
    model.Params.PreSparsify    = 1
    model.Params.CoverCuts      = 2
    model.Params.Cuts           = 2
    model.Params.CliqueCuts     = 2
    model.Params.GomoryPasses   = 3
    model.Params.Method         = 2
    model.Params.NodeMethod     = 1
    model.Params.BranchDir      = 1   
    model.Params.VarBranch      = 0   

    if mode == 'optimize':
        model.Params.MIPFocus   = 0
        model.Params.Heuristics = 0.2
    else:
        model.Params.MIPFocus   = 3
        model.Params.Heuristics = 0.05

    if time_limit is not None:
        model.Params.TimeLimit = float(time_limit)


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def solve(
    N: int,
    *,
    mode: str = 'optimize',
    prove_target: int | None = None,
    hint_queens: list[tuple] | None = None,
    time_limit: float | None = None,
    use_symmetry_breaking: bool = True,
    num_workers: int = 8,
    fixed_positions: set[tuple] | None = None,
    forbidden_positions: set[tuple] | None = None,
    verbose: bool = True,
    checkpoint_path: str | None = None,
    hard_lower_bound: int | None = None,
    attackers: dict[tuple, list[tuple]] | None = None,
) -> SolveResult:
    assert mode in ('optimize', 'prove')
    if mode == 'prove':
        assert prove_target is not None

    if attackers is None:
        attackers = precompute_attackers(N)

    if mode == 'optimize':
        hard_ub        = len(hint_queens) if hint_queens else None
        effective_hint = hint_queens
    else:
        hard_ub = prove_target
        if hint_queens and len(hint_queens) <= prove_target:
            effective_hint = hint_queens
        elif hint_queens:
            effective_hint = trim_hint_by_coverage(hint_queens, attackers, prove_target)
        else:
            effective_hint = None

    with make_gurobi_env(verbose=verbose) as env:
        model, board, total_queens = build_model(
            N, attackers,
            env                   = env,
            hard_upper_bound      = hard_ub,
            hard_lower_bound      = hard_lower_bound,
            hint_queens           = effective_hint,
            use_symmetry_breaking = use_symmetry_breaking,
            fixed_positions       = fixed_positions,
            forbidden_positions   = forbidden_positions,
        )

        _apply_solver_params(
            model,
            num_workers = num_workers,
            time_limit  = time_limit,
            verbose     = verbose,
            mode        = mode,
        )

        if mode == 'optimize':
            model.setObjective(total_queens, GRB.MINIMIZE)

        cells = list(itertools.product(range(N), repeat=3))

        callback: SolutionLogger | None = None
        if checkpoint_path and mode == 'optimize':
            callback = SolutionLogger(board, cells, N, checkpoint_path=checkpoint_path)

        start = time.time()
        model.optimize(callback) if callback is not None else model.optimize()
        elapsed = time.time() - start

        status = model.Status
        sol_count = model.SolCount
        
        placement = None
        if status == GRB.OPTIMAL or sol_count > 0:
            placement =[c for c in cells if board[c].X > 0.5]

        model.dispose()

        if status == GRB.OPTIMAL:
            if verbose:
                print(f"  [OPTIMAL] {len(placement)} queens ({elapsed:.2f}s)")
            return SolveResult('optimal', placement, elapsed, N)
        elif sol_count > 0:
            if verbose:
                print(f"[FEASIBLE] {len(placement)} queens ({elapsed:.2f}s)")
            return SolveResult('feasible', placement, elapsed, N)
        elif status == GRB.INFEASIBLE:
            if verbose:
                print(f"  [INFEASIBLE] Proven in {elapsed:.2f}s")
            return SolveResult('infeasible', None, elapsed, N)
        else:
            if verbose:
                print(f"  [UNKNOWN/TIMEOUT] status={status}  {elapsed:.2f}s")
            return SolveResult('unknown', None, elapsed, N)


# ---------------------------------------------------------------------------
# Parallel multi-N optimization
# ---------------------------------------------------------------------------

def _solve_worker(args: dict) -> SolveResult:
    return solve(**args)


def parallel_optimize_range(
    ns: list[int],
    workers_per_job: int = 4,
    time_limit: float | None = None,
    hint_map: dict[int, list[tuple]] | None = None,
) -> dict[int, SolveResult]:
    if hint_map is None:
        hint_map = {}
    for n in ns:
        if n not in hint_map:
            hint = get_hint_for_n(n, canonicalize=True)
            if hint is not None:
                hint_map[n] = hint

    jobs =[
        dict(N=n, mode='optimize', hint_queens=hint_map.get(n),
             time_limit=time_limit, num_workers=workers_per_job,
             use_symmetry_breaking=True, verbose=False)
        for n in ns
    ]

    print(f"\nParallel optimize: N={ns}, {workers_per_job} workers/job")
    print("=" * 60)

    results: dict[int, SolveResult] = {}
    with ProcessPoolExecutor(max_workers=len(ns)) as executor:
        future_to_n = {executor.submit(_solve_worker, job): n
                       for job, n in zip(jobs, ns)}
        with tqdm(total=len(ns), desc="Solving N values", unit="N") as pbar:
            for future in as_completed(future_to_n):
                n      = future_to_n[future]
                result = future.result()
                results[n] = result
                q = len(result.placement) if result.placement else '—'
                tqdm.write(f"  N={n}: γ = {q} [{result.status}]")
                pbar.update(1)
    return results


# ---------------------------------------------------------------------------
# Decomposed lower-bound prover
# ---------------------------------------------------------------------------

def _decomposed_worker(args: dict) -> tuple[tuple, SolveResult]:
    first_queen: tuple  = args.pop('first_queen')
    earlier_cells: list = args.pop('earlier_cells')
    
    # Extract use_symmetry_breaking from args so it isn't passed twice in **args
    usb = args.pop('use_symmetry_breaking', False)
    
    result = solve(
        fixed_positions       = {first_queen},
        forbidden_positions   = set(earlier_cells),
        use_symmetry_breaking = usb,
        **args,
    )
    return (first_queen, result)


def decomposed_prove(
    N: int,
    prove_target: int,
    *,
    n_processes: int | None = None,
    workers_per_process: int | None = None,
    time_limit_per_job: float | None = None,
    max_first_queen_index: int | None = None,
    use_fundamental_domain: bool = False,
    hint_queens: list[tuple] | None = None,
) -> bool | list[tuple] | None:
    total_cpus = mp.cpu_count()

    if n_processes is None:
        n_processes = max(1, total_cpus // 2)
    if workers_per_process is None:
        workers_per_process = max(1, total_cpus // n_processes)

    print(f"  Precomputing attackers for N={N}...")
    attackers = precompute_attackers(N)

    cells = list(itertools.product(range(N), repeat=3))
    cell_to_lex_idx = {c: i for i, c in enumerate(cells)}

    if use_fundamental_domain:
        all_candidates = fundamental_domain(N)
        label = f"fundamental domain ({len(all_candidates)} cells)"
    else:
        all_candidates = cells
        label = f"full ({N**3} cells)"

    candidates = (
        all_candidates[:max_first_queen_index]
        if max_first_queen_index else all_candidates
    )

    print(f"\n{'=' * 60}")
    print(f"  DECOMPOSED PROVE  N={N}  target ≤ {prove_target}")
    print(f"  Candidates: {label}")
    print(f"  {len(candidates)} subproblems | {n_processes} procs × "
          f"{workers_per_process} threads  (total={n_processes*workers_per_process}/{total_cpus} CPUs)")
    if max_first_queen_index:
        print(f"  *** PARTIAL: first {max_first_queen_index} candidates ***")
    print(f"{'=' * 60}")

    jobs =[
        dict(
            N            = N,
            mode         = 'prove',
            prove_target = prove_target,
            first_queen  = p,
            earlier_cells= cells[:cell_to_lex_idx[p]],
            hint_queens  = None,
            time_limit   = time_limit_per_job,
            num_workers  = workers_per_process,
            verbose      = False,
            use_symmetry_breaking = False,
            attackers    = attackers,   
        )
        for p in candidates
    ]

    has_unknown = False
    
    executor = ProcessPoolExecutor(max_workers=n_processes)
    try:
        future_to_job = {executor.submit(_decomposed_worker, job): job for job in jobs}
        counts = {"infeasible": 0, "unknown": 0, "feasible": 0}
        with tqdm(total=len(jobs), desc=f"Proving N={N} ≤ {prove_target}q", unit="sub") as pbar:
            for future in as_completed(future_to_job):
                first_queen, result = future.result()

                if result.status in ('optimal', 'feasible'):
                    counts["feasible"] += 1
                    tqdm.write(f"\n  [FEASIBLE] Witness @ {first_queen}: {result.placement}")
                    
                    for f in future_to_job: f.cancel()
                    executor.shutdown(wait=False, cancel_futures=True)
                    
                    return result.placement

                elif result.status == 'unknown':
                    has_unknown = True
                    counts["unknown"] += 1
                    tqdm.write(f"  {first_queen}: TIMEOUT ({result.elapsed:.0f}s)")
                else:
                    counts["infeasible"] += 1

                pbar.update(1)
                pbar.set_postfix(counts)
                
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    if has_unknown:
        print(f"\n[INCONCLUSIVE] Some subproblems timed out.")
        return None

    print(f"\n[PROVEN] All {len(jobs)} subproblems infeasible.")
    print(f"  => γ(Q³_{N}) ≥ {prove_target + 1}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    MAX_N          = 7
    NUM_WORKERS    = mp.cpu_count()
    RESULTS_FILE   = "3d_queen_domination_results.txt"
    CHECKPOINT_DIR = "."

    TIME_LIMITS = {
        1: 60,   2: 60,    3: 60,     4: 300,
        5: 600,  6: 3600,  7: 1440000,  8: 28800,
    }
    DECOMPOSE_TIME_PER_JOB = {
        5: 300,  6: 1800,  7: 1440000,  8: 28800,
    }

    KNOWN_LOWER_BOUNDS = {
        7: 10,
    }

    print("=" * 60)
    print("  Verifying all known optima / witnesses")
    print("=" * 60)
    for n in sorted(KNOWN_OPTIMA):
        entry = KNOWN_OPTIMA[n]
        if not verify_solution(entry["placement"], n):
            raise ValueError(f"KNOWN_OPTIMA[{n}] failed verification!")

    summary: dict[int, SolveResult] = {}

    for n in range(7, MAX_N + 1):
        tl = TIME_LIMITS.get(n, 28800)

        print(f"\n{'=' * 60}")
        print(f"  N = {n}  |  Workers: {NUM_WORKERS}  |  Time limit: {tl}s")
        print(f"{'=' * 60}")

        hint = get_hint_for_n(n, canonicalize=(n > 2))

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"q3d_n{n}_checkpoint.txt")
        ckpt = load_checkpoint(ckpt_path)
        if ckpt is not None:
            ckpt_n, ckpt_queens = ckpt
            if ckpt_n == n and (hint is None or len(ckpt_queens) < len(hint)):
                if verify_solution(ckpt_queens, n):
                    print(f"  [CHECKPOINT] Using {len(ckpt_queens)}-queen checkpoint")
                    hint = ckpt_queens
                    if n > 2:
                        hint = canonicalize_placement(hint, n)

        if hint:
            print(f"  Hint: {len(hint)} queens")

        lb = KNOWN_LOWER_BOUNDS.get(n)
        if lb:
            print(f"  Injecting Proven Lower Bound: >= {lb} queens")

        result = solve(
            N                    = n,
            mode                 = 'optimize',
            hint_queens          = hint,
            use_symmetry_breaking= True,
            time_limit           = tl,
            num_workers          = NUM_WORKERS,
            checkpoint_path      = ckpt_path if n >= 5 else None,
            hard_lower_bound     = lb,
        )

        if result.status == 'feasible' and result.placement is not None:
            best = len(result.placement)
            print(f"\n  Optimization found {best} queens but could not prove optimal.")
            print(f"  Launching decomposed proof: ≤ {best - 1} queens infeasible?")

            t0 = time.time()
            proven = decomposed_prove(
                N                    = n,
                prove_target         = best - 1,
                use_fundamental_domain = True,
                time_limit_per_job   = DECOMPOSE_TIME_PER_JOB.get(n, 7200),
                hint_queens          = result.placement,
            )
            t_proof = time.time() - t0

            if proven is True:
                result = SolveResult('optimal', result.placement,
                                     result.elapsed + t_proof, n)
            elif isinstance(proven, list):
                print(f"  Decomposed proof found a better witness!")
                result = SolveResult('feasible', proven,
                                     result.elapsed + t_proof, n)

        save_result_to_txt(result, filepath=RESULTS_FILE, append=(n > 1))
        summary[n] = result

        if result.status == 'optimal':
            print(f"\n  ✓ PROVEN: γ(Q³_{n}) = {len(result.placement)}")
        elif result.status == 'feasible':
            print(f"\n  ~ BEST: {len(result.placement)} queens (not proven optimal)")
        else:
            print(f"\n  ✗ {result.status.upper()}")

    print(f"\n\n{'=' * 60}")
    print("  SUMMARY — 3D Queen Domination γ(Q³_N)")
    print(f"{'=' * 60}")
    print(f"  {'N':>3}  {'γ':>4}  {'Status':<10}  {'Time':>12}")
    print(f"  {'—'*3}  {'—'*4}  {'—'*10}  {'—'*12}")
    for n in sorted(summary):
        r = summary[n]
        q = str(len(r.placement)) if r.placement else '—'
        print(f"  {n:>3}  {q:>4}  {r.status:<10}  {r.elapsed:.1f}s")
    print(f"{'=' * 60}")
