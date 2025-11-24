"""
Compute Narayana numbers T(n,k) using Legendre-product formula with segmented sieve. 
This should work moderately okay for larger n as well.
Mahesh Ramani, November 24, 2025
"""

from math import isqrt
from time import perf_counter
import gc

import gmpy2
from gmpy2 import mpz


# ===== CONFIGURATION - Edit these values =====
NMAX = 405                    # Maximum row n
OUTPUT_FILE = None            # Output file path (None = auto-generated name)
FLUSH_EVERY = 0               # Flush every N entries (0 = disabled)
GC_EVERY = 128                # Garbage collect every N rows
VERBOSE = True                # Show progress messages
# =============================================


def simple_sieve(upper_bound):
    if upper_bound < 2:
        return []
    if upper_bound == 2:
        return [2]
    
    max_idx = (upper_bound - 3) // 2
    flags = bytearray(b'\x01') * (max_idx + 1)
    sqrt_bound = isqrt(upper_bound)
    
    for i in range((sqrt_bound - 3) // 2 + 1):
        if flags[i]:
            p = 2 * i + 3
            first = (p * p - 3) // 2
            if first <= max_idx:
                flags[first : max_idx + 1 : p] = b'\x00' * (((max_idx - first) // p) + 1)
    
    return [2] + [2 * i + 3 for i, f in enumerate(flags) if f]


def _sieve_segment(low, high, initial_primes):
    seg_size = (high - low) // 2 + 1
    seg_flags = bytearray(b'\x01') * seg_size
    
    for p in initial_primes:
        if p == 2:
            continue
        first_mult = ((low + p - 1) // p) * p
        if first_mult % 2 == 0:
            first_mult += p
        if first_mult < p * p:
            first_mult = p * p
        if first_mult <= high:
            start = (first_mult - low) // 2
            seg_flags[start : seg_size : p] = b'\x00' * (((seg_size - 1 - start) // p) + 1)
    
    return [low + 2 * i for i, f in enumerate(seg_flags) if f]


def segmented_sieve(max_n):
    if max_n < 2:
        return []
    if max_n == 2:
        return [2]
    
    sqrt_n = isqrt(max_n)
    initial_primes = simple_sieve(sqrt_n)
    all_primes = list(initial_primes)
    chunk_size = max(sqrt_n, 65536)
    low = sqrt_n + 1 if sqrt_n % 2 == 0 else sqrt_n + 2

    while low <= max_n:
        high = min(low + 2 * chunk_size - 1, max_n)
        if high % 2 == 0:
            high -= 1
        if low <= high:
            all_primes.extend(_sieve_segment(low, high, initial_primes))
        low = high + 2

    return all_primes


def precompute_L_table(nmax, primes):
    L = {}
    for p in primes:
        if p > nmax:
            break
        arr = [0] * (nmax + 1)
        for m in range(1, nmax + 1):
            arr[m] = (m // p) + arr[m // p]
        L[p] = arr
    return L


def narayana_by_legendre(n, k, L_table, primes):
    res = mpz(1)
    
    for p in primes:
        if p > n:
            break
        
        Lp = L_table.get(p)
        if Lp:
            E = Lp[n] + Lp[n - 1] - Lp[k] - Lp[k - 1] - Lp[n - k] - Lp[n - k + 1]
            if E > 0:
                res *= mpz(p) ** E
    
    return res


def write_bfile(nmax, out_path=None, flush_every=0, gc_every=128, verbose=True):
    if out_path is None:
        out_path = f"narayana_bfile_1..{nmax}.txt"

    start = perf_counter()
    
    if verbose:
        print(f"Generating primes up to {nmax}...")
    primes = segmented_sieve(nmax) if nmax > 10000 else simple_sieve(nmax)
    sieve_time = perf_counter() - start
    
    if verbose:
        print(f"Generated {len(primes)} primes in {sieve_time:.3f}s")
        print("Precomputing L-table...")
    
    L_table = precompute_L_table(nmax, primes)
    table_time = perf_counter() - start - sieve_time
    
    if verbose:
        print(f"L-table computed in {table_time:.3f}s")

    entries_written = 0
    idx = 1

    with open(out_path, "w", encoding="utf-8") as out:
        for n in range(1, nmax + 1):
            half = (n + 1) // 2
            row_vals = [None] * (n + 1)
            
            for k in range(1, half + 1):
                val = narayana_by_legendre(n, k, L_table, primes)
                row_vals[k] = val
                other_k = n + 1 - k
                if other_k != k:
                    row_vals[other_k] = val

            for k in range(1, n + 1):
                out.write(f"{idx} {row_vals[k]}\n")
                entries_written += 1
                idx += 1
                if flush_every > 0 and entries_written % flush_every == 0:
                    out.flush()
            
            if verbose and (n % 10 == 0 or n <= 5):
                print(f"Row {n}/{nmax} complete ({entries_written} entries, {perf_counter() - start:.2f}s)")

            if gc_every and n % gc_every == 0:
                gc.collect()

    elapsed = perf_counter() - start
    if verbose:
        print(f"\nWrote {entries_written} entries to '{out_path}'")
        print(f"Total: {elapsed:.3f}s (sieve: {sieve_time:.3f}s, table: {table_time:.3f}s, compute: {elapsed - sieve_time - table_time:.3f}s)")
    
    return out_path, elapsed


if __name__ == "__main__":
    out, t = write_bfile(
        nmax=NMAX,
        out_path=OUTPUT_FILE,
        flush_every=FLUSH_EVERY,
        gc_every=GC_EVERY,
        verbose=VERBOSE
    )
    print(f"Output: {out}")
                                                                                                                                                                                               
