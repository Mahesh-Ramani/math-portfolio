from math import isqrt, log10, floor
from time import perf_counter
from collections import defaultdict
import os
from multiprocessing import Pool, cpu_count

def _sieve_segment_worker(args):
    low, high, initial_primes = args
    segment_primes = []
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
            step = p
            count = ((seg_size - 1 - start) // step) + 1
            seg_flags[start : seg_size : step] = b'\x00' * count
    for i, f in enumerate(seg_flags):
        if f:
            segment_primes.append(low + 2 * i)
    return segment_primes

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
            step = p
            if first <= max_idx:
                flags[first : max_idx + 1 : step] = b'\x00' * (((max_idx - first) // step) + 1)
    result = [2]
    result.extend(2 * i + 3 for i, f in enumerate(flags) if f)
    return result

def segmented_sieve(max_n, pool=None):
    if max_n < 2:
        return []
    if max_n == 2:
        return [2]
    sqrt_n = isqrt(max_n)
    initial_primes = simple_sieve(sqrt_n)
    all_primes = list(initial_primes)
    chunk_size = max(sqrt_n, 65536) #okay guess
    low = sqrt_n + 1
    if low % 2 == 0:
        low += 1

    segments = []
    while low <= max_n:
        high = min(low + 2 * chunk_size - 1, max_n)
        if high % 2 == 0:
            high -= 1
        if low <= high:
            segments.append((low, high))
        low = high + 2

    if pool and segments:
        worker_args = [(low, high, initial_primes) for low, high in segments]
        results = pool.map(_sieve_segment_worker, worker_args)
        for segment_primes in results:
            all_primes.extend(segment_primes)
    else:
        for low, high in segments:
            all_primes.extend(_sieve_segment_worker((low, high, initial_primes)))

    return all_primes

def legendre_exponent(n, p):
    exp = 0
    pk = p
    while pk <= n:
        exp += n // pk
        pk *= p
    return exp

def catalan_prime_factorization(n, significant_digits=12, out_dir=".", num_workers=None):
    start = perf_counter()

    if n <= 1:
        elapsed = perf_counter() - start
        return ("1e+0", elapsed, None, {"digits": 1, "primes_count": 0, "n": n, "file_path": None})

    if num_workers is None:
        num_workers = cpu_count()
    
    pool = Pool(processes=num_workers) if num_workers > 1 else None
    
    fpath = None
    log_sum = 0.0
    est_digits = 1
    total_primes = 0

    try:
        max_val = 2 * n
        primes = segmented_sieve(max_val, pool)
        total_primes = len(primes)
        
        split = 0
        for i, p in enumerate(primes):
            if p > n:
                split = i
                break
        
        exponents = {}
        for p in primes[:split]:
            e_2n = legendre_exponent(max_val, p)
            e_n = legendre_exponent(n, p)
            e_binom = e_2n - 2 * e_n
            if e_binom != 0:
                exponents[p] = e_binom
        
        for p in primes[split:]:
            exponents[p] = 1
        
        divisor = n + 1
        temp = divisor
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                val = 0
                while temp % p == 0:
                    temp //= p
                    val += 1
                if p in exponents:
                    exponents[p] -= val
                    if exponents[p] == 0:
                        del exponents[p]
                else:
                    exponents[p] = -val
        if temp > 1:
            remaining = temp
            if remaining in exponents:
                exponents[remaining] -= 1
                if exponents[remaining] == 0:
                    del exponents[remaining]
            else:
                exponents[remaining] = -1
        
        exp_groups = defaultdict(list)
        for p, e in exponents.items():
            if e > 0:
                log_sum += e * log10(p)
                exp_groups[e].append(p)
            elif e < 0:
                raise ValueError(f"Negative exponent for p={p}, e={e}")
        
        est_digits = int(floor(log_sum)) + 1 if log_sum > 0 else 1
        
        try:
            os.makedirs(out_dir, exist_ok=True)
            fpath = os.path.join(out_dir, f"catalan_{n}_factorization.txt")
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(f"# Prime factorization of Catalan({n})\n")
                for e in sorted(exp_groups):
                    ps = exp_groups[e]
                    f.write(f"# exponent={e} count={len(ps)}\n")
                    # Sort primes within each group for consistent output
                    ps.sort()
                    for i in range(0,len(ps), 20):
                        f.write(" ".join(str(p) for p in ps[i:i+20]) + "\n")
        except IOError as e:
            print(f"Warning: Could not write factorization file to {fpath}. Error: {e}")

    finally:
        if pool:
            pool.close()
            pool.join()

    sci_exp = int(floor(log_sum)) if log_sum > 0 else 0
    frac = log_sum - sci_exp if log_sum > 0 else 0.0
    coef = 10 ** frac
    leading = int(coef * (10 ** (significant_digits - 1)))
    lead_str = str(leading).zfill(significant_digits)
    sci_not = f"{lead_str[0]}.{lead_str[1:]}e+{sci_exp}"
    elapsed = perf_counter() - start
    
    meta = {
        "digits": est_digits,
        "estimated_log10": log_sum,
        "primes_count": total_primes,
        "n": n,
        "file_path": fpath
    }
    return (sci_not, elapsed, None, meta)

if __name__ == "__main__":
    test_n = 2050572903 #can be adjusted
    result, time_taken, _, info = catalan_prime_factorization(test_n, significant_digits=12, out_dir=".")
    
    print(f"\nn = {test_n}")
    if info.get('file_path'):
        print(f"Saved prime factorization to: {info['file_path']}")
    else:
        print("Prime factorization file was not saved.")
        
    print(f"Estimated decimal digits of C_n: {info['digits']:,}")
    print(f"Primes up to 2n: {info['primes_count']:,}")
    print(f"Scientific notation (~12 sig digits): {result}")
    print(f"Elapsed time: {time_taken:.6f} seconds")



