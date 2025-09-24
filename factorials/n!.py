from math import isqrt, log10, floor
from time import perf_counter
from collections import defaultdict
import gmpy2

def simple_sieve(upper_bound):
    if upper_bound < 2:
        return []
    
    # basic sieve implementation
    prime_flags = bytearray(b'\x01') * (upper_bound + 1)
    prime_flags[0:2] = b'\x00\x00'
    
    sqrt_bound = isqrt(upper_bound)
    for candidate in range(2, sqrt_bound + 1):
        if prime_flags[candidate]:
            step_val = candidate
            first_composite = candidate * candidate
            prime_flags[first_composite: upper_bound + 1: step_val] = b'\x00' * (((upper_bound - first_composite) // step_val) + 1)
    
    return [idx for idx, is_prime_flag in enumerate(prime_flags) if is_prime_flag]

def segmented_sieve(max_n):
    if max_n < 2:
        return []
    
    sqrt_n = isqrt(max_n)
    initial_primes = simple_sieve(sqrt_n)
    all_primes_list = list(initial_primes)
    
    chunk_size = max(sqrt_n, 32768)  # decent chunk size
    current_low = sqrt_n + 1
    
    while current_low <= max_n:
        current_high = min(current_low + chunk_size - 1, max_n)
        segment_flags = bytearray(b'\x01') * (current_high - current_low + 1)
        
        for prime_val in initial_primes:
            first_multiple = ((current_low + prime_val - 1) // prime_val) * prime_val
            if first_multiple < prime_val * prime_val:
                first_multiple = prime_val * prime_val
            
            for composite_num in range(first_multiple, current_high + 1, prime_val):
                segment_flags[composite_num - current_low] = 0
        
        for offset, flag_val in enumerate(segment_flags):
            if flag_val:
                all_primes_list.append(current_low + offset)
        
        current_low += chunk_size
    
    return all_primes_list

def legendre_exponent(num, prime_p):
    exponent_sum = 0
    power_of_p = prime_p
    
    while power_of_p <= num:
        exponent_sum += num // power_of_p
        power_of_p *= prime_p
    
    return exponent_sum

def product_tree_iter(number_list):
    if not number_list:
        return gmpy2.mpz(1)
    
    current_layer = [gmpy2.mpz(x) for x in number_list]
    
    while len(current_layer) > 1:
        next_layer_list = []
        layer_iterator = iter(current_layer)
        
        for first_num in layer_iterator:
            try:
                second_num = next(layer_iterator)
            except StopIteration:
                next_layer_list.append(first_num)
                break
            next_layer_list.append(gmpy2.mul(first_num, second_num))
        
        current_layer = next_layer_list
    
    return current_layer[0]

def pow_gmp(base_val: int, exponent_val: int):
    return gmpy2.pow(gmpy2.mpz(base_val), int(exponent_val))


def factorial_via_primes_with_mul(input_n, max_full_digits=200_000, significant_digits=12):
    start_time = perf_counter()
    
    if input_n < 2:
        time_elapsed = perf_counter() - start_time
        return ("1e+0", time_elapsed, 1, {"digits": 1, "primes_count": 0, "built_full": True, "n": input_n})

    # Get all primes up to n
    prime_list = segmented_sieve(input_n)
    total_primes = len(prime_list)
    sqrt_n = isqrt(input_n)

    # binary search to find largest prime <= sqrt(n)
    left_bound, right_bound = 0, total_primes - 1
    sqrt_prime_idx = -1
    
    while left_bound <= right_bound:
        middle_idx = (left_bound + right_bound) // 2
        if prime_list[middle_idx] <= sqrt_n:
            sqrt_prime_idx = middle_idx
            left_bound = middle_idx + 1
        else:
            right_bound = middle_idx - 1

    # Group large primes by their quotient k = n // p
    quotient_groups = defaultdict(list)
    for prime_val in prime_list[sqrt_prime_idx+1:]:
        quotient_k = input_n // prime_val
        if quotient_k > 0:
            quotient_groups[quotient_k].append(prime_val)

    large_prime_factors = []     # stores (product, exponent) pairs
    log10_accumulator = 0.0

    # Handle grouped large primes
    for k_val, prime_group in quotient_groups.items():
        group_product = product_tree_iter(prime_group)           # as mpz
        log10_accumulator += k_val * sum(log10(p) for p in prime_group)
        large_prime_factors.append((group_product, k_val))

    # Handle small primes with full Legendre formula
    small_prime_data = []
    for small_prime in prime_list[:sqrt_prime_idx+1]:
        prime_exponent = legendre_exponent(input_n, small_prime)
        if prime_exponent > 0:
            small_prime_data.append((small_prime, prime_exponent))
            log10_accumulator += prime_exponent * log10(small_prime)

    estimated_digits = int(floor(log10_accumulator)) + 1
    should_build_full = estimated_digits <= max_full_digits

    actual_factorial = None
    
    if should_build_full:
        # Build the actual factorial
        factorial_components = []
        
        # Add large prime factors
        for product_pk, exp_k in large_prime_factors:
            factorial_components.append(pow_gmp(product_pk, exp_k))
        
        # Add small prime factors  
        for prime_base, prime_exp in small_prime_data:
            factorial_components.append(pow_gmp(prime_base, prime_exp))
        
        # Compute final product
        factorial_mpz = product_tree_iter(factorial_components)
        actual_factorial = int(factorial_mpz)
        successfully_built = True
    else:
        successfully_built = False

    # Create scientific notation approximation
    total_log = log10_accumulator
    sci_exponent = int(floor(total_log))
    fractional_part = total_log - sci_exponent
    coefficient_val = 10 ** fractional_part
    
    # Format with specified significant digits
    leading_digits = int(coefficient_val * (10 ** (significant_digits - 1)))
    leading_str = str(leading_digits).zfill(significant_digits)
    scientific_notation = f"{leading_str[0]}.{leading_str[1:]}e+{sci_exponent}"

    final_time = perf_counter() - start_time
    metadata = {
        "digits": estimated_digits, 
        "estimated_log10": log10_accumulator, 
        "primes_count": total_primes, 
        "built_full": successfully_built, 
        "n": input_n
    }
    
    return (scientific_notation, final_time, actual_factorial, metadata)

def log10_fact_stirling_mpfr(test_n, prec_bits=200):
    ctx = gmpy2.get_context()
    ctx.precision = prec_bits

    mpn = gmpy2.mpfr(test_n)
    ln10 = gmpy2.log(gmpy2.mpfr(10))

    ln_n = gmpy2.log(mpn)
    term_main = mpn * ln_n - mpn
    term_half = gmpy2.log(2 * gmpy2.const_pi() * mpn) / 2
    inv_n = 1 / mpn
    c12 = gmpy2.mpfr(1) / (12) * inv_n
    c360 = - gmpy2.mpfr(1) / 360 * inv_n**3
    c1260 = gmpy2.mpfr(1) / 1260 * inv_n**5

    ln_fact = term_main + term_half + c12 + c360 + c1260
    # convert to log10
    log10_fact = ln_fact / ln10
    return log10_fact  # mpfr

def percent_error_from_logs(L_est_float, L_ref_mpfr):
    """Return percent error given L_est (float) and L_ref (mpfr)."""
    L_ref = float(L_ref_mpfr)        # small rounding here, but enough to compute delta
    delta = L_est_float - L_ref
    # exact percent:
    pct = (10**delta - 1.0) * 100.0
    # approximate percent:
    approx_pct = (gmpy2.log(gmpy2.mpfr(10)) * gmpy2.mpfr(delta) * 100)
    return pct, float(approx_pct)

# Main execution
if __name__ == "__main__":
    test_n = 10**6  # can adjust this
    result_sci, time_taken, full_factorial, info_dict = factorial_via_primes_with_mul(test_n, max_full_digits=200_000, significant_digits=12)
    
    print(f"\nn = {test_n}")
    print(f"Estimated decimal digits of n!: {info_dict['digits']:,}")
    print(f"Primes up to n: {info_dict['primes_count']:,}")
    print(f"Built full integer: {info_dict['built_full']}")
    print(f"Scientific notation (~{12} sig digits): {result_sci}")
    print(f"Elapsed time: {time_taken:.6f} seconds")

    ref = log10_fact_stirling_mpfr(test_n, prec_bits=300)
    # print("Reference log10(n!) (mpfr):", ref)
    L_est = info_dict['estimated_log10']
    pct_exact, pct_approx = percent_error_from_logs(L_est, ref)
    # print("Delta (L_est - L_ref) =", float(L_est - float(ref)))
    print("Percent error (exact formula):", pct_exact)
    # print("Percent error (linear approx):", pct_approx)
