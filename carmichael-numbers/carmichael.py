from __future__ import annotations
import math, time, bisect
from typing import List, Tuple, Optional, Iterable
import pandas as pd

def is_prime(num: int) -> bool:
    if num < 2:
        return False
    small_ones = (2,3,5,7,11,13,17,19,23,29)
    for thing in small_ones:
        if num % thing == 0:
            return num == thing
    
    temp_d = num - 1
    temp_s = 0
    while temp_d % 2 == 0:
        temp_d //= 2
        temp_s += 1
    
    test_bases = (2, 325, 9375, 28178, 450775, 9780504, 1795265022)
    
    def miller_rabin_check(base_val:int) -> bool:
        if base_val % num == 0:
            return True
        temp_x = pow(base_val, temp_d, num)
        if temp_x == 1 or temp_x == num-1:
            return True
        for _ in range(temp_s-1):
            temp_x = (temp_x * temp_x) % num
            if temp_x == num-1:
                return True
        return False
    
    for base_to_test in test_bases:
        if not miller_rabin_check(base_to_test):
            return False
    return True

def egcd(first_num:int, second_num:int) -> Tuple[int,int,int]:
    if second_num == 0:
        return (first_num,1,0)
    gcd_val, coeff1, coeff2 = egcd(second_num, first_num % second_num)
    return (gcd_val, coeff2, coeff1 - (first_num // second_num) * coeff2)

def modinv(val:int, modulus:int) -> Optional[int]:
    gcd_result, x_coeff, _ = egcd(val, modulus)
    if gcd_result != 1:
        return None
    return x_coeff % modulus


def crt_pair(remainder1:int, mod1:int, remainder2:int, mod2:int) -> Optional[Tuple[int,int]]:
    gcd_val, s_coeff, t_coeff = egcd(mod1, mod2)
    if (remainder2 - remainder1) % gcd_val != 0:
        return None
    
    lcm_val = mod1 // gcd_val * mod2
    k_val = ((remainder2 - remainder1) // gcd_val) % (mod2 // gcd_val)
    multiplier = (s_coeff * k_val) % (mod2 // gcd_val)
    combined_remainder = (remainder1 + mod1 * multiplier) % lcm_val
    return (combined_remainder, lcm_val)

def crt(congruence_list: Iterable[Tuple[int,int]]) -> Optional[Tuple[int,int]]:
    running_remainder, running_modulus = 0, 1
    for ai_val, mi_val in congruence_list:
        result = crt_pair(running_remainder, running_modulus, ai_val % mi_val, mi_val)
        if result is None:
            return None
        running_remainder, running_modulus = result
    return (running_remainder % running_modulus, running_modulus)

def sieve_primes(upper_limit:int) -> List[int]:
    if upper_limit < 2:
        return []
    
    # Using a sieve
    is_prime_array = bytearray(b'\x01') * (upper_limit+1)
    is_prime_array[0:2] = b'\x00\x00'
    
    for prime_candidate in range(2, int(upper_limit**0.5)+1):
        if is_prime_array[prime_candidate]:
            starting_point = prime_candidate * prime_candidate
            step_size = prime_candidate
            is_prime_array[starting_point: upper_limit+1: step_size] = b'\x00' * (((upper_limit - starting_point)//step_size) + 1)
    
    return [idx for idx, val in enumerate(is_prime_array) if val]

def integer_nth_root(number:int, root_n:int) -> int:
    if number < 0:
        raise ValueError("number must be non-negative")
    if number < 2:
        return number
    
    # Initial approximation
    rough_estimate = int(number ** (1.0 / root_n))
    
    # Adjust upward if needed
    if (rough_estimate+1) ** root_n <= number:
        while (rough_estimate+1) ** root_n <= number:
            rough_estimate += 1
        return rough_estimate
    
    # Adjust downward if needed
    while rough_estimate ** root_n > number:
        rough_estimate -= 1
    return rough_estimate


def find_carmichaels(upper_bound:int, prime_list:List[int]=None, verbose:bool=False) -> List[int]:
    if upper_bound < 3:
        return []
    
    if prime_list is None:
        sieve_upper = min(int(1_000_000), max(1000, int(upper_bound ** (1/3) * 10)))
        prime_list = sieve_primes(sieve_upper)
    
    prime_list = sorted(prime_list)
    num_primes = len(prime_list)
    
    # Precompute prefix products
    cumulative_products = [1]
    for prime_val in prime_list:
        cumulative_products.append(cumulative_products[-1] * prime_val)
        if cumulative_products[-1] > upper_bound:
            break
    
    # Figure out max k
    max_factors = 0
    running_product = 1
    for prime_val in prime_list:
        running_product *= prime_val
        if running_product > upper_bound:
            break
        max_factors += 1
    
    max_factors = max(3, max_factors)
    max_factors = min(max_factors, 16)  # Don't go crazy
    
    if verbose:
        print(f"Primes up to {prime_list[-1]}, max_k ~ {max_factors}")
    
    candidate_results = set()

    # Special case for k=3 (more efficient)
    def handle_three_factors():
        for i in range(num_primes):
            p_val = prime_list[i]
            if p_val > integer_nth_root(upper_bound, 3):
                break
            
            q_max_val = integer_nth_root(upper_bound // p_val, 2)
            j_upper_bound = bisect.bisect_right(prime_list, q_max_val) - 1
            
            if j_upper_bound <= i:
                continue
            
            for j in range(i+1, j_upper_bound+1):
                q_val = prime_list[j]
                pq_product = p_val * q_val
                
                if pq_product * (q_val + 1) > upper_bound:
                    continue
                
                # Build congruences
                congruence_pairs = []
                valid_config = True
                
                for divisor in (p_val, q_val):
                    mod_val = divisor - 1
                    quotient = pq_product // divisor
                    inverse_val = modinv(quotient % mod_val, mod_val)
                    if inverse_val is None:
                        valid_config = False
                        break
                    congruence_pairs.append((inverse_val, mod_val))
                
                if not valid_config:
                    continue
                
                crt_result = crt(congruence_pairs)
                if crt_result is None:
                    continue
                
                base_remainder, period = crt_result
                
                # Find first valid r
                t_start = (q_val - base_remainder) // period
                if base_remainder + t_start * period <= q_val:
                    t_start += 1
                
                current_r = base_remainder + t_start * period
                
                while current_r * pq_product <= upper_bound:
                    if current_r > q_val and is_prime(current_r):
                        if (pq_product - 1) % (current_r - 1) == 0:
                            candidate_results.add(pq_product * current_r)
                    current_r += period

    # General case for k >= 4
    def handle_general_case(num_factors:int):
        def get_min_product_from(start_idx:int, count_needed:int) -> int:
            if start_idx + count_needed > len(cumulative_products) - 1:
                temp_product = 1
                items_counted = 0
                current_idx = start_idx
                while items_counted < count_needed and current_idx < num_primes:
                    temp_product *= prime_list[current_idx]
                    if temp_product > upper_bound:
                        return temp_product
                    current_idx += 1
                    items_counted += 1
                return temp_product if items_counted == count_needed else upper_bound+1
            return cumulative_products[start_idx+count_needed] // cumulative_products[start_idx]

        def recursive_search(idx_start:int, chosen_primes:List[int], current_product:int):
            factors_so_far = len(chosen_primes)
            
            if current_product > upper_bound:
                return
            
            factors_remaining = num_factors - factors_so_far
            
            if idx_start >= num_primes and factors_remaining > 0:
                return
            
            # Base case: need one more factor
            if factors_so_far == num_factors-1:
                max_final_factor = upper_bound // current_product
                min_final_factor = chosen_primes[-1] + 1
                
                if min_final_factor > max_final_factor:
                    return
                
                # Build congruence system
                congruence_system = []
                for s_val in chosen_primes:
                    modulus = s_val - 1
                    partial_product = current_product // s_val
                    inv_val = modinv(partial_product % modulus, modulus)
                    if inv_val is None:
                        return
                    congruence_system.append((inv_val, modulus))
                
                crt_solution = crt(congruence_system)
                if crt_solution is None:
                    return
                
                remainder_val, step_size = crt_solution
                
                # Find starting point
                t_initial = (min_final_factor - remainder_val) // step_size
                if remainder_val + t_initial * step_size <= chosen_primes[-1]:
                    t_initial += 1
                
                test_r = remainder_val + t_initial * step_size
                
                while test_r <= max_final_factor:
                    if test_r > chosen_primes[-1] and is_prime(test_r):
                        if (current_product - 1) % (test_r - 1) == 0:
                            candidate_results.add(current_product * test_r)
                    test_r += step_size
                return

            # Recursive case: need more factors
            max_next_factor = integer_nth_root(upper_bound // current_product, factors_remaining)
            max_idx = bisect.bisect_right(prime_list, max_next_factor) - 1
            
            if max_idx < idx_start:
                return
            
            for candidate_idx in range(idx_start, max_idx + 1):
                next_prime = prime_list[candidate_idx]
                
                if chosen_primes and next_prime <= chosen_primes[-1]:
                    continue
                
                # Check if remaining factors can fit
                min_remaining_product = get_min_product_from(candidate_idx + 1, factors_remaining - 1)
                if current_product * next_prime * min_remaining_product > upper_bound:
                    break
                
                recursive_search(candidate_idx + 1, chosen_primes + [next_prime], current_product * next_prime)

        recursive_search(0, [], 1)

    # Run the search
    start_time = time.time()
    handle_three_factors()
    
    for k_val in range(4, max_factors + 1):
        if verbose:
            print(f"searching k={k_val}")
        handle_general_case(k_val)
    
    search_time = time.time() - start_time
    if verbose:
        print(f"search done in {search_time:.3f}s, found {len(candidate_results)} raw candidates")

    # Validate results
    final_results = []
    for candidate in sorted(candidate_results):
        if candidate > upper_bound:
            continue
        
        # Factor the number
        temp_num = candidate
        factor_list = []
        
        for p_test in prime_list:
            if p_test * p_test > temp_num:
                break
            if temp_num % p_test == 0:
                power_count = 0
                while temp_num % p_test == 0:
                    temp_num //= p_test
                    power_count += 1
                if power_count > 1:  # Not square-free
                    factor_list = None
                    break
                factor_list.append(p_test)
        
        if factor_list is None:
            continue
        
        if temp_num > 1:
            factor_list.append(temp_num)
        
        if len(factor_list) < 3:
            continue
        
        # Check Carmichael condition
        carmichael_valid = True
        for p_factor in factor_list:
            if (candidate - 1) % (p_factor - 1) != 0:
                carmichael_valid = False
                break
        
        if carmichael_valid:
            final_results.append(candidate)
    
    final_results.sort()
    return final_results


# Main execution
search_limit = 10**7
print(f"Running Carmichael search up to {search_limit}...")
start_timer = time.time()
carmichael_numbers = find_carmichaels(search_limit, verbose=True)
end_timer = time.time()
print(f"Done in {end_timer-start_timer:.3f}s. Found {len(carmichael_numbers)} Carmichael numbers ≤ {search_limit}.")
print(carmichael_numbers)
