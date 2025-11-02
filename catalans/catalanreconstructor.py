import gmpy2
from collections import defaultdict
from time import perf_counter
from datetime import datetime
import gc
import sys
import os
import hashlib
import tempfile
import uuid
from contextlib import ExitStack

def pairwise_product_reduce(items):
    if not items:
        return gmpy2.mpz(1)
    
    layer = list(items)
    while len(layer) > 1:
        next_layer = []
        it = iter(layer)
        for a in it:
            try:
                b = next(it)
                next_layer.append(gmpy2.mul(a, b))
            except StopIteration:
                next_layer.append(a)
                break
        layer = next_layer
    return layer[0]

def pow_gmp(base, exp):
    return gmpy2.mpz(base) ** int(exp)

def write_mpz_to_file(n, path):
    with open(path, 'wb') as f:
        f.write(gmpy2.to_binary(n))

def read_mpz_from_file(path):
    with open(path, 'rb') as f:
        return gmpy2.from_binary(f.read())

def split_factorization_by_exponent(fpath):
    print("Splitting factorization file by exponent...")
    sys.stdout.flush()

    exp_counts = defaultdict(int)
    temp_files = {}
    
    with open(fpath, 'r', encoding='utf-8') as main_file, ExitStack() as stack:
        file_handles = {}
        current_exp = None
        
        for line in main_file:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith("# exponent="):
                parts = line.split()
                for part in parts:
                    if part.startswith("exponent="):
                        current_exp = int(part.split("=")[1])
                        break
                if current_exp not in file_handles:
                    temp_path = os.path.join(tempfile.gettempdir(), f"exp_{current_exp}_{uuid.uuid4().hex}.tmp")
                    temp_files[current_exp] = temp_path
                    file_handles[current_exp] = stack.enter_context(open(temp_path, 'w', encoding='utf-8'))
            elif not line.startswith("#") and current_exp is not None:
                primes = line.split()
                file_handles[current_exp].write(' '.join(primes) + '\n')
                exp_counts[current_exp] += len(primes)

    total_primes = sum(exp_counts.values())
    print(f"Found {len(exp_counts)} exponent groups with {total_primes:,} total primes.")
    sys.stdout.flush()
    return temp_files, exp_counts

def process_exponent_file(exp, path, chunk_size):
    partial_products = []
    chunk = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            chunk.extend(gmpy2.mpz(p) for p in line.split())
            
            while len(chunk) >= chunk_size:
                product = pairwise_product_reduce(chunk[:chunk_size])
                partial_products.append(product)
                chunk = chunk[chunk_size:]
                
                if len(partial_products) >= 50:
                    print(f"  Consolidating {len(partial_products)} partial products for exponent {exp}...")
                    sys.stdout.flush()
                    partial_products = [pairwise_product_reduce(partial_products)]
                    gc.collect()

    if chunk:
        partial_products.append(pairwise_product_reduce(chunk))
    
    total_product = pairwise_product_reduce(partial_products)
    
    if exp == 1:
        return total_product
    else:
        print(f"  Raising product to power {exp}...")
        sys.stdout.flush()
        return pow_gmp(total_product, exp)

def combine_components_on_disk(components, tmp_dir):
    disk_paths = []
    try:
        for i, comp in enumerate(components):
            path = os.path.join(tmp_dir, f"component_{i}_{uuid.uuid4().hex}.bin")
            write_mpz_to_file(comp, path)
            disk_paths.append(path)
        
        del components
        gc.collect()

        while len(disk_paths) > 1:
            print(f"  Disk multiply round: {len(disk_paths)} files -> {((len(disk_paths)+1)//2)} files")
            sys.stdout.flush()
            
            next_paths = []
            it = iter(disk_paths)
            for path_a in it:
                try:
                    path_b = next(it)
                    num_a = read_mpz_from_file(path_a)
                    num_b = read_mpz_from_file(path_b)
                    
                    product = gmpy2.mul(num_a, num_b)
                    
                    out_path = os.path.join(tmp_dir, f"product_{uuid.uuid4().hex}.bin")
                    write_mpz_to_file(product, out_path)
                    next_paths.append(out_path)
                    
                    del num_a, num_b, product
                    gc.collect()
                    os.remove(path_a)
                    os.remove(path_b)
                except StopIteration:
                    next_paths.append(path_a)
                    break
            disk_paths = next_paths
        
        return read_mpz_from_file(disk_paths[0])
    finally:
        for p in disk_paths:
            try:
                os.remove(p)
            except OSError:
                pass

def hybrid_component_reduction(components, switch_to_disk_threshold, tmp_dir):
    current_layer = list(components)

    while len(current_layer) > 1:
        if len(current_layer) <= switch_to_disk_threshold:
            print(f"\nComponent count is {len(current_layer)} (<= threshold of {switch_to_disk_threshold}).")
            print("Switching to disk-based multiplication for final combination.")
            sys.stdout.flush()
            return combine_components_on_disk(current_layer, tmp_dir)

        print(f"  In-memory multiply round: {len(current_layer)} components -> {((len(current_layer) + 1) // 2)} components")
        sys.stdout.flush()
        
        next_layer = []
        it = iter(current_layer)
        for a in it:
            try:
                b = next(it)
                next_layer.append(gmpy2.mul(a, b))
            except StopIteration:
                next_layer.append(a)
                break
        current_layer = next_layer
        gc.collect()

    if not current_layer:
        return gmpy2.mpz(1)
    return current_layer[0]

def reconstruct_from_factorization(fpath, chunk_size=100000, switch_to_disk_threshold=10):
    start = perf_counter()
    temp_files = {}
    
    try:
        temp_files, exp_counts = split_factorization_by_exponent(fpath)

        components = []
        for i, (exp, path) in enumerate(sorted(temp_files.items())):
            count = exp_counts[exp]
            print(f"\nProcessing exponent {exp} ({i+1}/{len(temp_files)}): {count:,} primes...")
            sys.stdout.flush()
            
            component = process_exponent_file(exp, path, chunk_size)
            components.append(component)
            gc.collect()
            print(f"  Completed exponent {exp}")
            sys.stdout.flush()

        print("\nComputing final product...")
        sys.stdout.flush()
        
        final_number = None

        with tempfile.TemporaryDirectory(prefix="catalan_") as tmp_dir:
            final_number = hybrid_component_reduction(components, switch_to_disk_threshold, tmp_dir)
        
        elapsed = perf_counter() - start
        meta = {
            "bits": final_number.bit_length(),
            "total_primes": sum(exp_counts.values()),
            "exponent_groups": len(exp_counts),
            "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return final_number, elapsed, meta

    finally:
        print("\nCleaning up temporary files...")
        sys.stdout.flush()
        for path in temp_files.values():
            try:
                os.remove(path)
            except OSError:
                pass
        print("Cleanup complete.")
        sys.stdout.flush()

if __name__ == "__main__":
    OUTPUT_FORMAT = 'binary'
    fpath = "%YOURPATH%\catalan_2050572903_factorization.txt" #copy path from the factorization you get from catalan.py

    try:
        print(f"Starting reconstruction at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sys.stdout.flush()
       
        catalan, time_taken, info = reconstruct_from_factorization(fpath, chunk_size=100000, switch_to_disk_threshold=3)
       
        print("\n--- Computation Complete ---")
        print(f"Total bits: {info['bits']:,}")
        print(f"Total primes in factorization: {info['total_primes']:,}")
        print(f"Number of distinct exponents: {info['exponent_groups']}")
        print(f"Time elapsed: {time_taken:.6f} seconds")
        print(f"Completed at: {info['completion_time']}")
        print("--------------------------\n")

        if OUTPUT_FORMAT == 'decimal':
            outfile = fpath.replace("_factorization.txt", "_reconstructed.txt")
            print(f"Writing decimal string to file: {outfile}")
            sys.stdout.flush()
            with open(outfile, 'w') as f:
                f.write(str(catalan))
            print("Decimal write complete.")
        
        elif OUTPUT_FORMAT == 'binary':
            outfile = fpath.replace("_factorization.txt", "_reconstructed.bin")
            print(f"Writing binary data to file: {outfile}")
            sys.stdout.flush()
            binary_data = gmpy2.to_binary(catalan)
            with open(outfile, 'wb') as f:
                f.write(binary_data)
            
            file_hash = hashlib.sha256(binary_data).hexdigest()
            print(f"  Wrote {len(binary_data):,} bytes.")
            print(f"  SHA-256 Hash: {file_hash}")
            print("Binary write complete.")

    except FileNotFoundError:
        print(f"Error: File '{fpath}' not found")
    except Exception as e:
        print(f"An error occurred during reconstruction: {e}")
        import traceback
        traceback.print_exc()


