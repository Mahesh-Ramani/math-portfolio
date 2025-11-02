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

# --- Core Algorithmic Components ---

def pairwise_product_reduce(items):
    """
    Efficiently computes the product of a list of numbers using a pairwise strategy.
    """
    if not items:
        return gmpy2.mpz(1)
    
    layer = list(items)
    while len(layer) > 1:
        next_layer = []
        it = iter(layer)
        for a in it:
            try:
                # Multiply pairs of items
                b = next(it)
                next_layer.append(gmpy2.mul(a, b))
            except StopIteration:
                # If there's an odd one out, carry it to the next layer
                next_layer.append(a)
                break
        layer = next_layer
    return layer[0]

def pow_gmp(base, exp):
    """Custom power function using gmpy2's overloaded operator."""
    return gmpy2.mpz(base) ** int(exp)


# --- File I/O for Large Numbers ---

def write_mpz_to_file(n, path):
    """Writes a gmpy2.mpz integer to a file in a compact binary format."""
    with open(path, 'wb') as f:
        f.write(gmpy2.to_binary(n))

def read_mpz_from_file(path):
    """Reads a gmpy2.mpz integer from a binary file."""
    with open(path, 'rb') as f:
        return gmpy2.from_binary(f.read())

# --- Main Reconstruction Logic ---

def _split_factorization_by_exponent(fpath):
    """
    First pass: Splits the main factorization file into temporary files, one for each exponent.
    Returns a dictionary mapping each exponent to its corresponding temp file path.
    """
    print("Pass 1: Splitting factorization file by exponent...")
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

def _process_exponent_file(exp, path, chunk_size):
    """
    Second pass: Processes a single exponent's temp file.
    It reads all primes, calculates their product, and raises it to the exponent.
    """
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

def _combine_components_on_disk(components, tmp_dir):
    """
    Writes large number components to disk and multiplies them pairwise
    to keep memory usage minimal.
    """
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


def _hybrid_component_reduction(components, switch_to_disk_threshold, tmp_dir):
    """
    Combines components using a hybrid memory/disk strategy.
    Reduces components in-memory until their count reaches the threshold,
    then switches to a disk-based method for the final, largest multiplications.
    """
    current_layer = list(components)

    while len(current_layer) > 1:
        # Check if we have reached the threshold to switch to disk
        if len(current_layer) <= switch_to_disk_threshold:
            print(f"\nComponent count is {len(current_layer)} (<= threshold of {switch_to_disk_threshold}).")
            print("Switching to disk-based multiplication for final combination.")
            sys.stdout.flush()
            # Hand off the remaining, very large components to the disk-based function
            return _combine_components_on_disk(current_layer, tmp_dir)

        # --- Perform one round of in-memory pairwise multiplication ---
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

    # If the loop finishes, it means the number of components was already 1 or 0
    if not current_layer:
        return gmpy2.mpz(1)
    return current_layer[0]

def reconstruct_from_factorization(fpath, chunk_size=100000, switch_to_disk_threshold=10):
    """
    Reconstructs an integer from its prime factorization file.
    
    Args:
        fpath (str): Path to the factorization file.
        chunk_size (int): Number of primes to process in-memory at a time.
        switch_to_disk_threshold (int): When the number of components to be multiplied
                                        drops to this number, switch from in-memory
                                        to disk-based multiplication to save RAM.
    """
    start = perf_counter()
    temp_files = {}
    
    try:
        temp_files, exp_counts = _split_factorization_by_exponent(fpath)

        components = []
        for i, (exp, path) in enumerate(sorted(temp_files.items())):
            count = exp_counts[exp]
            print(f"\nProcessing exponent {exp} ({i+1}/{len(temp_files)}): {count:,} primes...")
            sys.stdout.flush()
            
            component = _process_exponent_file(exp, path, chunk_size)
            components.append(component)
            gc.collect()
            print(f"  Completed exponent {exp}")
            sys.stdout.flush()

        # --- MODIFIED STEP 3: Use the new hybrid combination strategy ---
        print("\nComputing final product using hybrid memory/disk strategy...")
        sys.stdout.flush()
        
        final_number = None
        # The hybrid function needs a temporary directory in case it needs to switch.
        # This is managed safely with a context manager.
        with tempfile.TemporaryDirectory(prefix="catalan_hybrid_") as tmp_dir:
            final_number = _hybrid_component_reduction(components, switch_to_disk_threshold, tmp_dir)
        
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
    # --- CONFIGURATION ---
    OUTPUT_FORMAT = 'binary'
    fpath = "C:\Coding\catalan_2348957904_factorization.txt"
    # -------------------

    try:
        print(f"Starting reconstruction at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        sys.stdout.flush()
       
        # Call with the new, clearer parameter name. When the number of components
        # to be multiplied drops to 10, the program will automatically switch
        # to the safer disk-based method.
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
            print(f"  SHA-26 Hash: {file_hash}")
            print("Binary write complete.")

    except FileNotFoundError:
        print(f"Error: File '{fpath}' not found")
    except Exception as e:
        print(f"An error occurred during reconstruction: {e}")
        import traceback
        traceback.print_exc()