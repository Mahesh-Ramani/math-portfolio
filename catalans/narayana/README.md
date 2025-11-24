# Triangle of Narayana Numbers Generator

### Overview
This script is a lightweight adaptation of the algorithm developed for my "Exact Computation of the Catalan Number" paper. It serves as a practical demonstration of the modular architecture discussed in Section 6.3 of that research.

While the main project focused on extreme-scale computation of a single sequence, this script illustrates how the underlying engine—specifically the prime exponent domain approach—can be easily "rescraped" to compute other combinatorial sequences defined by factorial ratios.

### Methodology
Instead of computing the Catalan numbers $C_n$, this script targets the Narayana numbers $T(n, k)$, defined as:

$$T(n, k) = \frac{1}{n} {n \choose k} {n \choose k-1}$$

The algorithm avoids direct factorial computation, which is memory-prohibitive for large inputs. Instead, it:
1.  Sieves primes up to the required bound (using a segmented sieve).
2.  Calculates Exponents for each prime using Legendre’s Formula adapted for the Narayana recurrence (see Mathematical Foundation)
3.  Reconstructs the final integer using `gmpy2` for arbitrary-precision arithmetic.

### Mathematical Formulation
Note that $T(n, k)$ is computed via its prime factorization:

$$ T(n, k) = \prod_{p \le n} p^{E_p} $$

where the exponent $E_p$ is derived from the sum of the prime valuations of the factorials involved:

$$ E_p = L(n, p) + L(n-1, p) - L(k, p) - L(k-1, p) - L(n-k, p) - L(n-k+1, p) $$

Here, $L(m, p)$ denotes Legendre's formula for the $p$-adic valuation of $m!$:

$$ L(m, p) = \sum_{j \ge 1} \left\lfloor \frac{m}{p^j} \right\rfloor $$

### Usage
This is included in the portfolio as a fun tidbit to verify that the factorization logic generalizes cleanly to efficient computation of the triangle of Narayana numbers.

**Dependencies:** `gmpy2`
**Output:** Generates a b-file format (index value) text file.
