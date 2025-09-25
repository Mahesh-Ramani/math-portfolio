Check out factorials.pdf on information about n!.py.

n!.py outputs the approximate value of n! based on a given value of n.

Addendum to factorials.pdf:

The asymptotic cost I quoted hides an important detail -- it depends on the cost of big-integer multiplication, so state your bound in terms of M(b) (the cost to multiply two b-bit integers) or make explicit which multiplication model you assume. In practice I use gmpy2 (GMP) for arithmetic, so GMP’s automatic switch to sub-quadratic methods (Karatsuba/Toom/FFT-style for large sizes) means M(b) is effectively much better than schoolbook and the algorithm runs substantially faster than a naive quadratic estimate would suggest — but you should still use a balanced product tree so GMP multiplies operands of similar size, and use pow(product, k) for grouped-exponentiation to leverage GMP’s optimized routines. My practical claim about n = 5×10^9 is realistic only because I return scientific notation (mantissa + exponent) rather than the full decimal expansion — for reference log10((5·10^9)!) ≈ 4.63×10^10, which would be ~1.54×10^11 bits (≈19.3 GB) if you tried to store the entire integer — but if you only need a compact, accurate leading mantissa, the program already does that, so you're good.
