# Exact Computation of Large Catalan Numbers

This repository contains the implementation of a two-phase algorithm for computing extremely large Catalan numbers, as described in the paper "Exact Computation of the Catalan Number C(2,050,572,903)" by me :). The method avoids direct computation of enormous factorials by operating on prime-exponent data.

## Summary

The process is divided into two main phases:

1.  **Phase 1: Factorization (`catalan.py`)**: This script first enumerates all primes up to `2n` using a parallel segmented sieve. It then applies Legendre's formula to determine the exponent for each prime in the factorization of the nth Catalan number, C(n). The resulting prime factors are grouped by their exponents and saved to a text file.

2.  **Phase 2: Reconstruction (`catalanreconstructor.py`)**: This script reads the factorization file generated in Phase 1. It reconstructs the final Catalan number by first calculating the product of primes for each exponent group and then combining these results. To handle the immense size of the numbers involved, it employs a memory-efficient balanced product tree and a hybrid memory/disk strategy for intermediate calculations.

This approach makes it feasible to compute Catalan numbers with billions of digits on commodity hardware.

## Results and Data

The paper details the computation of C(2,050,572,903), a number with over 1.2 billion decimal digits. The output of this computation, including the factorization file and the final reconstructed number in binary format, is available for verification and review.

You can access the data via the following link:
[Catalan C(2,050,572,903) Computation Results](https://drive.google.com/drive/folders/1JIIeN6xsTsAl4cGJ5pPutKi7shUOJAd5?usp=drive_link)

## Code

This project includes two primary Python scripts:

*   `catalan.py`: For generating the prime factorization of C(n).
*   `catalanreconstructor.py`: For reconstructing the full integer value of C(n) from its factorization.

**Dependencies:**
*   Python 3
*   `gmpy2` library for arbitrary-precision arithmetic.

These scripts and the accompanying paper demonstrate a reproducible method for calculating exact Catalan numbers at a previously unreported scale.
