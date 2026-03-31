# Optimal Hybrid Basis for BCH-Algebras

This repository accompanies the paper:

> **An Optimal 14-Symbol Hybrid Basis for BCH-Algebras**  
> Mahesh Ramani, Shlok Kumar

## Overview

We present an optimally minimal two-axiom basis for BCH-algebras. The standard presentation requires three axioms — two equations (B1, B3) and one quasi-identity (B2). We prove that both equations can be replaced by the single 14-symbol equation:

```
((xy)z)((x(z0))y) = 0
```

while retaining the standard quasi-identity. We further prove this is *strictly minimal*: no equation of 12 or fewer symbols can serve the same role.

## Main Results

**Axiomatic equivalence.** The system `{G, A3}` is equivalent to the standard BCH basis `{B1, B2, B3}`, where:

- **(G)** `((xy)z)((x(z0))y) = 0`
- **(A3)** `xy = 0` and `yx = 0` implies `x = y`  (identical to the standard quasi-identity B2)

**Strict minimality.** No equation of 12 or fewer symbols can replace `(B1)` and `(B3)` when paired with `(A3)`. This was established by exhaustive generation of all candidate equations up to 12 symbols and automated countermodel construction via Mace4, with zero unverified cases.

## Reproducibility

All results are fully reproducible. Code and verification scripts are publicly archived on Zenodo:

- **Prover9 verification scripts** (equivalence proofs for both directions):  
  https://doi.org/10.5281/zenodo.19339276

- **Python + Mace4 exhaustive minimality search**:  
  https://doi.org/10.5281/zenodo.19339110

## Tools

- [Prover9](https://www.cs.unm.edu/~mccune/prover9/) — automated theorem prover used to verify axiomatic equivalence
- [Mace4](https://www.cs.unm.edu/~mccune/prover9/) — finite model builder used to generate minimality countermodels

## Authors

- **Mahesh Ramani**
- **Shlok Kumar** — whyshlok@gmail.com

## References

1. McCune et al. (2002). Short single axioms for Boolean algebra. *Journal of Automated Reasoning*, 29(1), 1–16.
2. Hu, Q. & Li, X. (1983). On BCH-algebras. *Mathematics Seminar Notes Kobe University*, 11(2), 313–320.
3. Wroński, A. (1983). BCK-algebras do not form a variety. *Mathematica Japonica*, 28(2), 211–213.
