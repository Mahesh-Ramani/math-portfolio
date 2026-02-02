# Network Percolation and Robustness in LEO Satellite Constellations

A simulation framework for analyzing connectivity and robustness profiles of Low Earth Orbit (LEO) satellite constellations through percolation theory.

## Overview

This project investigates how structured Walker Delta configurations contrast with randomized Stochastic distributions across multiple design parameters. The analysis reveals a masking effect where network redundancy conceals structural vulnerabilities to targeted attacks until a critical failure threshold is reached.

**Key Findings:**
- Maximum degree (links per satellite) is the dominant robustness factor (r = 0.99 correlation with plane attack resilience)
- Plane count shows strong negative correlation with attack resilience (r = −0.91)
- Inclination-dependent robustness patterns peak in the 50°–60° range
- Masking threshold varies with maximum degree: ~15% at degree=2 to ~25% at degree=5

## Repository Contents

- `constellation-resilience-simulator.py` — Main simulation framework implementing Walker Delta and Stochastic constellation generation, network topology construction, and attack simulations
- `Network_Percolation_and_Robustness_in_LEO_Satellite_Constellations.pdf` — Full research paper with methodology and analysis
- `result_graphs/` — Parameter sweep visualizations with corresponding run logs

## Methodology

The simulator uses a discrete snapshot approach to analyze constellation behavior:

**Phase Transition Analysis:** Incrementally increases satellite population (50–5000) to identify percolation thresholds where giant components emerge

**Kessler Syndrome Analysis:** Simulates two failure modes on fully-formed networks:
- Random failure (uniform probability node removal)
- Plane attack (entire orbital plane removal)

**Parameter Sweeps:** One-at-a-time analysis of:
- Altitude (300–1100 km)
- Inter-satellite link distance (500–3000 km)
- Maximum degree constraints (2–5 links)
- Constellation density (1000–8000 satellites)
- Phasing, plane count, orbital inclination

## Key Metrics

- **Giant Component Fraction (GCC)** — Proportion of nodes in largest connected cluster
- **Algebraic Connectivity (λ₂)** — Second eigenvalue of graph Laplacian
- **Susceptibility (χ)** — Sum of squared sizes of finite clusters
- **Clustering Coefficient** — Ratio of closed triplets to connected triplets
- **Average Path Length** — Mean shortest path between node pairs

## Configuration

Default baseline configuration (5000-satellite constellation):

```python
SIM_CONFIG = {
    'N_SATS': 5000,
    'PLANES': 71,
    'PHASING': 1,
    'INC_DEG': 53.0,
    'ALT_KM': 550.0,
    'D_MAX_KM': 1000.0,
    'MAX_DEGREE': 4,
    'REPLICATES': 12,
    'CONFIDENCE_LEVEL': 0.95
}
```

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- Pandas
- Seaborn

## Usage

Run the full parameter sweep analysis:

```bash
python constellation-resilience-simulator.py
```

Results are saved to the `results/` directory with timestamped logs, plots, and correlation heatmaps.

## Design Recommendations

Based on empirical findings:

1. **Provision for 4+ inter-satellite terminals** per satellite to materially improve survivability
2. **Prefer fewer planes with more satellites per plane** when attack resilience is prioritized
3. **Select mid-latitude inclinations (50°–60°)** for balanced resilience and coverage
4. **Design ISL hardware for 1000–1500 km range** to capture most robustness gains

**Operational insight:** The masking threshold (15–25% removal depending on configuration) represents a deceptively redundant region where structural vulnerabilities remain hidden. Monitor intermediate removal fractions closely.

## Limitations

- Static snapshot methodology (no dynamic handover modeling)
- One-at-a-time parameter sweeps (no interaction effects)
- Simplified Earth occultation (80 km atmospheric buffer, no refraction)
- Greedy adjacency construction (conservative connectivity baseline)



For detailed methodology, mathematical formulations, and comprehensive analysis, see the [full paper](Network_Percolation_and_Robustness_in_LEO_Satellite_Constellations.pdf).
