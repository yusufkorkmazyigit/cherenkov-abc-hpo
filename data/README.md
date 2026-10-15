# Data (not included)

The Geant4 simulation data belongs to the Erciyes Neutrino Research Group (ENRG) and is **not distributed with this repository**. Members of the group can place the two files here:

| File | Rows | Used by |
|---|---:|---|
| `Balanced700kDataset.csv` | 700,000 | `experiment.py`, `analysis.ipynb` |
| `Unbalanced1MDataset.csv` | 1,167,380 | `experiment.py`, `leakage_check.py`, `analysis.ipynb` |

Each file needs at least these columns (one row per detected photon):

| Column | Meaning |
|---|---|
| `PMTno` | PMT number that detected the photon |
| `Time` | Hit time (ns) |
| `Energy` | Photon energy (eV) |
| `isCherenkov` | Target: 1 = Cherenkov, 0 = scintillation |

A different folder can be used with `DATA_DIR=/path/to/data python experiment.py`. Everything in this folder except this README is ignored by git.
