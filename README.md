# Boolean Network Reduction Benchmark

Code and data used for the numerical experiments reported in:

"A Dominant-Vertex Approach to Reducing Cancer-Related Boolean Networks"

## Methods

- Dominant Vertices (DV)
- Node Elimination (NE)
- Two-Step Reduction (TS)

## Dataset

Fourteen cancer-related Boolean regulatory networks:

1. Acute myeloid leukemia
2. Basal cell carcinoma
3. Bladder cancer
4. Chronic myeloid leukemia
5. Colorectal cancer
6. Endometrial cancer
7. Glioma
8. Melanoma
9. Non-small cell lung cancer
10. Pancreatic cancer
11. Prostate cancer
12. Renal cell carcinoma
13. Small cell lung cancer
14. Thyroid cancer

## Installation

pip install -r requirements.txt

## Reproducing the experiments

python experiments/run_all.py

## Output

Results are stored in:

results/raw/
results/tables/
results/figures/
