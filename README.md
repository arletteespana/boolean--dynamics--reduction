# Boolean Dynamics Reduction

This repository contains the code and Boolean-network models used to reproduce the reduction and dynamical analyses presented in the accompanying study on reduction methods for cancer-related Boolean networks.

The repository compares five approaches:

1. Original model
2. Dominant Vertices
3. Node Elimination
4. Two-Step Reduction
5. Leaf-Node Removal

All models are represented using the `.bnet` format.

---

## Repository structure

```text
boolean-dynamics-reduction/
├── README.md
├── requirements.txt
├── run_all.py
├── analyze_results.py
│
├── networks/
│   ├── acute_myeloid_leukemia.bnet
│   ├── basal_cell_carcinoma.bnet
│   ├── ...
│   └── thyroid_cancer.bnet
│
├── src/
│   ├── methods/
│   │   ├── original.py
│   │   ├── dominant_vertices.py
│   │   ├── node_elimination.py
│   │   ├── two_step.py
│   │   └── leaf_node_removal.py
│   │
│   └── metrics/
│       ├── __init__.py
│       ├── utils.py
│       ├── structural.py
│       ├── attractors.py
│       └── transients.py
│
└── results/
    ├── models/
    └── analysis/
```

The `results/` directory is generated automatically after running the reduction and analysis scripts.

---

## Requirements

Python 3.10 or newer is recommended.

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

The current `requirements.txt` is:

```text
networkx>=3.2
sympy>=1.12
mpbn>=4.4
```

---

## Boolean-network format

Each network is stored as a `.bnet` file.

Each line contains a target variable and its Boolean update rule:

```text
A, B & !C
B, A | C
C, C
```

The supported Boolean operators are:

```text
!   NOT
&   AND
|   OR
```

Constants `0` and `1` are also supported.

---

# Reduction methods

All methods receive an input `.bnet` model and generate one or more reduced `.bnet` models.

The examples below use:

```text
networks/acute_myeloid_leukemia.bnet
```

as the input network.

---

## 1. Original model

The original model is copied without modifying its Boolean rules and is used as the baseline for comparison.

```bash
python src/methods/original.py \
    networks/acute_myeloid_leukemia.bnet \
    results/models/acute_myeloid_leukemia/original.bnet
```

Output:

```text
results/models/acute_myeloid_leukemia/original.bnet
```

---

## 2. Dominant Vertices

The Dominant Vertices method computes all dominant sets of minimum cardinality and constructs the induced Boolean dynamics associated with each minimum dominant set.

Because the minimum dominant set is not necessarily unique, this method may generate several reduced models.

```bash
python src/methods/dominant_vertices.py \
    networks/acute_myeloid_leukemia.bnet \
    results/models/acute_myeloid_leukemia/dominant_vertices/
```

Example output:

```text
results/models/acute_myeloid_leukemia/dominant_vertices/
├── dominant_vertices_01.bnet
├── dominant_vertices_02.bnet
├── ...
└── dominant_vertices_summary.csv
```

The summary file reports information such as:

```text
dominant-set size
dominant set
depth
recurrence length
state dimension
```

For this method, two different notions of size are distinguished:

```text
retained_variables = |U|
state_dimension    = |U| * ell(U)
```

where `U` is a minimum dominant set and `ell(U)` is its recurrence length.

---

## 3. Node Elimination

The Node Elimination method follows the dynamically consistent reduction proposed by Naldi et al. It iteratively removes non-autoregulated variables and substitutes their Boolean functions into the rules of their targets.

```bash
python src/methods/node_elimination.py \
    networks/acute_myeloid_leukemia.bnet \
    results/models/acute_myeloid_leukemia/node_elimination.bnet
```

Output:

```text
results/models/acute_myeloid_leukemia/node_elimination.bnet
```

The execution log reports the original size, reduced size, number of eliminated nodes, removal order, and retained variables.

---

## 4. Two-Step Reduction

The Two-Step Reduction follows the method proposed by Saadatpour, Albert, and Reluga.

It consists of:

1. iterative elimination of stabilized variables;
2. iterative elimination of eligible simple mediator nodes.

```bash
python src/methods/two_step.py \
    networks/acute_myeloid_leukemia.bnet \
    results/models/acute_myeloid_leukemia/two_step.bnet
```

Output:

```text
results/models/acute_myeloid_leukemia/two_step.bnet
```

The execution report distinguishes variables eliminated during the stabilized-node step from variables removed as simple mediators.

---

## 5. Leaf-Node Removal

Leaf-Node Removal iteratively removes nodes with effective out-degree equal to zero.

No Boolean-rule substitution is required because leaf nodes do not regulate any surviving node.

```bash
python src/methods/leaf_node_removal.py \
    networks/acute_myeloid_leukemia.bnet \
    results/models/acute_myeloid_leukemia/leaf_node_removal.bnet
```

Output:

```text
results/models/acute_myeloid_leukemia/leaf_node_removal.bnet
```

Leaf nodes are removed iteratively because removing one layer can expose new leaf nodes.

---

# Running all reduction methods

To run all five methods for every `.bnet` file in the `networks/` directory, execute from the repository root:

```bash
python run_all.py
```

The script automatically finds all input networks and applies every reduction method.

For each network, the resulting directory has the form:

```text
results/
└── models/
    └── acute_myeloid_leukemia/
        ├── original.bnet
        ├── node_elimination.bnet
        ├── two_step.bnet
        ├── leaf_node_removal.bnet
        │
        ├── dominant_vertices/
        │   ├── dominant_vertices_01.bnet
        │   ├── dominant_vertices_02.bnet
        │   ├── ...
        │   └── dominant_vertices_summary.csv
        │
        └── logs/
            ├── original.log
            ├── dominant_vertices.log
            ├── node_elimination.log
            ├── two_step.log
            └── leaf_node_removal.log
```

Custom directories can also be specified:

```bash
python run_all.py \
    --networks-dir networks \
    --methods-dir src/methods \
    --results-dir results/models
```

The script uses a fail-fast strategy: if one method fails for a network, execution stops and the corresponding log can be inspected.

---

# Dynamical and structural analysis

After generating the reduced models, run:

```bash
python analyze_results.py
```

The analysis automatically decides whether to use exhaustive state-space enumeration or Monte Carlo simulation according to the Boolean dimension of each model.

The default settings are:

```text
Exact-dimension threshold : 20
Monte Carlo samples       : 10000
Base random seed          : 2026
```

Therefore:

```text
D <= 20  -> exact exhaustive analysis
D > 20   -> Monte Carlo analysis
```

The decision is made independently for every generated model.

For example, a single biological network may produce:

```text
Original                 D=67  -> Monte Carlo
Dominant Vertices 01     D=18  -> Exact
Node Elimination         D=21  -> Monte Carlo
Two-Step Reduction       D=48  -> Monte Carlo
Leaf-Node Removal        D=35  -> Monte Carlo
```

---

## Changing the analysis threshold

The exact-analysis threshold can be changed with:

```bash
python analyze_results.py --exact-dimension 22
```

The Monte Carlo sample size can also be changed:

```bash
python analyze_results.py \
    --exact-dimension 20 \
    --monte-carlo-samples 50000
```

A different base random seed can be specified with:

```bash
python analyze_results.py --seed 1234
```

A deterministic model-specific seed is derived from the base seed, network name, method, and variant so that Monte Carlo experiments are reproducible.

---

# Evaluation metrics

The benchmark considers structural efficiency, computational performance, asymptotic dynamics, and transient behavior.

## Structural metrics

For each model, the following quantities are computed exactly:

```text
original_variables
retained_variables
state_dimension
retained_fraction
eliminated_fraction
state_space_size
state_space_ratio
log2_state_space_ratio
effective_state_space_reduction
```

The retained fraction is

```text
retained_fraction = N_m / N
```

and the eliminated fraction is

```text
eliminated_fraction = 1 - N_m / N
```

The state-space ratio is

```text
2^(D_m - N)
```

where `D_m` denotes the actual Boolean dimension of the analyzed model.

For Dominant Vertices:

```text
N_m = |U|
D_m = |U| * ell(U)
```

For the other methods:

```text
D_m = N_m
```

---

## Fixed points

Fixed points are computed exactly for every model using Boolean satisfiability rather than exhaustive enumeration.

A fixed point satisfies:

```text
x_i = f_i(x)
```

for every Boolean variable.

The output includes:

```text
fixed_points
fixed_point_time_seconds
fixed_point_count_match
```

`fixed_point_count_match` indicates whether the number of fixed points agrees with the original model.

---

## Exact synchronous analysis

If the Boolean dimension satisfies:

```text
D <= exact_dimension
```

the complete synchronous state space is explored.

The following quantities are obtained exactly:

```text
number of attractors
number of fixed-point attractors
number of periodic attractors
attractor periods
basin sizes
basin fractions
mean transient length
maximum transient length
dynamic analysis time
```

For exact analyses:

```text
analysis_type = exact
attractor_count_is_complete = True
coverage_fraction = 1
```

---

## Monte Carlo synchronous analysis

For models above the exact-dimension threshold, uniformly sampled initial conditions are used.

For each sampled initial state, the synchronous trajectory is followed until a repeated state identifies its asymptotic cycle.

The Monte Carlo analysis reports:

```text
analysis_type
initial_states_analyzed
attractors_observed
fixed_attractors_observed
periodic_attractors_observed
attractor_periods
estimated basin fractions
mean transient length
maximum observed transient length
dynamic analysis time
```

For Monte Carlo analyses:

```text
analysis_type = monte_carlo
attractor_count_is_complete = False
```

The reported number of attractors is therefore the number of attractors observed in the sample and should not be interpreted as a guaranteed complete attractor count.

Basin fractions are accompanied by 95% Wilson confidence intervals.

---

# Analysis output

Running:

```bash
python analyze_results.py
```

creates:

```text
results/
└── analysis/
    ├── structural_metrics.csv
    ├── fixed_point_metrics.csv
    ├── dynamic_metrics.csv
    ├── basin_metrics.csv
    └── summary.csv
```

## `structural_metrics.csv`

Contains structural reduction information for every generated model.

## `fixed_point_metrics.csv`

Contains exact fixed-point counts and fixed-point analysis times.

## `dynamic_metrics.csv`

Contains exact or Monte Carlo synchronous dynamical metrics, depending on model dimension.

## `basin_metrics.csv`

Contains one row per observed attractor, including:

```text
network
method
variant
analysis_type
attractor_id
attractor_signature
period
basin_count_or_sample_count
basin_fraction
basin_fraction_ci95_low
basin_fraction_ci95_high
```

For exact analyses, the basin fraction is exact.

For Monte Carlo analyses, it is an estimate from sampled initial conditions.

## `summary.csv`

Combines the main structural, fixed-point, attractor, transient, and runtime metrics into a single table.

This is the main file intended for comparative tables and figures.

---

# Complete workflow

A complete analysis can be reproduced with:

```bash
pip install -r requirements.txt
python run_all.py
python analyze_results.py
```

The first script generates the original and reduced Boolean models.

The second script computes the structural and dynamical comparison metrics.

---

# Methodological notes

The `.bnet` files represent Boolean update functions, while theoretical preservation guarantees may depend on the update scheme assumed by each reduction method.

The dynamical benchmark implemented in `analyze_results.py` uses synchronous dynamics for attractor, basin, and transient analyses.

Fixed points are update-scheme independent and are therefore compared exactly across all methods.

For large state spaces, Monte Carlo results describe observed attractors and estimated basin properties rather than exhaustive attractor enumeration.

For Dominant Vertices, all minimum dominant sets are retained and analyzed separately instead of selecting one arbitrarily.

---

# References

The implemented reduction methods are based on the following works:

- España, A., Funez, W., and Ugalde, E. (2026). *Dominant Vertices and Attractors' Landscape for Boolean Networks*. Discrete and Continuous Dynamical Systems - Series B. 254–274. 
- Naldi, A., Remy, E., Thieffry, D., and Chaouiya, C. (2011). *Dynamically consistent reduction of logical regulatory graphs*. Theoretical Computer Science, 412, 2207-2218.
- Saadatpour, A., Albert, R., and Reluga, T. C. (2013). *A reduction method for Boolean network models proven to conserve attractors*. SIAM Journal on Applied Dynamical Systems, 12(4), 1997-2011.
- Richardson, K. (2005). *Simplifying Boolean Networks*. Advances in Complex Systems 8.4, 365–381.

---

## Reproducibility

The repository is designed so that all generated models and metrics can be reproduced directly from the `.bnet` files in `networks/`.

Generated files should not be edited manually. To regenerate the complete set of outputs, run:

```bash
python run_all.py
python analyze_results.py
```

---

## Citation

This repository accompanies the work:

España, A., Molés, J., and Posadas-García, Y. (In process). *A Dominant-Vertex Approach to Reducing Cancer-Related Boolean Networks*

If you use any of the code, Boolean network models, reduction procedures, computational methodology, or results provided in this repository, please consider citing this work.

The complete bibliographic reference will be updated upon publication.
