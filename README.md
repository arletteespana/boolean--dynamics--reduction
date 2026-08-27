# Boolean Network Reduction Benchmark

Code and data used for the numerical experiments reported in:

"A Dominant-Vertex Approach to Reducing Cancer-Related Boolean Networks"

## Methods

- Dominant Vertices (DV)
- Node Elimination (NE)
- Two-Step Reduction (TS)
- Leaf-Node Removal (LR)

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

## Running the reduction methods

All methods receive a Boolean network in `.bnet` format as input. The scripts are located in:

```text
src/methods/
```

The examples below assume the input network is:

```text
networks/acute_myeloid_leukemia.bnet
```

and that generated models will be stored under:

```text
results/models/acute_myeloid_leukemia/
```

### 1. Original model

The original model is copied without modifying its Boolean rules. It is used as the baseline for comparison with the reduced models.

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

### 2. Dominant Vertices

The Dominant Vertices method identifies all dominant sets of minimum cardinality and constructs the induced dynamics associated with each of them.

Because a network may have more than one minimum dominant set, this method receives an **output directory** instead of a single output file.

```bash
python src/methods/dominant_vertices.py 
    networks/acute_myeloid_leukemia.bnet 
    results/models/acute_myeloid_leukemia/dominant_vertices/
```

The method may generate several induced models:

```text
results/models/acute_myeloid_leukemia/dominant_vertices/
├── dominant_vertices_01.bnet
├── dominant_vertices_02.bnet
├── ...
└── dominant_vertices_summary.csv
```

Each `.bnet` corresponds to the induced dynamics associated with one minimum dominant set.

The summary file contains information such as the dominant-set size, depth, recurrence length, and dimension of the induced system.

---

### 3. Node Elimination

The Node Elimination method iteratively removes non-autoregulated variables and propagates their Boolean rules into the functions of their targets.

```bash
python src/methods/node_elimination.py 
    networks/acute_myeloid_leukemia.bnet 
    results/models/acute_myeloid_leukemia/node_elimination.bnet
```

Output:

```text
results/models/acute_myeloid_leukemia/node_elimination.bnet
```

During execution, the script also reports the original and reduced network sizes, the number of iterations, the elimination order, and the retained variables.

---

### 4. Two-Step Reduction

The Two-Step Reduction applies two successive procedures:

1. elimination of stabilized variables;
2. iterative elimination of eligible simple mediator nodes.

```bash
python src/methods/two_step.py 
    networks/acute_myeloid_leukemia.bnet 
    results/models/acute_myeloid_leukemia/two_step.bnet
```

Output:

```text
results/models/acute_myeloid_leukemia/two_step.bnet
```

The execution report distinguishes the variables eliminated during the stabilized-node step from those removed as simple mediators.

---

### 5. Leaf-Node Removal

Leaf-Node Removal iteratively removes variables with effective out-degree equal to zero.

```bash
python src/methods/leaf_node_removal.py 
    networks/acute_myeloid_leukemia.bnet 
    results/models/acute_myeloid_leukemia/leaf_node_removal.bnet
```

Output:

```text
results/models/acute_myeloid_leukemia/leaf_node_removal.bnet
```

Leaf nodes are removed iteratively because eliminating one layer of leaves can expose new leaf nodes. The script reports each removal round separately.

---

## Resulting files

After running all five methods, the directory for one network will have the following structure:

```text
results/
└── models/
    └── acute_myeloid_leukemia/
        ├── original.bnet
        ├── node_elimination.bnet
        ├── two_step.bnet
        ├── leaf_node_removal.bnet
        │
        └── dominant_vertices/
            ├── dominant_vertices_01.bnet
            ├── dominant_vertices_02.bnet
            ├── ...
            └── dominant_vertices_summary.csv
```

The resulting `.bnet` files can then be used as inputs for the common metrics and dynamical-analysis pipeline.
