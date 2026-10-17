# PDB-Quantifier

**An Unsupervised Machine Learning Framework for Reference-Based Protein Structure Quality Assessment and Prioritization**

PDB-Quantifier is a machine learning framework designed to assess and prioritize experimentally determined protein structures based on their conformity to a reference distribution of structural and validation metadata.

The framework employs an Isolation Forest model trained on 25 features retrieved from the RCSB Protein Data Bank (PDB).

## 1. Overview

Selecting suitable experimentally determined protein structures is an important step in structural bioinformatics, molecular docking, and structure-based drug discovery.

PDB-Quantifier provides an unsupervised, reference-based approach for identifying structures whose metadata profiles are consistent with a curated reference dataset.

The framework provides:

- Reference-based structural conformity scores
- Inlier and outlier identification
- Normalized conformity scores between 0 and 1
- Four conformity tiers: low, moderate, acceptable, and high
- Comparison of independent datasets against the training reference distribution

**Important:** Conformity measures similarity to the reference distribution, not independently validated absolute structural quality.

## 2. Datasets

Three datasets were used in the original evaluation.

| Dataset | Role | Structures |
|---|---|---:|
| ADS | Training reference | 85 |
| MTi | Independent evaluation | 27 |
| Random PDB | Independent evaluation | 100 |

The feature datasets are supplied as CSV files.

- `train_ads.csv`
- `test_set_1.csv`
- `test_set_2.csv`

The corresponding scored datasets are also included.

Raw PDB structure files are not required to reproduce training from the supplied CSV files. Dataset preparation scripts expect local PDB files for ADS and MTi.

## 3. Feature Engineering

The model uses 25 numerical features derived from RCSB entry metadata.

The features describe:

- Experimental structure determination methods
- Resolution
- Deposition and release dates
- Structure age
- Polymer and protein entity counts
- Assembly and deposited model counts
- Atom, residue, and nonpolymer counts
- Atom-to-residue and ligand-to-protein ratios
- R-factor statistics
- Clashscore
- Ramachandran and rotamer outliers
- RSRZ outliers
- DCC R and R-free values

Missing numerical feature values are imputed using the training-set mean.

## 4. Model Architecture

The framework uses a scikit-learn pipeline:

1. Mean imputation with `SimpleImputer`
2. Unsupervised anomaly detection with `IsolationForest`
3. Decision-score calculation
4. Reference-based score normalization
5. Conformity-tier assignment

### Isolation Forest Parameters

| Parameter | Value |
|---|---|
| Estimators | 400 |
| Contamination | 0.10 |
| Random state | 42 |
| Parallel jobs | -1 |
| Number of features | 25 |

The 5th and 95th percentiles of training decision scores are used for normalization. The 20th, 50th and 80th percentiles define the four conformity tiers.

These reference thresholds are retained when evaluating independent datasets.

## 5. Experimental Results

| Metric | ADS | MTi | Random PDB |
|---|---:|---:|---:|
| Structures | 85 | 27 | 100 |
| Inlier percentage | 89.41% | 59.26% | 57.00% |
| Mean decision score | 0.0888 | 0.0241 | 0.0011 |
| Mean normalized conformity | 0.6372 | 0.3673 | 0.3210 |

Kolmogorov-Smirnov comparisons of decision-score distributions:

- ADS vs. MTi: D = 0.4148, p = 0.001076
- ADS vs. Random PDB: D = 0.4365, p approximately 2.07e-8

These results demonstrate differences in the distributions of reference-based conformity scores.

They do not establish absolute crystallographic quality or performance against independently annotated quality labels.

## 6. Installation

Python 3 is required.

Create and activate a Python environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The supplied pretrained model was serialized with joblib. Using the recorded dependency versions is recommended for compatibility.

## 7. Reproduce the Experiments

The provided training and evaluation datasets can be used directly:

```bash
python 4_train_the_model.py
```

The script will:

1. Load the training and test datasets.
2. Fit the Isolation Forest pipeline.
3. Calculate reference-based conformity scores.
4. Evaluate the two independent datasets.
5. Save the model and validation summary.
6. Export scored CSV files.

### Generated Files

- `isolationforest_ads.joblib`
- `isolationforest_validation_summary.json`
- `train_ads_scored.csv`
- `test_set_1_scored.csv`
- `test_set_2_scored.csv`

## 8. Dataset Preparation

The original preparation scripts are included:

```bash
python 1_prepare_training_set.py
python 2_prepare_test_set_1.py
python 3_prepare_test_set_2.py
```

The ADS and MTi scripts expect PDB files under:

- `data/ADS/Receptorlib/`
- `data/MTi/Receptorlib/`

The scripts retrieve metadata from the RCSB PDB API.

The random dataset builder samples from the currently available RCSB entry IDs. Therefore, regenerating this dataset at a later date may produce different structures.

For exact experimental reproduction, use the provided CSV datasets.

## 9. Pretrained Model

The trained pipeline is available as:

`isolationforest_ads.joblib`

The serialized artifact contains:

- Fitted preprocessing and Isolation Forest pipeline
- Feature names and ordering
- Reference decision-score quantiles
- Contamination setting
- Random state

Only load model artifacts from trusted sources.

## 10. Repository Contents

```text
PDB-Quantifier/
├── README.md
├── requirements.txt
├── 1_prepare_training_set.py
├── 2_prepare_test_set_1.py
├── 3_prepare_test_set_2.py
├── 4_train_the_model.py
├── isolationforest_ads.joblib
├── isolationforest_validation_summary.json
├── train_ads.csv
├── test_set_1.csv
├── test_set_2.csv
├── train_ads_scored.csv
├── test_set_1_scored.csv
├── test_set_2_scored.csv
└── figures/
```

## 11. Limitations

PDB-Quantifier is an unsupervised reference-conformity framework.

Its limitations include:

- Dependence on the composition of the training reference dataset
- Dependence on the availability of RCSB metadata
- No supervised ground-truth structural-quality labels
- No demonstrated equivalence between conformity scores and absolute physical or experimental quality
- Reference-specific percentile thresholds

It is intended to support structure prioritization rather than replace established structural validation procedures.

## 12. Repository

https://github.com/yauz3/PDB-Quantifier

## 13. Data Source

RCSB Protein Data Bank: https://www.rcsb.org/

RCSB Data API: https://data.rcsb.org/
