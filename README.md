# Predicting 30-Day Hospital Readmission in Diabetic Inpatients

**Cambridge MSt Healthcare Data Science — Module 5 Assignment**
Author: Mahdi Osman

A methodologically corrected comparison of Logistic Regression, Random Forest, and XGBoost for the prediction of 30-day unplanned readmission in diabetic inpatients, with a fairness audit, decision curve analysis, post-hoc isotonic calibration, threshold optimisation, individual SHAP explanations, and a head-to-head benchmark against the LACE clinical risk score.

---

## Abstract

**Purpose.** Unplanned 30-day hospital readmission is a globally recognised quality indicator and a primary policy target in diabetic inpatients. Comparative evaluations on the Diabetes 130-Hospitals dataset have repeatedly been confounded by methodological flaws including double imbalance correction, label encoding of nominal variables, and pre-split data leakage. This study presents a methodologically corrected comparison of three models, with explicit attention to calibration, threshold selection, and benchmarking against a clinical risk score.

**Methods.** The Diabetes 130-US Hospitals dataset (101,766 encounters, 1999–2008) was preprocessed via patient-level deduplication, hospice exclusion, and ICD-9 grouping, yielding 69,973 encounters (9.0% positive). Logistic regression (LR), random forest (RF), and XGBoost (XGB) were fitted within scikit-learn pipelines with one-hot encoded nominal features and class weighting. Probabilities were post-hoc calibrated using isotonic regression. Models were evaluated on a 20% held-out test set using AUC-ROC, AUPRC, Brier score, decision curve analysis, validation-selected operating thresholds, fairness audit, and per-subgroup calibration. SHAP values were computed via TreeSHAP. Performance was benchmarked against a LACE-proxy clinical risk score.

**Results.** Cross-validated AUC-ROC was 0.649 (LR), 0.654 (RF), and 0.651 (XGB); no significant pairwise differences were detected. Uncalibrated Brier scores (0.155–0.228) substantially exceeded the no-skill baseline of 0.082; isotonic calibration recovered Brier scores to 0.079 across all three models. Decision curve analysis on uncalibrated probabilities showed net benefit below the "treat none" reference, but calibrated probabilities yielded positive net benefit at all clinically plausible thresholds (10–25%). At validation-selected F1-optimal thresholds (~0.11), RF achieved F1 = 0.243 with recall = 0.450. Discharge disposition, prior inpatient visits, age, and length of stay were the dominant predictors; gender ranked 29th of 110 features. Subgroup AUC ranged 0.638–0.809; all 18 model×subgroup combinations beat their respective no-skill calibration baselines. ML models outperformed the LACE-proxy clinical risk score (AUC 0.550) by approximately 0.10 AUC.

**Conclusion.** Three appropriately tuned machine learning models achieved modest but clinically useful discrimination once calibration was applied. Post-hoc isotonic calibration was essential: uncalibrated probabilities were not deployment-ready, but calibrated probabilities yielded positive net benefit across clinical thresholds, equal-baseline performance across racial subgroups, and meaningful advantage over a traditional clinical risk score. Calibration, not raw discrimination, was the binding constraint on clinical utility.

---

## Repository contents

| File | Description |
|---|---|
| `diabetes_readmission_improved.ipynb` | Main analysis notebook (full pipeline, end-to-end reproducible) |
| `HDS_ML_MAHDIOSMAN_2605.docx` | Report write-up (~3000 words) |
| `requirements.txt` | Python package dependencies |
| `diabetic_data.csv` | UCI Diabetes 130-Hospitals dataset (Strack et al., 2014) |
| `README.md` | This file |

---

## Dataset

**Source:** UCI Machine Learning Repository, ID 296 — Diabetes 130-US Hospitals for years 1999–2008
**Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0)
**Original citation:** Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J. and Clore, J.N. (2014) *Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records.* BioMed Research International, 2014, 781670.

**Size:** 101,766 inpatient encounters across 130 US hospitals (50 features)
**After preprocessing:** 69,973 unique patient encounters; 9.0% 30-day readmission prevalence

The dataset file (`diabetic_data.csv`) is included in this repository for reproducibility. It can also be downloaded directly from the [UCI ML Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).

---

## How to reproduce

### 1. Clone the repository
```bash
git clone https://github.com/mahdi2382/Module_5.git
cd Module_5
```

### 2. Set up a Python environment (Python 3.11+ required; tested on 3.11–3.14)
```bash
python -m venv venv
source venv/bin/activate          # Mac/Linux
# OR
venv\Scripts\activate             # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> Note: the PyPI package name is `scikit-learn` (with a hyphen), even though it is imported in Python as `sklearn`.

### 4. Open and run the notebook
```bash
jupyter notebook diabetes_readmission_improved.ipynb
```

In Jupyter: **Kernel → Restart Kernel and Run All Cells**. Full pipeline runs end-to-end in approximately 10–15 minutes on a standard laptop.

---

## Notebook structure

The notebook is organised into 15 numbered sections, executable top-to-bottom from a clean kernel. Figure numbers below match the figure numbering used in the report.

| # | Section | What it does | Report figures |
|---|---|---|---|
| 0 | Environment Setup | Imports and global configuration | — |
| 1 | Data Loading | Loads the raw 101,766-encounter CSV | — |
| 2 | Preprocessing | Deduplication, hospice exclusion, ICD-9 grouping, ordinal encoding | — |
| 3 | Exploratory Data Analysis | Class distribution, feature distributions, correlations | — |
| 4 | Train/Test Split | Stratified 80:20 split with `random_state=42` | — |
| 5 | Hyperparameter Search | 5-fold `RandomizedSearchCV` over LR, RF, XGBoost | — |
| 6 | Stratified 5-Fold CV | Cross-validation performance with Wilcoxon comparison | — |
| 7 | Final Evaluation on Held-Out Test Set | Test-set classification reports | — |
| 8 | Evaluation Visualisations | ROC + Precision-Recall curves; confusion matrices | Fig 1, Fig 2 |
| 9 | **Probability Calibration** | Isotonic regression; before/after Brier scores; calibration curves | Fig 3 |
| 10 | Decision Curve Analysis | Net-benefit analysis on uncalibrated vs calibrated probabilities | Fig 4 |
|  | *Per-subgroup calibration* | Subgroup Brier scores after isotonic calibration | — |
|  | *Threshold Optimisation* | F1-optimal and recall-prioritising thresholds from out-of-fold CV | — |
| 11 | Feature Importance | MDI for RF and XGB | Fig 5 |
| 12 | SHAP Interpretability | TreeSHAP beeswarm + mean \|SHAP\|; patient-level waterfalls | Fig 6a/6b, Fig 7 |
| 13 | **LACE-proxy Benchmark** | Head-to-head ROC vs LACE clinical risk score | Fig 8 |
| 14 | Fairness Audit | Subgroup AUC, precision, recall by race | Fig 9 |
| 15 | Summary Results Table | Numeric summary export to JSON for the report | — |

---

## Methodology summary

### Preprocessing
- Patient-level deduplication (earliest encounter retained) to prevent temporal leakage
- Removal of identifier columns and high-missingness features
- Exclusion of hospice / in-hospital death discharges (codes 11, 13, 14, 19, 20, 21)
- ICD-9 diagnosis codes grouped into nine clinical categories
- Diabetes codes captured through the 250.xx family (numeric range `250 ≤ code < 251`, i.e. all 250.x subcodes)
- Race preserved as a string variable for the fairness audit

### Imbalance correction
- **Single-correction weighting only.** `class_weight='balanced'` for Logistic Regression and Random Forest; `scale_pos_weight ≈ 10.15` for XGBoost.
- **SMOTE is deliberately not used** to avoid the double-correction artefact identified in prior analyses of this dataset (Blagus & Lusa, 2013).

### Encoding
- All nominal categorical variables (race, diagnoses, admission type, discharge disposition, admission source) are **one-hot encoded** inside a `scikit-learn` `ColumnTransformer`, fitted only within training folds (no test-set leakage).
- After one-hot encoding, the feature space contains 110 model features.

### Models
- L2-regularised Logistic Regression (LR) — transparent, monotonic effects
- Random Forest (RF) — non-linear, moderate complexity
- XGBoost (XGB) — gradient-boosted ensemble

Hyperparameters selected via 5-fold stratified `RandomizedSearchCV` optimising AUC-ROC.

### Evaluation
- Discrimination: AUC-ROC, AUPRC
- Calibration: Brier score (vs no-skill prevalence baseline of 0.082); reliability diagrams
- Clinical utility: Decision Curve Analysis (Vickers & Elkin, 2006)
- Statistical testing: Wilcoxon signed-rank on fold-level AUCs (interpreted with appropriate underpowered-test caveat)
- Interpretability: TreeSHAP via XGBoost's native algorithm (Lundberg & Lee, 2017)
- Fairness: subgroup-specific AUC, precision, recall, Brier; equal-opportunity and predictive-parity audits across racial subgroups

### Benchmark comparison
- LACE-proxy clinical risk score (van Walraven et al., 2010) computed from available dataset variables and compared head-to-head against the three ML models

---

## Key findings

- All three models achieve modest, statistically similar discrimination (AUC ≈ 0.65)
- Uncalibrated models systematically overpredict readmission probability; isotonic calibration brings all three Brier scores below the no-skill baseline
- Calibrated decision curve analysis shows positive net benefit at clinically plausible thresholds (10–25%)
- Discharge disposition, prior inpatient visits, age, length of stay, and diabetes medication burden are the dominant predictors
- The gender-dominant feature importance reported in prior analyses of this dataset is shown to be an encoding-and-impurity-bias artefact
- All racial subgroups achieve calibrated Brier scores below their respective no-skill baselines
- ML models outperform the LACE-proxy clinical risk score by approximately 0.10 AUC in this diabetic-specific cohort

---

## Reproducibility notes

- **Random seed:** `RANDOM_STATE = 42` is set globally for all stochastic operations
- **Runtime:** approximately 10–15 minutes end-to-end on a standard laptop
- **Memory:** the calibration step (5-fold `CalibratedClassifierCV` for three models) is the heaviest step; approximately 4 GB peak RAM
- **Outputs generated:** all figures saved as `.png` at 150 DPI; key results exported to `results_for_report.json`, `subgroup_calibration.csv`, `threshold_optimisation.csv`, `lace_comparison.csv`

---

## Acknowledgements

Dataset: Strack et al. (2014), via the UCI Machine Learning Repository
Methodology references throughout the report and notebook
