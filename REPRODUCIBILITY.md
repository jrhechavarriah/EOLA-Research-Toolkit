\# EOLA Research Toolkit — Reproducibility Guide



Release candidate: \*\*v2.0.0\*\*



\## 1. Purpose



This document describes the computational workflow required to reproduce the scientific results associated with the EOLA Research Toolkit.



The workflow covers:



\- canonical dataset acquisition and integrity verification;

\- dataset-contract validation;

\- exact reproduction of the published Data in Brief baseline results;

\- 30-seed robust predictive and subgroup-fairness evaluation;

\- threshold sensitivity and Pareto analysis;

\- explanation-level disparity analysis;

\- GEDI and Weighted GEDI estimation;

\- final scientific freeze generation.



The toolkit is a research and reproducibility package. It is not an institutional production decision system.



\---



\## 2. Canonical Data Source



EOLA uses the public:



\*\*ECOTEC\_Online Student Risk Dataset\*\*



Version:



`v1.0.4`



DOI:



`10.5281/zenodo.22015963`



Expected dataset dimensions:



\- 12,632 records;

\- 18 variables.



Expected SHA-256:



`8466b02d028be1fb11c39a320517d0d388580436b5278f949d60e05ee06899ac`



The raw dataset is intentionally not duplicated in the EOLA software release.



The acquisition module retrieves the canonical dataset and verifies its SHA-256 digest before downstream analysis.



\---



\## 3. Computational Environment



The validated release-candidate environment uses:



\- Python 3.12.14

\- NumPy 2.0.2

\- pandas 2.2.2

\- SciPy 1.16.3

\- scikit-learn 1.6.1

\- joblib 1.5.3

\- matplotlib 3.10.0

\- SHAP 0.52.0

\- PyArrow 25.0.1

\- openpyxl 3.1.5

\- PyYAML 6.0.3

\- pytest 9.1.1



Three environment artifacts are provided:



`requirements-lock.txt`



contains the complete Python package freeze from the validated environment.



`environment.yml`



provides the Conda environment specification.



`environment\_versions.txt`



records the principal scientific software versions used for the validated release candidate.



Because some dependencies were installed through pip, `requirements-lock.txt` is the more complete package-level record.



\---



\## 4. Project Configuration



The principal computational configuration is:



`config/eola\_config.yaml`



It records:



\- canonical dataset metadata;

\- dataset SHA-256;

\- expected schema;

\- target variable;

\- identifier;

\- sensitive attribute;

\- train/test split;

\- seed range;

\- threshold grid;

\- fairness tolerance values;

\- feature-selection policy.



The public configuration is portable and does not depend on a workstation-specific absolute path.



\---



\## 5. Feature Contract



The public dataset contains 18 variables.



The following variables are not candidate model predictors:



\- `PUBLIC\_RECORD\_ID`: public record identifier;

\- `TARGET\_RISK`: model target.



This leaves 16 candidate predictors.



A zero-variance audit identifies:



`FACULTY`



as invariant across all 12,632 records.



Therefore:



\- gender-aware EOLA configuration: 15 effective predictors;

\- gender-blind EOLA configuration: 14 effective predictors.



For the gender-blind configuration, `GENDER` is excluded from the predictive feature matrix but retained separately for subgroup fairness auditing.



The published Data in Brief benchmark reproduction deliberately retains the original 16-predictor specification because its objective is exact reproduction of the previously published benchmark.



\---



\## 6. Target Interpretation Constraint



`TARGET\_RISK` is a cross-sectional contemporaneous administrative proxy observed in the institutional dataset snapshot.



It is not a prospectively observed dropout, persistence, graduation, or retention outcome.



Consequently, this release evaluates retrospective predictive discrimination, robustness, subgroup fairness, and explanation-level behavior.



It does not establish prospective early-warning effectiveness.



\---



\## 7. Reproduction Sequence



Run all commands from the project root.



\### Step 1 — Acquire the canonical dataset



```bash

python -m src.data\_acquisition

