# EOLA Research Toolkit



**Equity-Oriented Learning Analytics for Reproducible Fairness-Aware and Explainable Student-Risk Modeling**



**Current release:** v2.0.0

## Release Metadata

- **Software:** EOLA Research Toolkit
- **Version:** 2.0.0
- **Zenodo DOI:** https://doi.org/10.5281/zenodo.23218886
- **License:** BSD-3-Clause
- **License scope:** Public Scientific/Reproducibility Layer only
- **Release status:** Public scientific release




## Overview



The EOLA Research Toolkit is a reproducible computational framework for evaluating predictive performance, subgroup fairness, threshold sensitivity, and explanation-level disparities in learning analytics.



The toolkit was developed and validated using the public ECOTEC_Online Student Risk Dataset and provides the complete scientific workflow required to reproduce the analyses reported in the associated EOLA study.



EOLA is intended as a research and reproducibility framework. It is not a deployed institutional decision system and does not define operational intervention policies.



## Canonical Dataset



- Dataset: ECOTEC_Online Student Risk Dataset

- Version: v1.0.4

- Records: 12,632

- Variables: 18

- DOI: 10.5281/zenodo.22015963

- Canonical SHA-256:

  `8466b02d028be1fb11c39a320517d0d388580436b5278f949d60e05ee06899ac`



The raw dataset is not duplicated in this software release. It is retrieved by the acquisition script from its canonical public source and verified against the expected SHA-256 digest.



## Feature Policy



After excluding the public record identifier and target variable, 16 candidate predictors remain.



A zero-variance audit identified:



- `FACULTY`



as invariant across all 12,632 records. It is therefore excluded from EOLA modeling.



The resulting configurations contain:



- 15 effective predictors in the gender-aware configuration;

- 14 effective predictors in the gender-blind configuration.



In the gender-blind configuration, `GENDER` is excluded from model predictors but retained externally for subgroup fairness auditing.



## Experimental Design



The robust evaluation uses:



- 30 random seeds (`0–29`);

- stratified 70/30 train-test partitions;

- zero person/record overlap between train and test partitions;

- Logistic Regression and Random Forest baselines;

- gender-aware and gender-blind configurations;

- predictive performance metrics;

- subgroup fairness metrics;

- threshold sensitivity analysis;

- Pareto analysis;

- explanation-level disparity analysis.



## Canonical Research Reference Configuration



The final research-reference configuration selected under the predefined robust governance analysis is:



- Model: Logistic Regression

- Mode: gender-blind

- Effective predictors: 14

- Decision threshold: 0.50

- Research fairness tolerance (`tau`): 0.05



Robust performance across 30 seeds:



- Mean Balanced Accuracy: 0.951330

- Mean Macro-F1: 0.800475

- Mean ROC-AUC: 0.990792

- Equalized Odds Difference P95: 0.043401



The threshold of 0.50 is a research reference configuration and must not be interpreted as an institutional operational threshold.



## Explainability and GEDI



EOLA evaluates explanation-level subgroup differences using exact linear SHAP decomposition for the selected Logistic Regression model.



Attributions are calculated on the model decision-function/log-odds scale and aggregated from encoded variables to the original predictor level.



Across 30 seeds:



- Mean GEDI: 0.113597

- GEDI P95: 0.144513

- Mean Weighted GEDI: 0.070973

- Weighted GEDI P95: 0.091367



The maximum SHAP reconstruction error was approximately `1.1e-14`.



Explanation disparities represent subgroup-associated differences in model attribution patterns. They must not be interpreted as causal evidence of discrimination, proxy discrimination, or institutional mechanisms.



## Target Interpretation



`TARGET_RISK` is a cross-sectional contemporaneous administrative proxy recorded in the institutional snapshot.



It is **not** a prospectively observed dropout, persistence, or retention outcome.



Accordingly, the present study evaluates retrospective classification and governance properties rather than prospective early-warning effectiveness.



## Reproducibility Workflow



The principal workflow is:



```text

Canonical dataset acquisition

        |

        v

Data contract validation

        |

        v

Published DIB benchmark reproduction

        |

        v

30-seed robust model evaluation

        |

        v

Threshold / Pareto / tau analysis

        |

        v

SHAP / explanation disparity / GEDI

        |

        v

Final scientific freeze

