# EOLA Manuscript–Toolkit Artifact Mapping



## Purpose



This document maps the canonical computational artifacts produced by the

EOLA Research Toolkit v2.0.0 to the tables and figures used in the

associated manuscript.



The toolkit numbering reflects the canonical computational outputs.

The manuscript includes one additional literature-synthesis table and

one additional research-architecture figure that are editorial

manuscript assets and are therefore not part of the public computational

toolkit.



---



## Figures



| Toolkit artifact | Manuscript artifact | Description |

|---|---|---|

| `figures/final/Figure_1_robust_model_comparison.png` | Figure 2 | Robust model comparison across 30 stratified train/test seeds |

| `figures/final/Figure_2_pareto_frontier.png` | Figure 3 | Robust Pareto frontier for gender-blind Logistic Regression |

| `figures/final/Figure_3_SHAP_global_importance.png` | Figure 4 | Global mean absolute attribution importance |

| `figures/final/Figure_4_explanation_disparity.png` | Figure 5 | Feature-level explanation disparity across gender |

| `figures/final/Figure_5_GEDI_stability.png` | Figure 6 | Stability of GEDI and Weighted GEDI across 30 seeds |



### Manuscript-only Figure 1



Manuscript Figure 1, **EOLA Research Architecture**, is a manuscript

editorial/research-architecture asset maintained outside the public

computational repository.



It is not required to reproduce the numerical results of the toolkit.



---



## Tables



| Toolkit artifact | Manuscript artifact | Description |

|---|---|---|

| `reports/final/tables/Table_1_DIB_exact_reproduction.csv` | Table 2 | Exact reproduction of the published Data in Brief benchmark |

| `reports/final/tables/Table_2_robust_model_comparison.csv` | Table 3 | Robust 30-seed comparison of candidate models and gender-feature policies |

| `reports/final/tables/Table_3_robust_pareto_frontier.csv` | Table 4 | Robust Pareto frontier for the final EOLA experiment |

| `reports/final/tables/Table_4_tau_sensitivity.csv` | Table 5 | Sensitivity of research-reference selection to the fairness-tolerance parameter tau |

| `reports/final/tables/Table_5_explanation_disparity.csv` | Table 6 | Global attribution importance and feature-level explanation disparity |

| `reports/final/tables/Table_6_GEDI_summary.csv` | Table 7 | Stability summary for GEDI and Weighted GEDI |



### Manuscript-only Table 1



Manuscript Table 1, **Critical Comparison of Research Strands and the

EOLA Research Architecture**, is a literature-synthesis table and is

maintained outside the public computational toolkit.



---



## Publication Formatting



Manuscript tables may present a publication-formatted subset of the

columns available in the canonical CSV artifacts.



For example, Manuscript Table 5 is a compact representation of

`Table_4_tau_sensitivity.csv` and displays:



- `tau`

- `status`

- `n_feasible`

- selected `model` and `gender_mode`

- `threshold`

- `balanced_accuracy_mean`

- `equalized_odds_difference_p95`



Additional audit fields remain available in the canonical CSV and are

retained for reproducibility.



Formatting changes such as `0.5` → `0.50`,

`FEASIBLE` → `Feasible`, or

`LogisticRegression / gender_blind` → `LR / gender-blind`

do not alter the underlying result.



---



## Research Boundary



The threshold `t = 0.50` and fairness tolerance `tau = 0.05` are

study-specific research-reference values used for reproducibility.



They are not ECOTEC operational thresholds, intervention rules, or

institutional fairness standards.

