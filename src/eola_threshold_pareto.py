from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.eola_config import load_config
from src.eola_multiseed import (
    binary_group_rates,
    fairness_from_groups,
    make_model,
    predictive_metrics,
)


def aggregate_threshold_results(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (model, mode, threshold), grp in df.groupby(
        ["model", "gender_mode", "threshold"],
        sort=True,
    ):
        rows.append(
            {
                "model": model,
                "gender_mode": mode,
                "threshold": float(threshold),
                "predictor_count": int(grp["predictor_count"].iloc[0]),
                "n_seeds": int(len(grp)),

                "accuracy_mean": float(grp["accuracy"].mean()),

                "balanced_accuracy_mean": float(
                    grp["balanced_accuracy"].mean()
                ),
                "balanced_accuracy_std": float(
                    grp["balanced_accuracy"].std(ddof=1)
                ),

                "macro_f1_mean": float(grp["macro_f1"].mean()),
                "f1_class1_mean": float(grp["f1_class1"].mean()),

                "roc_auc_mean": float(grp["roc_auc"].mean()),
                "average_precision_mean": float(
                    grp["average_precision"].mean()
                ),
                "brier_score_mean": float(grp["brier_score"].mean()),

                "demographic_parity_difference_mean": float(
                    grp["demographic_parity_difference"].mean()
                ),
                "equal_opportunity_difference_mean": float(
                    grp["equal_opportunity_difference"].mean()
                ),
                "fpr_difference_mean": float(
                    grp["fpr_difference"].mean()
                ),

                "equalized_odds_difference_mean": float(
                    grp["equalized_odds_difference"].mean()
                ),
                "equalized_odds_difference_p95": float(
                    grp["equalized_odds_difference"].quantile(0.95)
                ),
                "equalized_odds_difference_max": float(
                    grp["equalized_odds_difference"].max()
                ),
            }
        )

    return pd.DataFrame(rows)


def pareto_frontier(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Robust Pareto frontier:
      maximize mean Balanced Accuracy
      minimize P95 Equalized Odds Difference.
    """

    ba = summary["balanced_accuracy_mean"].to_numpy()
    eo = summary["equalized_odds_difference_p95"].to_numpy()

    efficient = np.ones(len(summary), dtype=bool)

    for i in range(len(summary)):
        for j in range(len(summary)):
            if i == j:
                continue

            no_worse = (
                ba[j] >= ba[i]
                and eo[j] <= eo[i]
            )

            strictly_better = (
                ba[j] > ba[i]
                or eo[j] < eo[i]
            )

            if no_worse and strictly_better:
                efficient[i] = False
                break

    out = summary.loc[efficient].copy()

    return out.sort_values(
        [
            "equalized_odds_difference_p95",
            "balanced_accuracy_mean",
        ],
        ascending=[True, False],
    ).reset_index(drop=True)


def tau_selection(
    summary: pd.DataFrame,
    taus: list[float],
    max_ba_loss: float,
) -> pd.DataFrame:

    best_ba = float(
        summary["balanced_accuracy_mean"].max()
    )

    ba_floor = best_ba - max_ba_loss

    work = summary.copy()

    work["ba_loss_from_best"] = (
        best_ba
        - work["balanced_accuracy_mean"]
    )

    work["performance_eligible"] = (
        work["balanced_accuracy_mean"] >= ba_floor
    )

    rows = []

    for tau in taus:

        feasible = work[
            (work["performance_eligible"])
            & (
                work["equalized_odds_difference_p95"]
                <= float(tau)
            )
        ].copy()

        if feasible.empty:
            rows.append(
                {
                    "tau": float(tau),
                    "status": "NO_FEASIBLE_CONFIGURATION",
                    "n_feasible": 0,
                    "best_balanced_accuracy": best_ba,
                    "balanced_accuracy_floor": ba_floor,
                    "max_balanced_accuracy_loss": max_ba_loss,
                    "model": "",
                    "gender_mode": "",
                    "threshold": np.nan,
                    "balanced_accuracy_mean": np.nan,
                    "macro_f1_mean": np.nan,
                    "roc_auc_mean": np.nan,
                    "equalized_odds_difference_mean": np.nan,
                    "equalized_odds_difference_p95": np.nan,
                    "ba_loss_from_best": np.nan,
                }
            )
            continue

        feasible["distance_from_0_5"] = (
            feasible["threshold"] - 0.5
        ).abs()

        feasible = feasible.sort_values(
            [
                "balanced_accuracy_mean",
                "macro_f1_mean",
                "equalized_odds_difference_p95",
                "roc_auc_mean",
                "distance_from_0_5",
                "model",
                "gender_mode",
                "threshold",
            ],
            ascending=[
                False,
                False,
                True,
                False,
                True,
                True,
                True,
                True,
            ],
        )

        selected = feasible.iloc[0]

        rows.append(
            {
                "tau": float(tau),
                "status": "FEASIBLE",
                "n_feasible": int(len(feasible)),
                "best_balanced_accuracy": best_ba,
                "balanced_accuracy_floor": ba_floor,
                "max_balanced_accuracy_loss": max_ba_loss,
                "model": selected["model"],
                "gender_mode": selected["gender_mode"],
                "threshold": float(selected["threshold"]),
                "balanced_accuracy_mean": float(
                    selected["balanced_accuracy_mean"]
                ),
                "macro_f1_mean": float(
                    selected["macro_f1_mean"]
                ),
                "roc_auc_mean": float(
                    selected["roc_auc_mean"]
                ),
                "equalized_odds_difference_mean": float(
                    selected["equalized_odds_difference_mean"]
                ),
                "equalized_odds_difference_p95": float(
                    selected["equalized_odds_difference_p95"]
                ),
                "ba_loss_from_best": float(
                    selected["ba_loss_from_best"]
                ),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()

    root = Path(config["_project_root"])
    ds = config["dataset"]
    modeling = config["modeling"]
    governance = config["governance"]

    b3_manifest_path = (
        root
        / "outputs"
        / "eola_multiseed"
        / "multiseed_manifest.json"
    )

    if not b3_manifest_path.exists():
        raise RuntimeError(
            "B.3 multiseed manifest not found."
        )

    b3_manifest = json.loads(
        b3_manifest_path.read_text(encoding="utf-8")
    )

    if not b3_manifest.get(
        "multiseed_contract_ok",
        False,
    ):
        raise RuntimeError(
            "B.3 multiseed contract has not passed."
        )

    csv_path = (
        root
        / "data"
        / "raw"
        / ds["filename"]
    )

    df = pd.read_csv(csv_path)

    identifier = modeling["identifier"]
    target = modeling["target"]
    sensitive = modeling["primary_sensitive_attribute"]

    y = df[target].astype(int)
    ids = df[identifier].astype(str)
    sensitive_values = df[sensitive].astype(str)

    candidate_predictors = [
        c
        for c in df.columns
        if c not in [identifier, target]
    ]

    if len(candidate_predictors) != 16:
        raise RuntimeError(
            f"Expected 16 candidate predictors; "
            f"found {len(candidate_predictors)}."
        )

    zero_variance_predictors = [
        c
        for c in candidate_predictors
        if df[c].nunique(dropna=False) <= 1
    ]

    if zero_variance_predictors != ["FACULTY"]:
        raise RuntimeError(
            "Unexpected zero-variance predictor set: "
            f"{zero_variance_predictors}"
        )

    base_predictors = [
        c
        for c in candidate_predictors
        if c not in zero_variance_predictors
    ]

    if len(base_predictors) != 15:
        raise RuntimeError(
            f"Expected 15 non-constant predictors; "
            f"found {len(base_predictors)}."
        )

    seed_start, seed_end = modeling["multiseed_range"]

    seeds = list(
        range(
            int(seed_start),
            int(seed_end) + 1,
        )
    )

    thresholds = [
        float(x)
        for x in governance["threshold_grid"]
    ]

    taus = [
        float(x)
        for x in governance["tau_values"]
    ]

    max_ba_loss = float(
        governance["max_balanced_accuracy_loss"]
    )

    model_names = [
        "LogisticRegression",
        "RandomForestClassifier",
    ]

    modes = [
        "gender_aware",
        "gender_blind",
    ]

    indices = np.arange(len(df))

    rows = []

    total_fits = (
        len(seeds)
        * len(model_names)
        * len(modes)
    )

    fit_counter = 0

    for seed in seeds:

        train_idx, test_idx = train_test_split(
            indices,
            test_size=float(
                modeling["split_test_size"]
            ),
            random_state=int(seed),
            stratify=y,
        )

        train_ids = set(ids.iloc[train_idx])
        test_ids = set(ids.iloc[test_idx])

        overlap = len(
            train_ids.intersection(test_ids)
        )

        if overlap != 0:
            raise RuntimeError(
                f"Leakage detected at seed {seed}: "
                f"{overlap} overlapping IDs."
            )

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        A_test = sensitive_values.iloc[test_idx]

        for mode in modes:

            predictors = list(base_predictors)

            if mode == "gender_blind":
                predictors.remove(sensitive)

            expected_predictors = (
                15
                if mode == "gender_aware"
                else 14
            )

            if len(predictors) != expected_predictors:
                raise RuntimeError(
                    f"Unexpected predictor count: {mode}"
                )

            X = df[predictors]

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            for model_name in model_names:

                fit_counter += 1

                print(
                    f"[FIT {fit_counter:03d}/{total_fits}] "
                    f"seed={seed:02d} "
                    f"{model_name} "
                    f"{mode}"
                )

                model = make_model(
                    model_name,
                    X_train,
                    int(seed),
                )

                model.fit(
                    X_train,
                    y_train,
                )

                y_prob = model.predict_proba(
                    X_test
                )[:, 1]

                for threshold in thresholds:

                    y_pred = (
                        y_prob >= threshold
                    ).astype(int)

                    pmetrics = predictive_metrics(
                        y_test,
                        y_pred,
                        y_prob,
                    )

                    groups = binary_group_rates(
                        y_test,
                        y_pred,
                        A_test,
                    )

                    fmetrics = fairness_from_groups(
                        groups
                    )

                    rows.append(
                        {
                            "seed": int(seed),
                            "model": model_name,
                            "gender_mode": mode,
                            "predictor_count": len(predictors),
                            "threshold": float(threshold),
                            "train_rows": int(len(train_idx)),
                            "test_rows": int(len(test_idx)),
                            "id_overlap": int(overlap),
                            **pmetrics,
                            **fmetrics,
                        }
                    )

    threshold_metrics = pd.DataFrame(rows)

    summary = aggregate_threshold_results(
        threshold_metrics
    )

    frontier = pareto_frontier(summary)

    best_ba = float(
        summary["balanced_accuracy_mean"].max()
    )

    ba_floor = best_ba - max_ba_loss

    summary["ba_loss_from_best"] = (
        best_ba
        - summary["balanced_accuracy_mean"]
    )

    summary["performance_eligible"] = (
        summary["balanced_accuracy_mean"]
        >= ba_floor
    )

    tau_df = tau_selection(
        summary,
        taus,
        max_ba_loss,
    )

    out = (
        root
        / "outputs"
        / "eola_threshold_pareto"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    threshold_metrics.to_csv(
        out / "threshold_metrics.csv",
        index=False,
    )

    summary.to_csv(
        out / "threshold_summary.csv",
        index=False,
    )

    frontier.to_csv(
        out / "pareto_frontier.csv",
        index=False,
    )

    tau_df.to_csv(
        out / "tau_selection.csv",
        index=False,
    )

    expected_metric_rows = (
        len(seeds)
        * len(model_names)
        * len(modes)
        * len(thresholds)
    )

    expected_summary_rows = (
        len(model_names)
        * len(modes)
        * len(thresholds)
    )

    manifest = {
        "dataset_version": ds["version"],
        "dataset_doi": ds["doi"],
        "dataset_sha256": ds["expected_sha256"],

        "seeds": seeds,
        "n_seeds": len(seeds),

        "models": model_names,
        "gender_modes": modes,

        "threshold_grid": thresholds,
        "n_thresholds": len(thresholds),

        "tau_values": taus,

        "max_balanced_accuracy_loss": max_ba_loss,

        "best_balanced_accuracy_mean": best_ba,
        "balanced_accuracy_floor": ba_floor,

        "expected_threshold_metric_rows":
            expected_metric_rows,
        "observed_threshold_metric_rows":
            int(len(threshold_metrics)),

        "expected_summary_rows":
            expected_summary_rows,
        "observed_summary_rows":
            int(len(summary)),

        "pareto_rows":
            int(len(frontier)),

        "tau_rows":
            int(len(tau_df)),

        "max_id_overlap":
            int(threshold_metrics["id_overlap"].max()),
    }

    manifest["threshold_pareto_contract_ok"] = all(
        [
            len(threshold_metrics)
            == expected_metric_rows,

            len(summary)
            == expected_summary_rows,

            len(tau_df)
            == len(taus),

            threshold_metrics[
                "id_overlap"
            ].max()
            == 0,

            threshold_metrics.isna().sum().sum()
            == 0,

            summary.isna().sum().sum()
            == 0,
        ]
    )

    (
        out / "threshold_pareto_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\n================ PARETO FRONTIER ================\n"
    )

    pareto_cols = [
        "model",
        "gender_mode",
        "threshold",
        "balanced_accuracy_mean",
        "macro_f1_mean",
        "roc_auc_mean",
        "equalized_odds_difference_mean",
        "equalized_odds_difference_p95",
    ]

    print(
        frontier[
            pareto_cols
        ].to_string(index=False)
    )

    print(
        "\n================ TAU SELECTION ==================\n"
    )

    tau_cols = [
        "tau",
        "status",
        "n_feasible",
        "model",
        "gender_mode",
        "threshold",
        "balanced_accuracy_mean",
        "macro_f1_mean",
        "equalized_odds_difference_p95",
        "ba_loss_from_best",
    ]

    print(
        tau_df[
            tau_cols
        ].to_string(index=False)
    )

    print(
        "\n================================================="
    )

    print(
        f"THRESHOLD ROWS: "
        f"{len(threshold_metrics)} / "
        f"{expected_metric_rows}"
    )

    print(
        f"SUMMARY ROWS: "
        f"{len(summary)} / "
        f"{expected_summary_rows}"
    )

    print(
        f"PARETO ROWS: {len(frontier)}"
    )

    print(
        "MAX ID OVERLAP:",
        threshold_metrics["id_overlap"].max(),
    )

    if not manifest[
        "threshold_pareto_contract_ok"
    ]:
        raise SystemExit(
            "EOLA THRESHOLD-PARETO CONTRACT: FAILED"
        )

    print(
        "EOLA THRESHOLD-PARETO CONTRACT: PASS"
    )


if __name__ == "__main__":
    main()