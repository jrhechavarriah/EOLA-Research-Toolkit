from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.eola_config import load_config


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    categorical = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric = X.select_dtypes(
        exclude=["object", "category"]
    ).columns.tolist()

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_numeric:
        numeric_steps.append(
            ("scaler", StandardScaler())
        )

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical,
            ),
            (
                "num",
                Pipeline(steps=numeric_steps),
                numeric,
            ),
        ]
    )


def make_model(
    model_name: str,
    X_train: pd.DataFrame,
    seed: int,
) -> Pipeline:

    if model_name == "LogisticRegression":
        return Pipeline(
            steps=[
                (
                    "preprocess",
                    make_preprocessor(
                        X_train,
                        scale_numeric=True,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )

    if model_name == "RandomForestClassifier":
        return Pipeline(
            steps=[
                (
                    "preprocess",
                    make_preprocessor(
                        X_train,
                        scale_numeric=False,
                    ),
                ),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    raise ValueError(f"Unknown model: {model_name}")


def binary_group_rates(
    y_true: pd.Series,
    y_pred: np.ndarray,
    group: pd.Series,
) -> list[dict]:

    rows = []

    group_values = sorted(
        group.dropna().astype(str).unique().tolist()
    )

    if len(group_values) != 2:
        raise RuntimeError(
            "Primary fairness audit currently requires exactly "
            f"two GENDER groups; observed: {group_values}"
        )

    for value in group_values:
        mask = group.astype(str).to_numpy() == value

        yt = np.asarray(y_true)[mask]
        yp = np.asarray(y_pred)[mask]

        tn, fp, fn, tp = confusion_matrix(
            yt,
            yp,
            labels=[0, 1],
        ).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        positive_rate = float(np.mean(yp == 1))

        rows.append(
            {
                "group": value,
                "n": int(mask.sum()),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "tpr": float(tpr),
                "fpr": float(fpr),
                "positive_prediction_rate": positive_rate,
            }
        )

    return rows


def fairness_from_groups(group_rows: list[dict]) -> dict:
    if len(group_rows) != 2:
        raise RuntimeError("Expected exactly two subgroup rows.")

    g0, g1 = group_rows

    eopp = abs(g0["tpr"] - g1["tpr"])
    fpr_diff = abs(g0["fpr"] - g1["fpr"])
    dp = abs(
        g0["positive_prediction_rate"]
        - g1["positive_prediction_rate"]
    )

    eodds = max(eopp, fpr_diff)

    return {
        "demographic_parity_difference": float(dp),
        "equal_opportunity_difference": float(eopp),
        "fpr_difference": float(fpr_diff),
        "equalized_odds_difference": float(eodds),
    }


def predictive_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict:

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "f1_class1": float(
            f1_score(y_true, y_pred)
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_true, y_prob)
        ),
        "average_precision": float(
            average_precision_score(y_true, y_prob)
        ),
        "brier_score": float(
            brier_score_loss(y_true, y_prob)
        ),
    }


def summarize(metrics_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "f1_class1",
        "macro_f1",
        "roc_auc",
        "average_precision",
        "brier_score",
        "demographic_parity_difference",
        "equal_opportunity_difference",
        "fpr_difference",
        "equalized_odds_difference",
    ]

    rows = []

    for (model, mode), grp in metrics_df.groupby(
        ["model", "gender_mode"],
        sort=True,
    ):
        row = {
            "model": model,
            "gender_mode": mode,
            "n_seeds": int(len(grp)),
            "predictor_count": int(
                grp["predictor_count"].iloc[0]
            ),
        }

        for metric in metric_columns:
            values = grp[metric].astype(float)

            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_p05"] = float(values.quantile(0.05))
            row[f"{metric}_p95"] = float(values.quantile(0.95))

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()
    root = Path(config["_project_root"])
    ds = config["dataset"]
    modeling = config["modeling"]

    dib_report = (
        root
        / "outputs"
        / "dib_reproduction"
        / "dib_reproduction_report.json"
    )

    if not dib_report.exists():
        raise RuntimeError(
            "DIB benchmark reproduction report not found."
        )

    dib = json.loads(
        dib_report.read_text(encoding="utf-8")
    )

    if not dib.get("dib_reproduction_pass", False):
        raise RuntimeError(
            "DIB benchmark reproduction has not passed."
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

    base_predictors = [
        c
        for c in candidate_predictors
        if c not in zero_variance_predictors
    ]

    if zero_variance_predictors != ["FACULTY"]:
        raise RuntimeError(
            "Unexpected zero-variance predictor set: "
            f"{zero_variance_predictors}"
        )

    if len(base_predictors) != 15:
        raise RuntimeError(
            f"Expected 15 non-constant predictors; "
            f"found {len(base_predictors)}."
        )

    seed_start, seed_end = modeling["multiseed_range"]
    seeds = list(
        range(int(seed_start), int(seed_end) + 1)
    )

    threshold = float(
        modeling.get("default_threshold", 0.5)
    )

    metrics_rows = []
    subgroup_rows = []

    model_names = [
        "LogisticRegression",
        "RandomForestClassifier",
    ]

    modes = [
        "gender_aware",
        "gender_blind",
    ]

    indices = np.arange(len(df))

    total_runs = (
        len(seeds)
        * len(model_names)
        * len(modes)
    )

    run_counter = 0

    for seed in seeds:

        train_idx, test_idx = train_test_split(
            indices,
            test_size=float(modeling["split_test_size"]),
            random_state=int(seed),
            stratify=y,
        )

        train_ids = set(ids.iloc[train_idx])
        test_ids = set(ids.iloc[test_idx])

        overlap = len(train_ids.intersection(test_ids))

        if overlap != 0:
            raise RuntimeError(
                f"Leakage detected at seed {seed}: "
                f"{overlap} IDs overlap."
            )

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        A_test = sensitive_values.iloc[test_idx]

        for mode in modes:

            predictors = list(base_predictors)

            if mode == "gender_blind":
                predictors.remove(sensitive)

            expected_n = (
                15
                if mode == "gender_aware"
                else 14
            )

            if len(predictors) != expected_n:
                raise RuntimeError(
                    f"Unexpected predictor count for {mode}."
                )

            X = df[predictors]

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            for model_name in model_names:

                run_counter += 1

                print(
                    f"[{run_counter:03d}/{total_runs}] "
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

                metrics_rows.append(
                    {
                        "seed": int(seed),
                        "model": model_name,
                        "gender_mode": mode,
                        "predictor_count": len(predictors),
                        "threshold": threshold,
                        "train_rows": int(len(train_idx)),
                        "test_rows": int(len(test_idx)),
                        "id_overlap": int(overlap),
                        **pmetrics,
                        **fmetrics,
                    }
                )

                for group_row in groups:
                    subgroup_rows.append(
                        {
                            "seed": int(seed),
                            "model": model_name,
                            "gender_mode": mode,
                            "threshold": threshold,
                            **group_row,
                        }
                    )

    metrics_df = pd.DataFrame(metrics_rows)
    subgroup_df = pd.DataFrame(subgroup_rows)

    summary_df = summarize(metrics_df)

    out = (
        root
        / "outputs"
        / "eola_multiseed"
    )
    out.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(
        out / "eola_multiseed_metrics.csv",
        index=False,
    )

    subgroup_df.to_csv(
        out / "eola_multiseed_subgroup_metrics.csv",
        index=False,
    )

    summary_df.to_csv(
        out / "eola_multiseed_summary.csv",
        index=False,
    )

    manifest = {
        "dataset_version": ds["version"],
        "dataset_doi": ds["doi"],
        "dataset_sha256": ds["expected_sha256"],
        "n_seeds": len(seeds),
        "seeds": seeds,
        "test_size": float(modeling["split_test_size"]),
        "threshold": threshold,
        "models": model_names,
        "modes": modes,
        "candidate_predictors_before_zero_variance_filter": 16,
        "zero_variance_predictors": zero_variance_predictors,
        "gender_aware_predictors": 15,
        "gender_blind_predictors": 14,
        "expected_runs": total_runs,
        "observed_runs": int(len(metrics_df)),
        "expected_subgroup_rows": total_runs * 2,
        "observed_subgroup_rows": int(len(subgroup_df)),
        "max_id_overlap": int(
            metrics_df["id_overlap"].max()
        ),
    }

    manifest["multiseed_contract_ok"] = all(
        [
            len(metrics_df) == 120,
            len(subgroup_df) == 240,
            metrics_df["id_overlap"].max() == 0,
            set(metrics_df["predictor_count"]) == {14, 15},
            metrics_df.isna().sum().sum() == 0,
            subgroup_df.isna().sum().sum() == 0,
        ]
    )

    (
        out / "multiseed_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n================ SUMMARY ================\n")

    cols = [
        "model",
        "gender_mode",
        "predictor_count",
        "balanced_accuracy_mean",
        "macro_f1_mean",
        "roc_auc_mean",
        "equalized_odds_difference_mean",
        "equalized_odds_difference_p95",
    ]

    print(
        summary_df[cols].to_string(
            index=False
        )
    )

    print("\n=========================================")
    print(f"RUNS: {len(metrics_df)} / 120")
    print(f"SUBGROUP ROWS: {len(subgroup_df)} / 240")
    print(
        "MAX ID OVERLAP:",
        metrics_df["id_overlap"].max(),
    )

    if not manifest["multiseed_contract_ok"]:
        raise SystemExit(
            "EOLA MULTISEED CONTRACT: FAILED"
        )

    print("EOLA MULTISEED CONTRACT: PASS")


if __name__ == "__main__":
    main()