from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.eola_config import load_config


EXPECTED = {
    "LogisticRegression": {
        "accuracy": 0.940633,
        "f1_score": 0.967677,
        "roc_auc": 0.988559,
        "balanced_accuracy": 0.945843,
        "macro_f1": 0.802094,
        "confusion_matrix": [[197, 10], [215, 3368]],
    },
    "RandomForestClassifier": {
        "accuracy": 0.979683,
        "f1_score": 0.989271,
        "roc_auc": 0.984338,
        "balanced_accuracy": 0.889115,
        "macro_f1": 0.899102,
        "confusion_matrix": [[163, 44], [33, 3550]],
    },
}

TOLERANCE = 1e-6


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    precision, recall, class_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),

        "class_0_precision": float(precision[0]),
        "class_0_recall": float(recall[0]),
        "class_0_f1": float(class_f1[0]),
        "class_0_support": int(support[0]),

        "class_1_precision": float(precision[1]),
        "class_1_recall": float(recall[1]),
        "class_1_f1": float(class_f1[1]),
        "class_1_support": int(support[1]),

        "confusion_matrix": cm.tolist(),
    }


def compare_with_published(model_name: str, metrics: dict) -> dict:
    expected = EXPECTED[model_name]

    result = {}

    for key in [
        "accuracy",
        "f1_score",
        "roc_auc",
        "balanced_accuracy",
        "macro_f1",
    ]:
        observed = metrics[key]
        reference = expected[key]

        result[key] = {
            "observed": observed,
            "published": reference,
            "absolute_difference": abs(observed - reference),
            "pass": abs(observed - reference) <= TOLERANCE,
        }

    result["confusion_matrix"] = {
        "observed": metrics["confusion_matrix"],
        "published": expected["confusion_matrix"],
        "pass": metrics["confusion_matrix"] == expected["confusion_matrix"],
    }

    result["all_pass"] = all(
        item["pass"]
        for item in result.values()
        if isinstance(item, dict) and "pass" in item
    )

    return result


def main() -> None:
    config = load_config()
    root = Path(config["_project_root"])
    ds = config["dataset"]
    modeling = config["modeling"]

    csv_path = root / "data" / "raw" / ds["filename"]

    df = pd.read_csv(csv_path)

    # Match the published DIB benchmark cleaning.
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    df = df.drop_duplicates()
    df = df.dropna(subset=[modeling["target"]])
    df[modeling["target"]] = df[modeling["target"]].astype(int)

    X = df.drop(
        columns=[
            modeling["identifier"],
            modeling["target"],
        ]
    )
    y = df[modeling["target"]]

    if X.shape[1] != 16:
        raise RuntimeError(
            f"Expected 16 candidate predictors, found {X.shape[1]}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=modeling["split_test_size"],
        random_state=modeling["baseline_seed"],
        stratify=y,
    )

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_features = X.select_dtypes(
        exclude=["object", "category"]
    ).columns.tolist()

    lr_preprocessor = ColumnTransformer(
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
                categorical_features,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="median"),
                        ),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
        ]
    )

    rf_preprocessor = ColumnTransformer(
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
                categorical_features,
            ),
            (
                "num",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(strategy="median"),
                        ),
                    ]
                ),
                numeric_features,
            ),
        ]
    )

    models = {
        "LogisticRegression": Pipeline(
            steps=[
                ("preprocess", lr_preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=modeling["baseline_seed"],
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "RandomForestClassifier": Pipeline(
            steps=[
                ("preprocess", rf_preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        random_state=modeling["baseline_seed"],
                        class_weight="balanced",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    output_root = root / "outputs" / "dib_reproduction"
    output_root.mkdir(parents=True, exist_ok=True)

    full_report = {
        "dataset_version": ds["version"],
        "dataset_doi": ds["doi"],
        "n_rows": int(df.shape[0]),
        "n_candidate_predictors": int(X.shape[1]),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "seed": int(modeling["baseline_seed"]),
        "test_size": float(modeling["split_test_size"]),
        "models": {},
    }

    overall_pass = True

    for model_name, model in models.items():
        print(f"\nRunning {model_name}...")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(
            y_test,
            y_pred,
            y_prob,
        )

        comparison = compare_with_published(
            model_name,
            metrics,
        )

        model_dir = output_root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        metrics_row = {
            k: v
            for k, v in metrics.items()
            if k != "confusion_matrix"
        }

        pd.DataFrame([metrics_row]).to_csv(
            model_dir / "metrics.csv",
            index=False,
        )

        pd.DataFrame(
            metrics["confusion_matrix"],
            index=["actual_0", "actual_1"],
            columns=["predicted_0", "predicted_1"],
        ).to_csv(
            model_dir / "confusion_matrix.csv"
        )

        full_report["models"][model_name] = {
            "metrics": metrics,
            "comparison_to_published": comparison,
        }

        if not comparison["all_pass"]:
            overall_pass = False

        print(json.dumps(
            {
                "metrics": metrics,
                "comparison": comparison,
            },
            indent=2,
        ))

    full_report["dib_reproduction_pass"] = overall_pass

    report_path = output_root / "dib_reproduction_report.json"

    report_path.write_text(
        json.dumps(
            full_report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n===================================")
    print(f"TRAIN: {len(y_train)}")
    print(f"TEST : {len(y_test)}")
    print(f"PREDICTORS: {X.shape[1]}")

    if not overall_pass:
        raise SystemExit(
            "DIB BENCHMARK REPRODUCTION: FAILED"
        )

    print("DIB BENCHMARK REPRODUCTION: PASS")


if __name__ == "__main__":
    main()