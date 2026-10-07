from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.eola_config import load_config
from src.eola_multiseed import make_model


EPS = 1e-12


def to_dense(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def original_feature_slices(
    preprocessor,
    X_train: pd.DataFrame,
) -> dict[str, tuple[int, int]]:

    categorical = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric = X_train.select_dtypes(
        exclude=["object", "category"]
    ).columns.tolist()

    encoder = (
        preprocessor
        .named_transformers_["cat"]
        .named_steps["encoder"]
    )

    feature_slices = {}
    position = 0

    for feature, categories in zip(
        categorical,
        encoder.categories_,
    ):
        width = len(categories)
        feature_slices[feature] = (
            position,
            position + width,
        )
        position += width

    for feature in numeric:
        feature_slices[feature] = (
            position,
            position + 1,
        )
        position += 1

    return feature_slices


def summarize_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for feature, grp in feature_df.groupby(
        "feature",
        sort=False,
    ):
        nd = grp["normalized_disparity"].astype(float)
        importance = grp["global_mean_abs_shap"].astype(float)

        rows.append(
            {
                "feature": feature,
                "n_seeds": int(len(grp)),

                "global_mean_abs_shap_mean":
                    float(importance.mean()),

                "global_mean_abs_shap_std":
                    float(importance.std(ddof=1)),

                "female_mean_abs_shap_mean":
                    float(
                        grp["female_mean_abs_shap"].mean()
                    ),

                "male_mean_abs_shap_mean":
                    float(
                        grp["male_mean_abs_shap"].mean()
                    ),

                "normalized_disparity_mean":
                    float(nd.mean()),

                "normalized_disparity_std":
                    float(nd.std(ddof=1)),

                "normalized_disparity_median":
                    float(nd.median()),

                "abs_normalized_disparity_mean":
                    float(nd.abs().mean()),

                "abs_normalized_disparity_p95":
                    float(nd.abs().quantile(0.95)),

                "positive_seed_fraction":
                    float((nd > 0).mean()),

                "negative_seed_fraction":
                    float((nd < 0).mean()),
            }
        )

    out = pd.DataFrame(rows)

    out["importance_rank"] = (
        out["global_mean_abs_shap_mean"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    return out.sort_values(
        "global_mean_abs_shap_mean",
        ascending=False,
    ).reset_index(drop=True)


def summarize_gedi(gedi_df: pd.DataFrame) -> dict:
    result = {}

    for metric in [
        "gedi",
        "weighted_gedi",
    ]:
        x = gedi_df[metric].astype(float)

        result[metric] = {
            "mean": float(x.mean()),
            "std": float(x.std(ddof=1)),
            "median": float(x.median()),
            "p05": float(x.quantile(0.05)),
            "p95": float(x.quantile(0.95)),
            "min": float(x.min()),
            "max": float(x.max()),
        }

    return result


def main() -> None:
    config = load_config()

    root = Path(config["_project_root"])
    ds = config["dataset"]
    modeling = config["modeling"]

    reference_path = (
        root
        / "reports"
        / "phase_b4"
        / "EOLA_RESEARCH_REFERENCE_CONFIG_v1.0.json"
    )

    if not reference_path.exists():
        raise RuntimeError(
            "B.4 research reference configuration not found."
        )

    reference = json.loads(
        reference_path.read_text(
            encoding="utf-8"
        )
    )

    if reference["model"] != "LogisticRegression":
        raise RuntimeError(
            "B.5 requires the B.4-selected LogisticRegression."
        )

    if reference["gender_mode"] != "gender_blind":
        raise RuntimeError(
            "B.5 requires the B.4-selected gender_blind model."
        )

    if int(reference["effective_predictor_count"]) != 14:
         raise RuntimeError(
             "Expected 14 effective gender-blind predictors."
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

    labels = (
        df[sensitive]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    lower_map = {
        x.strip().lower(): x
        for x in labels
    }

    if (
        "female" not in lower_map
        or "male" not in lower_map
    ):
        raise RuntimeError(
            "Expected GENDER labels Female and Male; "
            f"observed: {labels}"
        )

    female_label = lower_map["female"]
    male_label = lower_map["male"]

    candidate_predictors = [
        c
        for c in df.columns
        if c not in [
            identifier,
            target,
            sensitive,
        ]
    ]

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

    predictors = [
        c
        for c in candidate_predictors
        if c not in zero_variance_predictors
    ]

    if len(predictors) != 14:
        raise RuntimeError(
            f"Expected 14 effective predictors; found {len(predictors)}."
        )

    X = df[predictors]
    y = df[target].astype(int)

    ids = df[identifier].astype(str)
    A = df[sensitive].astype(str)

    indices = np.arange(len(df))

    seed_start, seed_end = modeling["multiseed_range"]

    seeds = list(
        range(
            int(seed_start),
            int(seed_end) + 1,
        )
    )

    feature_rows = []
    gedi_rows = []

    max_reconstruction_error = 0.0
    max_id_overlap = 0

    for run_number, seed in enumerate(
        seeds,
        start=1,
    ):

        print(
            f"[{run_number:02d}/{len(seeds)}] "
            f"seed={seed:02d} "
            "LogisticRegression gender_blind"
        )

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

        max_id_overlap = max(
            max_id_overlap,
            overlap,
        )

        if overlap != 0:
            raise RuntimeError(
                f"Leakage detected at seed {seed}."
            )

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]

        A_test = A.iloc[test_idx].reset_index(drop=True)

        model = make_model(
            "LogisticRegression",
            X_train,
            int(seed),
        )

        model.fit(
            X_train,
            y_train,
        )

        preprocessor = model.named_steps["preprocess"]
        classifier = model.named_steps["classifier"]

        X_train_t = to_dense(
            preprocessor.transform(X_train)
        )

        X_test_t = to_dense(
            preprocessor.transform(X_test)
        )

        beta = np.asarray(
            classifier.coef_[0],
            dtype=float,
        )

        intercept = float(
            classifier.intercept_[0]
        )

        if X_train_t.shape[1] != len(beta):
            raise RuntimeError(
                "Transformed feature count does not match "
                "classifier coefficients."
            )

        background_mean = np.asarray(
            X_train_t.mean(axis=0),
            dtype=float,
        )

        # Exact interventional SHAP for a linear model:
        # phi_j = beta_j * (x_j - E[X_j])
        phi_encoded = (
            X_test_t - background_mean
        ) * beta

        base_value = (
            intercept
            + float(
                np.dot(
                    background_mean,
                    beta,
                )
            )
        )

        decision = classifier.decision_function(
            X_test_t
        )

        reconstructed = (
            base_value
            + phi_encoded.sum(axis=1)
        )

        reconstruction_error = float(
            np.max(
                np.abs(
                    decision
                    - reconstructed
                )
            )
        )

        max_reconstruction_error = max(
            max_reconstruction_error,
            reconstruction_error,
        )

        feature_slices = original_feature_slices(
            preprocessor,
            X_train,
        )

        if set(feature_slices) != set(predictors):
            raise RuntimeError(
                "Original-feature mapping failed."
            )

        mapped_width = max(
            end
            for _, end
            in feature_slices.values()
        )

        if mapped_width != phi_encoded.shape[1]:
            raise RuntimeError(
                "Encoded SHAP columns are not fully mapped."
            )

        phi_original = np.column_stack(
            [
                phi_encoded[
                    :,
                    feature_slices[feature][0]:
                    feature_slices[feature][1],
                ].sum(axis=1)
                for feature in predictors
            ]
        )

        female_mask = (
            A_test.to_numpy()
            == female_label
        )

        male_mask = (
            A_test.to_numpy()
            == male_label
        )

        if (
            female_mask.sum() == 0
            or male_mask.sum() == 0
        ):
            raise RuntimeError(
                f"Missing gender subgroup at seed {seed}."
            )

        global_mean_abs = np.mean(
            np.abs(phi_original),
            axis=0,
        )

        global_mean_signed = np.mean(
            phi_original,
            axis=0,
        )

        female_mean_signed = np.mean(
            phi_original[female_mask],
            axis=0,
        )

        male_mean_signed = np.mean(
            phi_original[male_mask],
            axis=0,
        )

        female_mean_abs = np.mean(
            np.abs(
                phi_original[female_mask]
            ),
            axis=0,
        )

        male_mean_abs = np.mean(
            np.abs(
                phi_original[male_mask]
            ),
            axis=0,
        )

        raw_disparity = (
            female_mean_signed
            - male_mean_signed
        )

        normalized_disparity = np.divide(
            raw_disparity,
            np.maximum(
                global_mean_abs,
                EPS,
            ),
        )

        abs_normalized = np.abs(
            normalized_disparity
        )

        importance_sum = float(
            global_mean_abs.sum()
        )

        if importance_sum <= EPS:
            raise RuntimeError(
                "Global SHAP importance collapsed to zero."
            )

        weights = (
            global_mean_abs
            / importance_sum
        )

        gedi = float(
            abs_normalized.mean()
        )

        weighted_gedi = float(
            np.sum(
                weights
                * abs_normalized
            )
        )

        gedi_rows.append(
            {
                "seed": int(seed),
                "female_label": female_label,
                "male_label": male_label,
                "female_n": int(
                    female_mask.sum()
                ),
                "male_n": int(
                    male_mask.sum()
                ),
                "gedi": gedi,
                "weighted_gedi": weighted_gedi,
                "reconstruction_error":
                    reconstruction_error,
                "id_overlap": int(overlap),
            }
        )

        for j, feature in enumerate(
            predictors
        ):
            feature_type = (
                "categorical"
                if X_train[feature].dtype
                == "object"
                else "numeric"
            )

            feature_rows.append(
                {
                    "seed": int(seed),
                    "feature": feature,
                    "feature_type": feature_type,

                    "global_mean_shap":
                        float(
                            global_mean_signed[j]
                        ),

                    "global_mean_abs_shap":
                        float(
                            global_mean_abs[j]
                        ),

                    "female_mean_shap":
                        float(
                            female_mean_signed[j]
                        ),

                    "male_mean_shap":
                        float(
                            male_mean_signed[j]
                        ),

                    "female_mean_abs_shap":
                        float(
                            female_mean_abs[j]
                        ),

                    "male_mean_abs_shap":
                        float(
                            male_mean_abs[j]
                        ),

                    "raw_disparity_female_minus_male":
                        float(
                            raw_disparity[j]
                        ),

                    "normalized_disparity":
                        float(
                            normalized_disparity[j]
                        ),

                    "abs_normalized_disparity":
                        float(
                            abs_normalized[j]
                        ),

                    "global_importance_weight":
                        float(
                            weights[j]
                        ),
                }
            )

    feature_df = pd.DataFrame(
        feature_rows
    )

    gedi_df = pd.DataFrame(
        gedi_rows
    )

    feature_summary = summarize_features(
        feature_df
    )

    gedi_summary = summarize_gedi(
        gedi_df
    )

    out = (
        root
        / "outputs"
        / "eola_explainability"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_df.to_csv(
        out / "shap_feature_by_seed.csv",
        index=False,
    )

    feature_summary.to_csv(
        out / "shap_feature_summary.csv",
        index=False,
    )

    gedi_df.to_csv(
        out / "gedi_by_seed.csv",
        index=False,
    )

    (
        out / "gedi_summary.json"
    ).write_text(
        json.dumps(
            gedi_summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    expected_feature_rows = (
        len(seeds)
        * len(predictors)
    )

    manifest = {
        "dataset_version": ds["version"],
        "dataset_doi": ds["doi"],
        "dataset_sha256":
            ds["expected_sha256"],

        "reference_model":
            reference["model"],

        "gender_mode":
            reference["gender_mode"],

        "candidate_predictors_before_zero_variance_filter":
            15,

        "zero_variance_predictors":
            zero_variance_predictors,

        "effective_predictor_count":
            len(predictors),

        "decision_threshold":
            reference["threshold"],

        "threshold_note":
            "SHAP values explain the continuous model "
            "decision function and are independent of "
            "the classification threshold.",

        "shap_method":
            "closed-form interventional linear SHAP",

        "shap_scale":
            "log-odds / decision-function scale",

        "background":
            "mean transformed training vector for each seed",

        "aggregation":
            "encoded SHAP contributions summed to "
            "original-variable level",

        "normalized_disparity_definition":
            "(mean_SHAP_Female - mean_SHAP_Male) / "
            "global_mean_absolute_SHAP",

        "gedi_definition":
            "mean absolute normalized explanation disparity "
            "across original predictors",

        "weighted_gedi_definition":
            "importance-weighted mean absolute normalized "
            "explanation disparity",

        "n_seeds":
            len(seeds),

        "expected_feature_rows":
            expected_feature_rows,

        "observed_feature_rows":
            int(len(feature_df)),

        "expected_gedi_rows":
            len(seeds),

        "observed_gedi_rows":
            int(len(gedi_df)),

        "max_id_overlap":
            int(max_id_overlap),

        "max_shap_reconstruction_error":
            float(
                max_reconstruction_error
            ),

        "interpretation_constraint":
            "Subgroup-associated differences in model "
            "attribution patterns; not causal evidence of "
            "discrimination or proxy effects.",
    }

    finite_feature_values = np.isfinite(
        feature_df.select_dtypes(
            include=[np.number]
        ).to_numpy()
    ).all()

    finite_gedi_values = np.isfinite(
        gedi_df.select_dtypes(
            include=[np.number]
        ).to_numpy()
    ).all()

    manifest["explainability_contract_ok"] = all(
        [
            len(feature_df)
            == expected_feature_rows,

            len(gedi_df)
            == len(seeds),

            feature_df[
                "feature"
            ].nunique()
            == 14,

            max_id_overlap
            == 0,

            finite_feature_values,

            finite_gedi_values,

            max_reconstruction_error
            < 1e-10,
        ]
    )

    (
        out / "explainability_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\n========== TOP GLOBAL SHAP FEATURES ==========\n"
    )

    display_cols = [
        "importance_rank",
        "feature",
        "global_mean_abs_shap_mean",
        "normalized_disparity_mean",
        "abs_normalized_disparity_mean",
        "abs_normalized_disparity_p95",
        "positive_seed_fraction",
        "negative_seed_fraction",
    ]

    print(
        feature_summary[
            display_cols
        ]
        .head(15)
        .to_string(index=False)
    )

    print(
        "\n================ GEDI ========================\n"
    )

    print(
        json.dumps(
            gedi_summary,
            indent=2,
        )
    )

    print(
        "\n=============================================="
    )

    print(
        f"FEATURE ROWS: "
        f"{len(feature_df)} / "
        f"{expected_feature_rows}"
    )

    print(
        f"GEDI ROWS: "
        f"{len(gedi_df)} / "
        f"{len(seeds)}"
    )

    print(
        "MAX ID OVERLAP:",
        max_id_overlap,
    )

    print(
        "MAX SHAP RECONSTRUCTION ERROR:",
        f"{max_reconstruction_error:.3e}",
    )

    if not manifest[
        "explainability_contract_ok"
    ]:
        raise SystemExit(
            "EOLA EXPLAINABILITY CONTRACT: FAILED"
        )

    print(
        "EOLA EXPLAINABILITY CONTRACT: PASS"
    )


if __name__ == "__main__":
    main()