from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.eola_config import load_config


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    config = load_config()
    root = Path(config["_project_root"])

    # ---------- canonical inputs ----------

    dib_path = (
        root / "outputs" / "dib_reproduction"
        / "dib_reproduction_report.json"
    )

    b3_manifest_path = (
        root / "outputs" / "eola_multiseed"
        / "multiseed_manifest.json"
    )

    b3_summary_path = (
        root / "outputs" / "eola_multiseed"
        / "eola_multiseed_summary.csv"
    )

    b4_manifest_path = (
        root / "outputs" / "eola_threshold_pareto"
        / "threshold_pareto_manifest.json"
    )

    pareto_path = (
        root / "outputs" / "eola_threshold_pareto"
        / "pareto_frontier.csv"
    )

    tau_path = (
        root / "outputs" / "eola_threshold_pareto"
        / "tau_selection.csv"
    )

    reference_path = (
        root / "reports" / "phase_b4"
        / "EOLA_RESEARCH_REFERENCE_CONFIG_v1.0.json"
    )

    b5_manifest_path = (
        root / "outputs" / "eola_explainability"
        / "explainability_manifest.json"
    )

    shap_summary_path = (
        root / "outputs" / "eola_explainability"
        / "shap_feature_summary.csv"
    )

    gedi_seed_path = (
        root / "outputs" / "eola_explainability"
        / "gedi_by_seed.csv"
    )

    gedi_summary_path = (
        root / "outputs" / "eola_explainability"
        / "gedi_summary.json"
    )

    explain_reference_path = (
        root / "reports" / "phase_b5"
        / "EOLA_EXPLAINABILITY_REFERENCE_v1.0.json"
    )

    zero_variance_path = (
        root / "reports" / "phase_b3"
        / "ZERO_VARIANCE_AUDIT_v1.0.json"
    )

    required = [
        dib_path,
        b3_manifest_path,
        b3_summary_path,
        b4_manifest_path,
        pareto_path,
        tau_path,
        reference_path,
        b5_manifest_path,
        shap_summary_path,
        gedi_seed_path,
        gedi_summary_path,
        explain_reference_path,
        zero_variance_path,
    ]

    missing = [
        str(p.relative_to(root))
        for p in required
        if not p.exists()
    ]

    require(
        not missing,
        f"Missing canonical freeze inputs: {missing}",
    )

    # ---------- contracts ----------

    dib = json.loads(dib_path.read_text(encoding="utf-8"))
    b3 = json.loads(b3_manifest_path.read_text(encoding="utf-8"))
    b4 = json.loads(b4_manifest_path.read_text(encoding="utf-8"))
    b5 = json.loads(b5_manifest_path.read_text(encoding="utf-8"))

    reference = json.loads(
        reference_path.read_text(encoding="utf-8")
    )

    explain_reference = json.loads(
        explain_reference_path.read_text(encoding="utf-8")
    )

    zero_variance = json.loads(
        zero_variance_path.read_text(encoding="utf-8")
    )

    require(
        dib.get("dib_reproduction_pass") is True,
        "DIB reproduction contract not passed.",
    )

    require(
        b3.get("multiseed_contract_ok") is True,
        "B.3 contract not passed.",
    )

    require(
        b4.get("threshold_pareto_contract_ok") is True,
        "B.4 contract not passed.",
    )

    require(
        b5.get("explainability_contract_ok") is True,
        "B.5 contract not passed.",
    )

    require(
        zero_variance["zero_variance_predictors"]
        == ["FACULTY"],
        "Unexpected zero-variance feature audit.",
    )

    require(
        reference["model"] == "LogisticRegression",
        "Unexpected selected model.",
    )

    require(
        reference["gender_mode"] == "gender_blind",
        "Unexpected selected gender mode.",
    )

    require(
        int(reference["effective_predictor_count"]) == 14,
        "Expected 14 effective predictors.",
    )

    require(
        abs(float(reference["threshold"]) - 0.5) < 1e-12,
        "Unexpected reference threshold.",
    )

    # ---------- load tables ----------

    b3_summary = pd.read_csv(b3_summary_path)
    pareto = pd.read_csv(pareto_path)
    tau = pd.read_csv(tau_path)
    shap_summary = pd.read_csv(shap_summary_path)
    gedi_seed = pd.read_csv(gedi_seed_path)

    gedi_summary = json.loads(
        gedi_summary_path.read_text(encoding="utf-8")
    )

    # ---------- final directories ----------

    final = root / "reports" / "final"
    tables = final / "tables"
    figures = root / "figures" / "final"

    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    # ---------- Table 1: DIB reproduction ----------

    dib_rows = []

    for model_name, model_data in dib["models"].items():
        metrics = model_data["metrics"]

        dib_rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy":
                    metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "class1_f1": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
                "confusion_matrix":
                    str(metrics["confusion_matrix"]),
            }
        )

    table1 = pd.DataFrame(dib_rows)

    table1.to_csv(
        tables / "Table_1_DIB_exact_reproduction.csv",
        index=False,
    )

    # ---------- Table 2: robust model comparison ----------

    b3_cols = [
        "model",
        "gender_mode",
        "predictor_count",
        "balanced_accuracy_mean",
        "balanced_accuracy_std",
        "macro_f1_mean",
        "macro_f1_std",
        "roc_auc_mean",
        "roc_auc_std",
        "equalized_odds_difference_mean",
        "equalized_odds_difference_p95",
    ]

    table2 = b3_summary[b3_cols].copy()

    table2.to_csv(
        tables / "Table_2_robust_model_comparison.csv",
        index=False,
    )

    # ---------- Table 3: Pareto ----------

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

    pareto[pareto_cols].to_csv(
        tables / "Table_3_robust_pareto_frontier.csv",
        index=False,
    )

    # ---------- Table 4: tau ----------

    tau.to_csv(
        tables / "Table_4_tau_sensitivity.csv",
        index=False,
    )

    # ---------- Table 5: explanation ----------

    shap_cols = [
        "importance_rank",
        "feature",
        "global_mean_abs_shap_mean",
        "global_mean_abs_shap_std",
        "normalized_disparity_mean",
        "normalized_disparity_std",
        "abs_normalized_disparity_mean",
        "abs_normalized_disparity_p95",
        "positive_seed_fraction",
        "negative_seed_fraction",
    ]

    shap_summary[shap_cols].to_csv(
        tables / "Table_5_explanation_disparity.csv",
        index=False,
    )

    # ---------- Table 6: GEDI ----------

    gedi_rows = []

    for metric, stats in gedi_summary.items():
        gedi_rows.append(
            {
                "metric": metric,
                **stats,
            }
        )

    pd.DataFrame(gedi_rows).to_csv(
        tables / "Table_6_GEDI_summary.csv",
        index=False,
    )

    # ---------- Figure 1: robust model comparison ----------

    fig1 = b3_summary.copy()

    order = [
        ("LogisticRegression", "gender_aware"),
        ("LogisticRegression", "gender_blind"),
        ("RandomForestClassifier", "gender_aware"),
        ("RandomForestClassifier", "gender_blind"),
    ]

    ordered_rows = []

    for model_name, gender_mode in order:
        match = fig1[
            (fig1["model"] == model_name)
            & (fig1["gender_mode"] == gender_mode)
        ]
        if not match.empty:
            ordered_rows.append(match.iloc[0])

    fig1 = pd.DataFrame(ordered_rows).reset_index(drop=True)

    label_map = {
        ("LogisticRegression", "gender_aware"): "LR aware",
        ("LogisticRegression", "gender_blind"): "LR blind",
        ("RandomForestClassifier", "gender_aware"): "RF aware",
        ("RandomForestClassifier", "gender_blind"): "RF blind",
    }

    x_labels = [
        label_map[(row["model"], row["gender_mode"])]
        for _, row in fig1.iterrows()
    ]

    x = list(range(len(fig1)))
    width = 0.18

    ba_vals = fig1["balanced_accuracy_mean"].tolist()
    f1_vals = fig1["macro_f1_mean"].tolist()
    roc_vals = fig1["roc_auc_mean"].tolist()
    eodds_vals = fig1["equalized_odds_difference_p95"].tolist()

    pos1 = [i - 1.5 * width for i in x]
    pos2 = [i - 0.5 * width for i in x]
    pos3 = [i + 0.5 * width for i in x]
    pos4 = [i + 1.5 * width for i in x]

    plt.figure(figsize=(8, 6))

    plt.bar(pos1, ba_vals, width=width, label="Balanced Accuracy")
    plt.bar(pos2, f1_vals, width=width, label="Macro-F1")
    plt.bar(pos3, roc_vals, width=width, label="ROC-AUC")
    plt.bar(pos4, eodds_vals, width=width, label="EOdds P95")

    plt.xticks(x, x_labels)
    plt.ylabel("Metric value")
    plt.title("Robust model comparison across 30 seeds")
    plt.legend(fontsize=8)
    plt.tight_layout()

    plt.savefig(
        figures / "Figure_1_robust_model_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ---------- Figure 2: Pareto frontier ----------

    p = pareto.sort_values(
        "equalized_odds_difference_p95"
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        p["equalized_odds_difference_p95"],
        p["balanced_accuracy_mean"],
        marker="o",
    )

    for _, row in p.iterrows():
        plt.annotate(
            f"{row['threshold']:.3f}",
            (
                row["equalized_odds_difference_p95"],
                row["balanced_accuracy_mean"],
            ),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.xlabel("Equalized Odds Difference (P95)")
    plt.ylabel("Mean Balanced Accuracy")
    plt.title(
        "Robust Pareto frontier for EOLA threshold selection"
    )
    plt.tight_layout()
    plt.savefig(
        figures / "Figure_2_pareto_frontier.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ---------- Figure 3: SHAP importance ----------

    s = shap_summary.sort_values(
        "global_mean_abs_shap_mean",
        ascending=True,
    )

    plt.figure(figsize=(9, 7))

    plt.barh(
        s["feature"],
        s["global_mean_abs_shap_mean"],
    )

    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title(
        "Global feature importance across 30 seeds"
    )
    plt.tight_layout()
    plt.savefig(
        figures / "Figure_3_SHAP_global_importance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ---------- Figure 4: explanation disparity ----------

    s2 = shap_summary.sort_values(
        "normalized_disparity_mean"
    )

    plt.figure(figsize=(9, 7))

    plt.barh(
        s2["feature"],
        s2["normalized_disparity_mean"],
    )

    plt.axvline(
        0,
        linewidth=1,
    )

    plt.xlabel(
        "Normalized explanation disparity "
        "(Female − Male)"
    )
    plt.ylabel("Feature")
    plt.title(
        "Feature-level explanation disparity across gender"
    )
    plt.tight_layout()
    plt.savefig(
        figures / "Figure_4_explanation_disparity.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ---------- Figure 5: GEDI distributions ----------

    plt.figure(figsize=(7, 6))

    plt.boxplot(
        [
            gedi_seed["gedi"],
            gedi_seed["weighted_gedi"],
        ],
        tick_labels=[
            "GEDI",
            "Weighted GEDI",
        ],
    )

    plt.ylabel("Index value")
    plt.title(
        "GEDI and Weighted GEDI stability across 30 seeds"
    )
    plt.tight_layout()
    plt.savefig(
        figures / "Figure_5_GEDI_stability.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # ---------- canonical result object ----------

    selected_tau = tau[
        (tau["tau"] == 0.05)
        & (tau["status"] == "FEASIBLE")
    ]

    require(
        len(selected_tau) == 1,
        "Expected one feasible tau=0.05 selection.",
    )

    selected_tau = selected_tau.iloc[0]

    canonical = {
        "release_candidate":
            "EOLA Research Toolkit v2.0.0",

        "freeze_utc":
            datetime.now(timezone.utc).isoformat(),

        "dataset": {
            "version":
                config["dataset"]["version"],
            "doi":
                config["dataset"]["doi"],
            "sha256":
                config["dataset"]["expected_sha256"],
            "rows":
                config["dataset"]["expected_rows"],
            "columns":
                config["dataset"]["expected_columns"],
        },

        "feature_contract": {
            "candidate_predictors": 16,
            "zero_variance_predictors":
                ["FACULTY"],
            "gender_aware_predictors": 15,
            "gender_blind_effective_predictors": 14,
        },

        "research_reference_configuration": {
            "model":
                reference["model"],
            "gender_mode":
                reference["gender_mode"],
            "threshold":
                float(reference["threshold"]),
            "research_tau":
                float(reference["research_tau"]),
            "balanced_accuracy_mean":
                float(
                    selected_tau[
                        "balanced_accuracy_mean"
                    ]
                ),
            "macro_f1_mean":
                float(
                    selected_tau[
                        "macro_f1_mean"
                    ]
                ),
            "roc_auc_mean":
                float(
                    reference["roc_auc_mean"]
                ),
            "equalized_odds_p95":
                float(
                    selected_tau[
                        "equalized_odds_difference_p95"
                    ]
                ),
        },

        "explainability": {
            "gedi_mean":
                float(
                    gedi_summary["gedi"]["mean"]
                ),
            "gedi_p95":
                float(
                    gedi_summary["gedi"]["p95"]
                ),
            "weighted_gedi_mean":
                float(
                    gedi_summary[
                        "weighted_gedi"
                    ]["mean"]
                ),
            "weighted_gedi_p95":
                float(
                    gedi_summary[
                        "weighted_gedi"
                    ]["p95"]
                ),
            "max_shap_reconstruction_error":
                float(
                    b5[
                        "max_shap_reconstruction_error"
                    ]
                ),
        },

        "contracts": {
            "dib_reproduction": True,
            "multiseed": True,
            "threshold_pareto": True,
            "explainability": True,
        },

        "interpretation_constraints": [
            "TARGET_RISK is a cross-sectional contemporaneous "
            "administrative proxy, not a prospectively observed "
            "dropout outcome.",

            "The 0.500 threshold is a research reference "
            "configuration, not an operational institutional "
            "decision threshold.",

            "Explanation disparities represent subgroup-associated "
            "differences in model attributions and are not causal "
            "evidence of discrimination or proxy effects.",
        ],
    }

    canonical_path = (
        final
        / "EOLA_CANONICAL_RESULTS_v2.0.0.json"
    )

    canonical_path.write_text(
        json.dumps(
            canonical,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ---------- final freeze manifest ----------

    generated = (
        sorted(tables.glob("*"))
        + sorted(figures.glob("*"))
        + [canonical_path]
    )

    freeze_manifest = {
        "release_candidate":
            "EOLA Research Toolkit v2.0.0",

        "generated_files": [
            {
                "path":
                    p.relative_to(root).as_posix(),
                "sha256":
                    sha256(p),
            }
            for p in generated
        ],

        "final_scientific_freeze_ok": True,
    }

    freeze_manifest_path = (
        final
        / "FINAL_SCIENTIFIC_FREEZE_MANIFEST.json"
    )

    freeze_manifest_path.write_text(
        json.dumps(
            freeze_manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n========== EOLA FINAL SCIENTIFIC FREEZE ==========\n")

    print(
        "Reference model       :",
        reference["model"],
    )

    print(
        "Gender mode           :",
        reference["gender_mode"],
    )

    print(
        "Effective predictors  :",
        reference["effective_predictor_count"],
    )

    print(
        "Threshold             :",
        reference["threshold"],
    )

    print(
        "Research tau          :",
        reference["research_tau"],
    )

    print(
        "Balanced Accuracy     :",
        f"{selected_tau['balanced_accuracy_mean']:.6f}",
    )

    print(
        "Macro-F1              :",
        f"{selected_tau['macro_f1_mean']:.6f}",
    )

    print(
        "ROC-AUC               :",
        f"{reference['roc_auc_mean']:.6f}",
    )

    print(
        "EOdds P95             :",
        f"{selected_tau['equalized_odds_difference_p95']:.6f}",
    )

    print(
        "GEDI mean             :",
        f"{gedi_summary['gedi']['mean']:.6f}",
    )

    print(
        "Weighted GEDI mean    :",
        f"{gedi_summary['weighted_gedi']['mean']:.6f}",
    )

    print(
        "\nTables generated       :",
        len(list(tables.glob("*.csv"))),
    )

    print(
        "Figures generated      :",
        len(list(figures.glob("*.png"))),
    )

    print(
        "\nEOLA FINAL SCIENTIFIC FREEZE: PASS"
    )


if __name__ == "__main__":
    main()