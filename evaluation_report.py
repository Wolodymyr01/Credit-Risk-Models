import html
import os
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPORT_COLORS = {
    "ink": "#1f2933",
    "muted": "#596574",
    "line": "#d8dee8",
    "blue": "#2f6fbb",
    "teal": "#008b84",
    "amber": "#c27c1a",
    "red": "#b84a52",
    "green": "#3d8b5f",
    "panel": "#f7f9fc",
}


def _save_figure(fig, save_path: str) -> str:
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _plot_line_curve(x, y, title, xlabel, ylabel, save_path, color=REPORT_COLORS["blue"]):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker=".", linestyle="-", color=color, linewidth=2)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    return _save_figure(fig, save_path)


def _plot_calibration(calibration: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(calibration["mean_pred"], calibration["actual_rate"], marker="o", linestyle="-", color=REPORT_COLORS["amber"], linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#606a78", label="Perfect calibration")
    ax.set_title("Calibration by Score Decile", fontsize=13, weight="bold")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default rate")
    ax.legend(frameon=False)
    ax.grid(True, linestyle="--", alpha=0.35)
    return _save_figure(fig, save_path)


def _plot_score_distribution(scores: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    positives = scores[scores["y_true"] == 1]["y_score"]
    negatives = scores[scores["y_true"] == 0]["y_score"]
    ax.hist(negatives, bins=24, alpha=0.68, label="Non-default", color=REPORT_COLORS["blue"], edgecolor="white")
    ax.hist(positives, bins=24, alpha=0.68, label="Default", color=REPORT_COLORS["red"], edgecolor="white")
    ax.set_title("Predicted Score Distribution", fontsize=13, weight="bold")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    return _save_figure(fig, save_path)


def _plot_confusion_matrix(matrix: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    values = matrix.to_numpy()
    image = ax.imshow(values, cmap="Blues")
    ax.set_xticks(range(matrix.shape[1]), matrix.columns)
    ax.set_yticks(range(matrix.shape[0]), matrix.index)
    ax.set_title("Confusion Matrix", fontsize=13, weight="bold")
    threshold = values.max() / 2 if values.size else 0
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(
                col,
                row,
                f"{values[row, col]:,.0f}",
                ha="center",
                va="center",
                color="white" if values[row, col] > threshold else REPORT_COLORS["ink"],
                fontsize=12,
                weight="bold",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return _save_figure(fig, save_path)


def _plot_lift(lift: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, lift.shape[0] + 1)
    ax.bar(x, lift["default_rate"], color=REPORT_COLORS["teal"], alpha=0.82, label="Default rate")
    ax.plot(x, lift["cumulative_capture_rate"], color=REPORT_COLORS["red"], marker="o", linewidth=2, label="Cumulative capture")
    ax.set_title("Lift and Cumulative Default Capture", fontsize=13, weight="bold")
    ax.set_xlabel("Score decile, highest risk first")
    ax.set_ylabel("Rate")
    ax.set_xticks(x)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    return _save_figure(fig, save_path)


def _plot_significance(coefficients: pd.DataFrame, save_path: str, top_n: int = 24):
    if coefficients.empty:
        return None
    data = coefficients[coefficients["term"] != "intercept"].copy()
    if data.empty:
        return None
    data["abs_z"] = data["z_score"].abs()
    data = data.sort_values("abs_z", ascending=True).tail(top_n)
    colors = np.where(data["significant"], REPORT_COLORS["green"], REPORT_COLORS["red"])

    fig, ax = plt.subplots(figsize=(8, max(5, 0.28 * data.shape[0])))
    ax.barh(data["term"], data["z_score"], color=colors, alpha=0.9)
    ax.axvline(0, color="#667085", linewidth=1)
    ax.axvline(1.96, color="#667085", linestyle="--", linewidth=1)
    ax.axvline(-1.96, color="#667085", linestyle="--", linewidth=1)
    ax.set_title("Indicator Significance by Wald z-score", fontsize=13, weight="bold")
    ax.set_xlabel("z-score")
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    return _save_figure(fig, save_path)


def _plot_test_p_values(tests: pd.DataFrame, save_path: str):
    if tests.empty:
        return None
    data = tests.copy()
    data["minus_log10_p"] = -np.log10(data["p_value"].clip(lower=np.finfo(float).tiny))
    colors = np.where(data["p_value"] < 0.05, REPORT_COLORS["green"], REPORT_COLORS["amber"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(data["test"], data["minus_log10_p"], color=colors)
    ax.axvline(-np.log10(0.05), color="#667085", linestyle="--", linewidth=1, label="p = 0.05")
    ax.set_title("Statistical Test Strength", fontsize=13, weight="bold")
    ax.set_xlabel("-log10(p-value)")
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(frameon=False)
    return _save_figure(fig, save_path)


def _format_float(value):
    if isinstance(value, (float, np.floating)):
        return f"{value:,.4f}"
    return value


def _table_html(df: pd.DataFrame, class_name: str) -> str:
    if df is None or df.empty:
        return "<p class='note'>No rows to display.</p>"
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda x: "" if pd.isna(x) else f"{x:,.4f}")
    return display.to_html(index=False, classes=class_name, border=0, escape=True)


def _cross_validation_html(cross_validation: Optional[Iterable[Dict[str, object]]]) -> str:
    if not cross_validation:
        return ""
    cv_results = list(cross_validation)
    if not cv_results:
        return ""

    summary = pd.concat([result["summary"] for result in cv_results], ignore_index=True)
    folds = pd.concat([result["folds"] for result in cv_results], ignore_index=True)
    selected_metrics = ["roc_auc", "average_precision", "brier_score", "ks_statistic", "f1_score"]
    compact_rows = []
    for _, row in summary[summary["metric"].isin(selected_metrics)].iterrows():
        compact_rows.append(
            {
                "model": row["model"],
                "metric": row["metric"],
                "mean": row["mean"],
                "std": row["std"],
                "min": row["min"],
                "max": row["max"],
            }
        )

    return (
        "<section><h2>Cross-Validation</h2>"
        "<p class='note'>Stratified folds estimate out-of-sample stability. Higher is better except for Brier score.</p>"
        f"{_table_html(pd.DataFrame(compact_rows), 'cv-summary-table')}"
        "<h3>Fold Results</h3>"
        f"{_table_html(folds, 'cv-folds-table')}"
        "</section>"
    )


def _sampling_diagnostics_html(sampling_diagnostics: Optional[Dict[str, object]]) -> str:
    if not sampling_diagnostics:
        return ""
    parts = ["<section><h2>Sampling and Weighting Diagnostics</h2>"]
    representative = sampling_diagnostics.get("representative", {})
    if representative:
        parts.extend(
            [
                "<h3>Representative Sample</h3>",
                (
                    "<p class='note'>"
                    f"Sampled {representative['sample_size']:,} of {representative['full_size']:,} rows "
                    f"({representative['sample_share']:.1%}) using stratified proportional sampling."
                    "</p>"
                ),
                _table_html(representative["strata"].head(20), "sampling-strata-table"),
                "<h3>Numeric Balance</h3>",
                _table_html(representative["numeric_balance"], "numeric-balance-table"),
            ]
        )
    weighted = sampling_diagnostics.get("weighted", {})
    if weighted:
        parts.extend(
            [
                "<h3>Inverse-Frequency Weights</h3>",
                (
                    "<p class='note'>"
                    f"Mean weight {weighted['mean_weight']:.4f}; min {weighted['min_weight']:.4f}; "
                    f"max {weighted['max_weight']:.4f}; effective sample size {weighted['effective_sample_size']:.0f}."
                    "</p>"
                ),
                _table_html(weighted["strata"].head(20), "weight-strata-table"),
            ]
        )
    parts.append("</section>")
    return "".join(parts)


def _segment_performance_html(segment_performance: Optional[pd.DataFrame]) -> str:
    if segment_performance is None or segment_performance.empty:
        return ""
    display = segment_performance.sort_values(
        ["segment", "value", "model"],
        key=lambda values: values.astype(str),
    )
    return (
        "<section><h2>Segment Stability</h2>"
        "<p class='note'>Segment metrics show whether sampling or weighting improves stability across portfolio groups.</p>"
        f"{_table_html(display, 'segment-performance-table')}"
        "</section>"
    )


def _metric_cards(metrics: Dict[str, float]) -> str:
    featured = [
        ("ROC AUC", metrics["roc_auc"], "Ranking quality"),
        ("Avg. precision", metrics["average_precision"], "Precision-recall area"),
        ("Brier score", metrics["brier_score"], "Calibration loss"),
        ("KS", metrics["ks_statistic"], "Score separation"),
        ("AIC", metrics["aic"], "Fit with complexity penalty"),
        ("McFadden R2", metrics["mcfadden_r2"], "Pseudo-R2"),
    ]
    cards = ["<div class='metric-grid'>"]
    for label, value, note in featured:
        cards.append(
            "<article class='metric-card'>"
            f"<span>{html.escape(label)}</span>"
            f"<strong>{_format_float(value)}</strong>"
            f"<small>{html.escape(note)}</small>"
            "</article>"
        )
    cards.append("</div>")
    return "\n".join(cards)


def _commentary(evaluation: Dict[str, object]) -> str:
    metrics = evaluation["metrics"]
    non_significant = evaluation.get("non_significant_predictors", [])
    non_significant_groups = evaluation.get("non_significant_groups", [])
    tests = evaluation.get("global_tests", pd.DataFrame())
    hl = tests.loc[tests["test"].eq("Hosmer-Lemeshow calibration"), "p_value"]
    calibration_comment = "Grouped calibration does not show a statistically significant mismatch."
    if not hl.empty and hl.iloc[0] < 0.05:
        calibration_comment = "Grouped calibration is statistically imperfect; score calibration should be monitored before deployment."
    significance_parts = []
    if non_significant_groups:
        significance_parts.append(
            f"{len(non_significant_groups)} dummy-variable groups fail the LR screen and are candidates for grouped removal."
        )
    if non_significant:
        significance_parts.append(
            f"{len(non_significant)} ungrouped indicators fail the Wald screen."
        )
    significance_comment = " ".join(significance_parts) if significance_parts else "Grouped LR and ungrouped Wald screens do not flag removable indicators."
    return (
        "<div class='comment-box'>"
        f"<p><strong>Model read:</strong> ROC AUC is {_format_float(metrics['roc_auc'])}, "
        f"KS is {_format_float(metrics['ks_statistic'])}, and Brier score is {_format_float(metrics['brier_score'])}.</p>"
        f"<p><strong>Statistical tests:</strong> {html.escape(significance_comment)} {html.escape(calibration_comment)}</p>"
        "</div>"
    )


def _render_model_section(evaluation: Dict[str, object], save_dir: str, prefix: str) -> str:
    image_paths = []
    image_paths.append(
        (
            "ROC Curve",
            _plot_line_curve(
                evaluation["roc_curve"]["fpr"],
                evaluation["roc_curve"]["tpr"],
                "ROC Curve",
                "False positive rate",
                "True positive rate",
                os.path.join(save_dir, f"{prefix}_roc_curve.png"),
            ),
            "Higher curves indicate stronger ranking of defaults above non-defaults.",
        )
    )
    image_paths.append(
        (
            "Precision-Recall Curve",
            _plot_line_curve(
                evaluation["pr_curve"]["recall"],
                evaluation["pr_curve"]["precision"],
                "Precision-Recall Curve",
                "Recall",
                "Precision",
                os.path.join(save_dir, f"{prefix}_pr_curve.png"),
                color=REPORT_COLORS["teal"],
            ),
            "Useful when defaults are a minority class because it focuses on positive-class retrieval.",
        )
    )
    image_paths.append(("Calibration", _plot_calibration(evaluation["calibration"], os.path.join(save_dir, f"{prefix}_calibration_curve.png")), "Observed rates should track the dashed perfect-calibration line."))
    image_paths.append(("Score Distribution", _plot_score_distribution(evaluation["scores"], os.path.join(save_dir, f"{prefix}_score_distribution.png")), "Good risk scores separate the default and non-default distributions."))
    image_paths.append(("Confusion Matrix", _plot_confusion_matrix(evaluation["confusion_matrix"], os.path.join(save_dir, f"{prefix}_confusion_matrix.png")), "Counts are computed at the selected operating threshold."))
    image_paths.append(("Lift", _plot_lift(evaluation["lift"], os.path.join(save_dir, f"{prefix}_lift_curve.png")), "The first deciles should capture a disproportionate share of defaults."))

    significance_path = _plot_significance(
        evaluation["coefficient_significance"],
        os.path.join(save_dir, f"{prefix}_coefficient_significance.png"),
    )
    if significance_path:
        image_paths.append(("Indicator Significance", significance_path, "Bars inside the dashed +/-1.96 region are weak at the 5% level."))

    p_value_path = _plot_test_p_values(
        evaluation["global_tests"],
        os.path.join(save_dir, f"{prefix}_statistical_tests.png"),
    )
    if p_value_path:
        image_paths.append(("Statistical Tests", p_value_path, "Longer bars mean smaller p-values and stronger evidence against the null."))

    metrics_table = pd.DataFrame(evaluation["metrics"].items(), columns=["metric", "value"])
    coefficients = evaluation["coefficient_significance"].copy()
    if not coefficients.empty:
        coefficients = coefficients.sort_values(["significant", "p_value"], ascending=[True, True])

    parts = [
        f"<section class='model-section'><h2>{html.escape(evaluation['model_name'])}</h2>",
        _metric_cards(evaluation["metrics"]),
        _commentary(evaluation),
        "<h3>Metrics</h3>",
        _table_html(metrics_table, "metrics-table"),
        "<h3>Statistical Tests</h3>",
        _table_html(evaluation["global_tests"], "tests-table"),
        "<h3>Grouped Dummy LR Tests</h3>",
        _table_html(evaluation.get("group_likelihood_ratio_tests", pd.DataFrame()), "group-tests-table"),
        "<h3>Indicator Significance</h3>",
        _table_html(coefficients, "coefficients-table"),
        "<h3>Calibration Deciles</h3>",
        _table_html(evaluation["calibration"], "calibration-table"),
        "<h3>Lift Summary</h3>",
        _table_html(evaluation["lift"], "lift-table"),
        "<div class='figure-grid'>",
    ]
    for title, image_path, caption in image_paths:
        parts.append(
            "<figure>"
            f"<img src='{html.escape(os.path.basename(image_path))}' alt='{html.escape(title)}'>"
            f"<figcaption><strong>{html.escape(title)}.</strong> {html.escape(caption)}</figcaption>"
            "</figure>"
        )
    parts.extend(["</div>", "</section>"])
    return "\n".join(parts)


def create_html_evaluation_report(
    evaluations: Iterable[Dict[str, object]] | Dict[str, object],
    save_dir: str = "evaluation",
    report_name: str = "evaluation_report.html",
    comparison: Optional[pd.DataFrame] = None,
    cross_validation: Optional[Iterable[Dict[str, object]]] = None,
    sampling_diagnostics: Optional[Dict[str, object]] = None,
    segment_performance: Optional[pd.DataFrame] = None,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    if isinstance(evaluations, dict):
        evaluations = [evaluations]
    evaluations = list(evaluations)

    report_path = os.path.join(save_dir, report_name)
    model_sections = [
        _render_model_section(evaluation, save_dir, f"model_{index + 1}")
        for index, evaluation in enumerate(evaluations)
    ]

    comparison_html = ""
    if comparison is not None and not comparison.empty:
        comparison_html = (
            "<section><h2>Model Comparison</h2>"
            "<p class='note'>The adjusted model removes whole dummy-variable groups that fail nested likelihood-ratio tests, plus ungrouped indicators that fail the Wald screen.</p>"
            f"{_table_html(comparison, 'comparison-table')}"
            "</section>"
        )

    html_body = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>Model Evaluation Report</title>",
        "<style>",
        ":root { --ink: #1f2933; --muted: #596574; --line: #d8dee8; --panel: #f7f9fc; --blue: #2f6fbb; }",
        "body { margin: 0; font-family: Inter, Segoe UI, Arial, sans-serif; color: var(--ink); background: #ffffff; }",
        "header { padding: 28px 36px 18px; border-bottom: 1px solid var(--line); background: #f4f7fb; }",
        "main { padding: 24px 36px 40px; max-width: 1320px; margin: 0 auto; }",
        "h1 { margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }",
        "h2 { margin: 28px 0 14px; font-size: 22px; letter-spacing: 0; }",
        "h3 { margin: 22px 0 8px; font-size: 16px; letter-spacing: 0; }",
        "p { line-height: 1.5; }",
        ".note { color: var(--muted); margin-top: 0; }",
        ".comment-box { background: #f7f9fc; border-left: 4px solid var(--blue); padding: 12px 16px; margin: 16px 0; }",
        ".comment-box p { margin: 6px 0; }",
        ".metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 18px 0; }",
        ".metric-card { border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: white; }",
        ".metric-card span { display: block; color: var(--muted); font-size: 13px; }",
        ".metric-card strong { display: block; margin: 6px 0; font-size: 24px; }",
        ".metric-card small { color: var(--muted); }",
        "table { border-collapse: collapse; width: 100%; margin: 10px 0 18px; font-size: 13px; }",
        "th, td { border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; }",
        "th { background: #eef3f8; font-weight: 700; }",
        "tr:nth-child(even) td { background: #fbfcfe; }",
        ".figure-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 18px; }",
        "figure { margin: 0; border: 1px solid var(--line); border-radius: 8px; background: white; overflow: hidden; }",
        "figure img { width: 100%; height: auto; display: block; }",
        "figcaption { border-top: 1px solid var(--line); padding: 10px 12px; color: var(--muted); font-size: 13px; line-height: 1.45; }",
        ".model-section { border-bottom: 1px solid var(--line); padding-bottom: 26px; }",
        "@media (max-width: 760px) { header, main { padding-left: 18px; padding-right: 18px; } .figure-grid { grid-template-columns: 1fr; } }",
        "</style>",
        "</head>",
        "<body>",
        "<header><h1>Model Evaluation Report</h1><p class='note'>Classification performance, statistical diagnostics, grouped dummy-variable tests, indicator significance, and adjusted-model comparison.</p></header>",
        "<main>",
        comparison_html,
        _sampling_diagnostics_html(sampling_diagnostics),
        _cross_validation_html(cross_validation),
        _segment_performance_html(segment_performance),
        *model_sections,
        "</main>",
        "</body>",
        "</html>",
    ]

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_body))
    return report_path
