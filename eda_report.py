import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


def _plot_numeric(series: pd.Series, save_path: str, show: bool) -> None:
    data = series.dropna()
    if data.empty:
        return

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    fig.suptitle(f"Numeric feature: {series.name}", fontsize=14)

    axes[0].hist(data, bins=20, color="#4c72b0", edgecolor="black")
    axes[0].set_title("Histogram")
    axes[0].set_xlabel(series.name)
    axes[0].set_ylabel("Count")

    axes[1].boxplot(data, vert=False, patch_artist=True, boxprops={"facecolor": "#55a868"})
    axes[1].set_title("Boxplot")
    axes[1].set_xlabel(series.name)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)


def _plot_categorical(series: pd.Series, save_path: str, show: bool) -> None:
    counts = series.fillna("<missing>").value_counts(dropna=False)
    if counts.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    labels = counts.index.astype(str).tolist()
    values = counts.values.tolist()
    ax.bar(range(len(values)), values, color="#dd8452", edgecolor="black")
    ax.set_title(f"Categorical feature: {series.name}")
    ax.set_xlabel(series.name)
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)


def _plot_data_quality_tests(data_tests: pd.DataFrame, save_path: str, show: bool) -> None:
    if data_tests.empty:
        return
    severity = {"pass": 0, "warn": 1, "fail": 2}
    colors = {"pass": "#3d8b5f", "warn": "#c27c1a", "fail": "#b84a52"}
    data = data_tests.copy()
    data["severity"] = data["status"].map(severity).fillna(0)
    data = data.sort_values(["severity", "affected_rows"], ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * data.shape[0])))
    ax.barh(data["test"], data["affected_rows"], color=data["status"].map(colors))
    ax.set_title("Data Quality Test Findings")
    ax.set_xlabel("Affected rows or findings")
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)


def _plot_loan_grade_target_association(default_rates: pd.DataFrame, save_path: str, show: bool) -> None:
    if default_rates.empty:
        return

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax1.bar(default_rates["loan_grade"], default_rates["count"], color="#8aa6c8", alpha=0.75, label="Applications")
    ax2.plot(default_rates["loan_grade"], default_rates["default_rate"], color="#b84a52", marker="o", linewidth=2.2, label="Default rate")
    ax1.set_title("Loan Grade vs Loan Status")
    ax1.set_xlabel("Loan grade")
    ax1.set_ylabel("Application count")
    ax2.set_ylabel("Default rate")
    ax2.set_ylim(0, min(1.05, max(default_rates["default_rate"].max() * 1.15, 0.2)))
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left")
    plt.tight_layout()
    fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)


def _render_dataframe_html(title: str, df_summary: pd.DataFrame) -> str:
    body = f"<section><h2>{title}</h2>"
    body += df_summary.reset_index().to_html(
        classes="summary-table",
        index=False,
        border=0,
        justify="center",
        escape=True,
    )
    body += "</section>"
    return body


def _eda_recommendations_html() -> str:
    recommendations = [
        "Add target-aware EDA: default rate by loan grade, home ownership, intent, and missingness indicators.",
        "Track data-quality tests over time so new extracts fail fast when plausibility checks regress.",
        "Add bivariate numeric plots against loan_status, especially income, interest rate, loan-to-income, and credit history.",
        "Review outliers with business rules before deletion; age above 120 and impossible employment length should be quarantined.",
        "Use stratified train/test validation after EDA so performance estimates are not only in-sample.",
    ]
    items = "".join(f"<li>{item}</li>" for item in recommendations)
    return f"<section><h2>Suggested EDA Improvements</h2><ul class=\"recommendations\">{items}</ul></section>"


def _image_section_html(title: str, image_paths: list[str]) -> str:
    body = [f"<section><h2>{title}</h2>"]
    for image_path in image_paths:
        body.append(
            f"<figure><img src=\"{os.path.basename(image_path)}\" alt=\"{os.path.basename(image_path)}\"><figcaption>{os.path.basename(image_path)}</figcaption></figure>"
        )
    body.append("</section>")
    return "\n".join(body)


def _create_plots(df: pd.DataFrame, analysis: Dict[str, object], save_dir: str, show: bool) -> Dict[str, list[str] | str]:
    numeric_columns = analysis["numeric_columns"]
    categorical_columns = analysis["categorical_columns"]

    numeric_images = []
    for column in numeric_columns:
        save_path = os.path.join(save_dir, f"{column}_numeric.png")
        _plot_numeric(df[column], save_path=save_path, show=show)
        if os.path.exists(save_path):
            numeric_images.append(save_path)

    categorical_images = []
    for column in categorical_columns:
        save_path = os.path.join(save_dir, f"{column}_categorical.png")
        _plot_categorical(df[column], save_path=save_path, show=show)
        if os.path.exists(save_path):
            categorical_images.append(save_path)

    tests_image = os.path.join(save_dir, "data_quality_tests.png")
    _plot_data_quality_tests(analysis["data_quality_tests"], tests_image, show=False)
    loan_grade_image = os.path.join(save_dir, "loan_grade_target_association.png")
    loan_grade_association = analysis.get("loan_grade_target_association", {})
    if loan_grade_association:
        _plot_loan_grade_target_association(
            loan_grade_association["default_rates"],
            loan_grade_image,
            show=show,
        )

    return {
        "numeric": numeric_images,
        "categorical": categorical_images,
        "data_quality_tests": tests_image,
        "loan_grade_target_association": loan_grade_image,
    }


def create_eda_report(
    df: pd.DataFrame,
    analysis: Dict[str, object],
    save_dir: Optional[str] = "eda_plots",
    report_name: str = "eda_report.html",
    show: bool = False,
) -> str:
    if save_dir is None:
        save_dir = "eda_plots"
    os.makedirs(save_dir, exist_ok=True)

    image_paths = _create_plots(df, analysis, save_dir, show=show)
    numeric_columns = analysis["numeric_columns"]
    categorical_columns = analysis["categorical_columns"]
    numeric_stats = analysis["numeric_summary"]
    categorical_stats = analysis["categorical_summary"]
    outlier_stats = analysis["outlier_summary"]
    data_tests = analysis["data_quality_tests"]
    loan_grade_association = analysis.get("loan_grade_target_association", {})
    report_path = os.path.join(save_dir, report_name)

    html = [
        "<!DOCTYPE html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<title>EDA Summary Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 24px; color: #222; }",
        "h1, h2 { color: #333; }",
        "section { margin-bottom: 32px; }",
        "table.summary-table { border-collapse: collapse; width: 100%; margin-top: 12px; }",
        "table.summary-table th, table.summary-table td { border: 1px solid #ccc; padding: 8px; text-align: left; vertical-align: top; }",
        "table.summary-table th { background: #f5f5f5; }",
        "td { white-space: pre-wrap; word-break: break-word; }",
        ".recommendations li { margin-bottom: 8px; }",
        ".status-pass { color: #2f7d52; font-weight: 700; }",
        ".status-warn { color: #9a650f; font-weight: 700; }",
        ".status-fail { color: #a33f46; font-weight: 700; }",
        "figure { display: inline-block; margin: 12px; width: calc(50% - 24px); vertical-align: top; }",
        "figure img { width: 100%; height: auto; border: 1px solid #ddd; }",
        "figcaption { text-align: center; font-size: 0.9em; margin-top: 6px; color: #555; }",
        "</style>",
        "</head>",
        "<body>",
        "<header><h1>EDA Summary Report</h1>",
        f"<p>Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns</p>",
        f"<p>Numeric features: {len(numeric_columns)} | Categorical features: {len(categorical_columns)}</p>",
        "</header>",
    ]

    if not numeric_stats.empty:
        html.append(_render_dataframe_html("Numeric Feature Summary", numeric_stats))

    if not categorical_stats.empty:
        html.append(_render_dataframe_html("Categorical Feature Summary", categorical_stats))

    if not outlier_stats.empty:
        html.append(_render_dataframe_html("Bonferroni Outlier Summary", outlier_stats))

    if not data_tests.empty:
        styled_tests = data_tests.copy()
        styled_tests["status"] = styled_tests["status"].map(
            lambda value: f"<span class=\"status-{value}\">{value}</span>"
        )
        html.append("<section><h2>Data Quality Tests</h2>")
        html.append(
            styled_tests.to_html(
                classes="summary-table",
                index=False,
                border=0,
                justify="center",
                escape=False,
            )
        )
        tests_image = image_paths["data_quality_tests"]
        if os.path.exists(tests_image):
            html.append(
                f"<figure><img src=\"{os.path.basename(tests_image)}\" alt=\"Data quality tests\"><figcaption>Data quality test findings by affected rows.</figcaption></figure>"
            )
        html.append("</section>")

    html.append(_eda_recommendations_html())

    if loan_grade_association:
        html.append("<section><h2>Loan Grade Target Association</h2>")
        html.append(
            "<p>Loan grade behaves like a prepared external risk estimate. The association below is used to justify excluding it from academic models.</p>"
        )
        html.append(
            loan_grade_association["metrics"].to_html(
                classes="summary-table",
                index=False,
                border=0,
                justify="center",
                escape=True,
            )
        )
        html.append(
            loan_grade_association["default_rates"].to_html(
                classes="summary-table",
                index=False,
                border=0,
                justify="center",
                escape=True,
            )
        )
        loan_grade_image = image_paths["loan_grade_target_association"]
        if os.path.exists(loan_grade_image):
            html.append(
                f"<figure><img src=\"{os.path.basename(loan_grade_image)}\" alt=\"Loan grade target association\"><figcaption>Default rate rises sharply by loan grade.</figcaption></figure>"
            )
        html.append("</section>")

    if image_paths["numeric"]:
        html.append(_image_section_html("Numeric Feature Visualizations", image_paths["numeric"]))

    if image_paths["categorical"]:
        html.append(_image_section_html("Categorical Feature Visualizations", image_paths["categorical"]))

    html.append("</body>")
    html.append("</html>")

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(html))
    return report_path
