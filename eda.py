import math
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _is_categorical_series(series: pd.Series, category_threshold: int) -> bool:
    if pd.api.types.is_categorical_dtype(series.dtype) or series.dtype == object or pd.api.types.is_bool_dtype(series.dtype):
        return True
    if pd.api.types.is_integer_dtype(series.dtype) and series.nunique(dropna=True) <= category_threshold:
        return True
    return False


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


def _numeric_summary(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    summary = []
    for column in numeric_columns:
        values = df[column].dropna()
        summary.append(
            {
                "feature": column,
                "count": int(values.count()),
                "missing": int(df[column].isna().sum()),
                "missing_pct": float(df[column].isna().mean()) * 100,
                "min": float(values.min()) if not values.empty else np.nan,
                "q1": float(values.quantile(0.25)) if not values.empty else np.nan,
                "median": float(values.median()) if not values.empty else np.nan,
                "mean": float(values.mean()) if not values.empty else np.nan,
                "q3": float(values.quantile(0.75)) if not values.empty else np.nan,
                "max": float(values.max()) if not values.empty else np.nan,
                "std": float(values.std(ddof=1)) if not values.empty else np.nan,
                "skew": float(values.skew()) if not values.empty else np.nan,
            }
        )
    return pd.DataFrame(summary).set_index("feature")


def _categorical_summary(df: pd.DataFrame, categorical_columns: list[str], top_n: int = 5) -> pd.DataFrame:
    summary = []
    for column in categorical_columns:
        values = df[column]
        value_counts = values.fillna("<missing>").value_counts(dropna=False)
        top_values = ", ".join(
            f"{label} ({count})" for label, count in value_counts.head(top_n).items()
        )
        summary.append(
            {
                "feature": column,
                "count": int(values.count()),
                "missing": int(values.isna().sum()),
                "missing_pct": float(values.isna().mean()) * 100,
                "unique_values": int(values.nunique(dropna=False)),
                "top_values": top_values,
            }
        )
    return pd.DataFrame(summary).set_index("feature")


def _bonferroni_outlier_test(series: pd.Series) -> pd.DataFrame:
    values = series.dropna()
    if values.size < 3:
        return pd.DataFrame(columns=["value", "z_score", "p_value", "bonferroni_p"])

    mean = values.mean()
    std = values.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.DataFrame(columns=["value", "z_score", "p_value", "bonferroni_p"])

    z_scores = (values - mean) / std
    p_values = np.array([math.erfc(abs(z) / math.sqrt(2)) for z in z_scores])
    bonferroni_p = np.minimum(p_values * values.size, 1.0)

    results = pd.DataFrame(
        {
            "value": values,
            "z_score": z_scores,
            "p_value": p_values,
            "bonferroni_p": bonferroni_p,
        }
    )
    return results.sort_values("bonferroni_p")


def _bonferroni_outlier_summary(df: pd.DataFrame, numeric_columns: list[str], alpha: float = 0.05) -> pd.DataFrame:
    summary = []
    for column in numeric_columns:
        outliers = _bonferroni_outlier_test(df[column])
        outlier_count = int((outliers["bonferroni_p"] < alpha).sum())
        top_outliers = ", ".join(
            f"{idx}:{val:.2f}" for idx, val in outliers.head(5)[["value"]].itertuples(index=True, name=None)
        )
        summary.append(
            {
                "feature": column,
                "count": int(df[column].dropna().count()),
                "outlier_count": outlier_count,
                "outlier_pct": float(outlier_count / df[column].dropna().shape[0]) * 100 if df[column].dropna().shape[0] > 0 else 0.0,
                "top_outliers": top_outliers or "None",
            }
        )
    return pd.DataFrame(summary).set_index("feature")


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


def _image_section_html(title: str, image_paths: list[str]) -> str:
    body = [f"<section><h2>{title}</h2>"]
    for image_path in image_paths:
        body.append(
            f"<figure><img src=\"{os.path.basename(image_path)}\" alt=\"{os.path.basename(image_path)}\"><figcaption>{os.path.basename(image_path)}</figcaption></figure>"
        )
    body.append("</section>")
    return "\n".join(body)


def _create_html_report(
    df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    report_path: str,
    save_dir: str,
) -> None:
    numeric_stats = _numeric_summary(df, numeric_columns)
    categorical_stats = _categorical_summary(df, categorical_columns)
    outlier_stats = _bonferroni_outlier_summary(df, numeric_columns)

    numeric_images = [
        os.path.join(save_dir, f"{column}_numeric.png") for column in numeric_columns
        if os.path.exists(os.path.join(save_dir, f"{column}_numeric.png"))
    ]
    categorical_images = [
        os.path.join(save_dir, f"{column}_categorical.png") for column in categorical_columns
        if os.path.exists(os.path.join(save_dir, f"{column}_categorical.png"))
    ]

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
        "figure { display: inline-block; margin: 12px; width: calc(50% - 24px); vertical-align: top; }",
        "figure img { width: 100%; height: auto; border: 1px solid #ddd; }",
        "figcaption { text-align: center; font-size: 0.9em; margin-top: 6px; color: #555; }",
        "</style>",
        "</head>",
        "<body>",
        "<header><h1>EDA Summary Report</h1>",
        f"<p>Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns</p>",
        f"<p>Numeric features: {len(numeric_columns)} | Categorical features: {len(categorical_columns)}</p>",
        "</header>",
    ]

    if not numeric_stats.empty:
        html.append(_render_dataframe_html("Numeric Feature Summary", numeric_stats))

    if not categorical_stats.empty:
        html.append(_render_dataframe_html("Categorical Feature Summary", categorical_stats))

    if not outlier_stats.empty:
        html.append(_render_dataframe_html("Bonferroni Outlier Summary", outlier_stats))

    if numeric_images:
        html.append(_image_section_html("Numeric Feature Visualizations", numeric_images))

    if categorical_images:
        html.append(_image_section_html("Categorical Feature Visualizations", categorical_images))

    html.append("</body>")
    html.append("</html>")

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(html))


def run_eda(
    df: pd.DataFrame,
    save_dir: Optional[str] = "eda_plots",
    category_threshold: int = 10,
    show: bool = False,
) -> None:
    """Run exploratory data analysis on the provided DataFrame.

    Numeric features are visualized with histogram + boxplot. Categorical features are visualized with bar plots.
    The generated plots are saved in `save_dir`.
    """
    if save_dir is None:
        save_dir = "eda_plots"
    os.makedirs(save_dir, exist_ok=True)

    categorical_columns = [
        name for name in df.columns if _is_categorical_series(df[name], category_threshold)
    ]
    numeric_columns = [
        name for name in df.select_dtypes(include=[np.number]).columns if name not in categorical_columns
    ]

    if numeric_columns:
        for column in numeric_columns:
            file_name = f"{column}_numeric.png"
            save_path = os.path.join(save_dir, file_name)
            _plot_numeric(df[column], save_path=save_path, show=show)

    if categorical_columns:
        for column in categorical_columns:
            file_name = f"{column}_categorical.png"
            save_path = os.path.join(save_dir, file_name)
            _plot_categorical(df[column], save_path=save_path, show=show)

    report_path = os.path.join(save_dir, "eda_report.html")
    _create_html_report(df, numeric_columns, categorical_columns, report_path, save_dir)
    print(f"EDA completed. Plots saved to: {save_dir}. HTML report saved to: {report_path}")
