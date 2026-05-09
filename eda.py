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

    print(f"EDA completed. Plots saved to: {save_dir}")
