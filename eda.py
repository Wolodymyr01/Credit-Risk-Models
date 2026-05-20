from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats


def is_categorical_series(series: pd.Series, category_threshold: int) -> bool:
    if pd.api.types.is_categorical_dtype(series.dtype) or series.dtype == object or pd.api.types.is_bool_dtype(series.dtype):
        return True
    if pd.api.types.is_integer_dtype(series.dtype) and series.nunique(dropna=True) <= category_threshold:
        return True
    return False


def infer_feature_types(df: pd.DataFrame, category_threshold: int = 10) -> Dict[str, list[str]]:
    categorical_columns = [
        name for name in df.columns if is_categorical_series(df[name], category_threshold)
    ]
    numeric_columns = [
        name for name in df.select_dtypes(include=[np.number]).columns if name not in categorical_columns
    ]
    return {"numeric": numeric_columns, "categorical": categorical_columns}


def numeric_summary(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
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


def categorical_summary(df: pd.DataFrame, categorical_columns: list[str], top_n: int = 5) -> pd.DataFrame:
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


def bonferroni_outlier_test(series: pd.Series) -> pd.DataFrame:
    """Flag univariate normal-approximation outliers with Bonferroni-adjusted p-values.

    SciPy does not expose a dedicated Bonferroni outlier-test function. It does provide
    the standard normal tail calculation used here; the Bonferroni adjustment is the
    usual min(raw_p * number_of_tests, 1) correction.
    """
    values = series.dropna()
    if values.size < 3:
        return pd.DataFrame(columns=["value", "z_score", "p_value", "bonferroni_p"])

    std = values.std(ddof=1)
    if std == 0 or np.isnan(std):
        return pd.DataFrame(columns=["value", "z_score", "p_value", "bonferroni_p"])

    z_scores = stats.zscore(values.to_numpy(dtype=float), ddof=1)
    p_values = 2 * stats.norm.sf(np.abs(z_scores))
    bonferroni_p = np.minimum(p_values * values.size, 1.0)

    results = pd.DataFrame(
        {
            "value": values,
            "z_score": z_scores,
            "p_value": p_values,
            "bonferroni_p": bonferroni_p,
        },
        index=values.index,
    )
    return results.sort_values("bonferroni_p")


def bonferroni_outlier_summary(df: pd.DataFrame, numeric_columns: list[str], alpha: float = 0.05) -> pd.DataFrame:
    summary = []
    for column in numeric_columns:
        outliers = bonferroni_outlier_test(df[column])
        outlier_count = int((outliers["bonferroni_p"] < alpha).sum())
        top_outliers = ", ".join(
            f"{idx}:{val:.2f}" for idx, val in outliers.head(5)[["value"]].itertuples(index=True, name=None)
        )
        non_missing_count = int(df[column].dropna().shape[0])
        summary.append(
            {
                "feature": column,
                "count": non_missing_count,
                "outlier_count": outlier_count,
                "outlier_pct": float(outlier_count / non_missing_count) * 100 if non_missing_count else 0.0,
                "top_outliers": top_outliers or "None",
                "method": "two-sided z test + Bonferroni correction",
            }
        )
    return pd.DataFrame(summary).set_index("feature")


def run_data_quality_tests(df: pd.DataFrame, target_column: str = "loan_status") -> pd.DataFrame:
    """Run lightweight dataset checks that are useful before credit-risk modeling."""
    checks = []

    def add_check(name: str, status: str, details: str, affected_rows: int = 0) -> None:
        checks.append(
            {
                "test": name,
                "status": status,
                "affected_rows": int(affected_rows),
                "details": details,
            }
        )

    duplicate_rows = int(df.duplicated().sum())
    add_check(
        "Duplicate rows",
        "pass" if duplicate_rows == 0 else "warn",
        "No exact duplicate rows found." if duplicate_rows == 0 else "Exact duplicate rows should be reviewed before modeling.",
        duplicate_rows,
    )

    if target_column in df.columns:
        target_values = set(df[target_column].dropna().unique())
        target_missing = int(df[target_column].isna().sum())
        is_binary = target_values.issubset({0, 1}) and len(target_values) == 2
        add_check(
            "Binary target validity",
            "pass" if is_binary and target_missing == 0 else "fail",
            f"Observed target values: {sorted(target_values)}; missing target rows: {target_missing}.",
            target_missing,
        )
        default_rate = float(df[target_column].mean())
        add_check(
            "Target imbalance",
            "pass" if 0.05 <= default_rate <= 0.5 else "warn",
            f"Default rate is {default_rate:.2%}; inspect precision-recall metrics when the event class is rare.",
        )

    missing_pct = df.isna().mean()
    high_missing = missing_pct[missing_pct > 0.2]
    add_check(
        "High missingness columns",
        "pass" if high_missing.empty else "warn",
        "No column has more than 20% missing values."
        if high_missing.empty
        else ", ".join(f"{column}: {value:.1%}" for column, value in high_missing.items()),
        int(df[high_missing.index].isna().any(axis=1).sum()) if not high_missing.empty else 0,
    )

    nonnegative_columns = [
        column for column in df.select_dtypes(include=[np.number]).columns
        if any(token in column for token in ["age", "income", "length", "rate", "amount", "amnt", "percent"])
    ]
    negative_cells = int((df[nonnegative_columns] < 0).sum().sum()) if nonnegative_columns else 0
    add_check(
        "Non-negative numeric fields",
        "pass" if negative_cells == 0 else "fail",
        "No negative values found in numeric fields that should be non-negative."
        if negative_cells == 0
        else "Negative values found in fields that should be non-negative.",
        negative_cells,
    )

    if "person_age" in df.columns:
        unrealistic_age = int((df["person_age"] > 120).sum())
        add_check(
            "Age plausibility",
            "pass" if unrealistic_age == 0 else "warn",
            "No borrower age above 120."
            if unrealistic_age == 0
            else "Borrower ages above 120 look like data-entry errors.",
            unrealistic_age,
        )

    if {"person_emp_length", "person_age"}.issubset(df.columns):
        impossible_employment = int((df["person_emp_length"] > (df["person_age"] - 14)).fillna(False).sum())
        add_check(
            "Employment length plausibility",
            "pass" if impossible_employment == 0 else "warn",
            "Employment length does not exceed a plausible working-age window."
            if impossible_employment == 0
            else "Employment length exceeds age minus 14 for some rows.",
            impossible_employment,
        )

    if {"cb_person_cred_hist_length", "person_age"}.issubset(df.columns):
        impossible_credit_history = int((df["cb_person_cred_hist_length"] > df["person_age"]).sum())
        add_check(
            "Credit history plausibility",
            "pass" if impossible_credit_history == 0 else "warn",
            "Credit history length is not greater than borrower age."
            if impossible_credit_history == 0
            else "Credit history length is greater than borrower age for some rows.",
            impossible_credit_history,
        )

    if {"loan_amnt", "person_income", "loan_percent_income"}.issubset(df.columns):
        expected_ratio = df["loan_amnt"] / df["person_income"].replace(0, np.nan)
        mismatch = (expected_ratio - df["loan_percent_income"]).abs() > 0.015
        mismatch_count = int(mismatch.fillna(False).sum())
        add_check(
            "Loan percent income consistency",
            "pass" if mismatch_count == 0 else "warn",
            "loan_percent_income is consistent with loan_amnt / person_income within tolerance."
            if mismatch_count == 0
            else "loan_percent_income differs from loan_amnt / person_income beyond tolerance.",
            mismatch_count,
        )

    categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns
    high_cardinality = [column for column in categorical_columns if df[column].nunique(dropna=True) > 50]
    add_check(
        "Categorical cardinality",
        "pass" if not high_cardinality else "warn",
        "No high-cardinality categorical columns found."
        if not high_cardinality
        else "High-cardinality columns may need grouping: " + ", ".join(high_cardinality),
    )

    if target_column in df.columns:
        missing_assoc = []
        for column in df.columns:
            if column == target_column or not df[column].isna().any():
                continue
            indicator = df[column].isna().astype(int)
            table = pd.crosstab(indicator, df[target_column])
            if table.shape == (2, 2):
                _, p_value, _, _ = stats.chi2_contingency(table)
                missing_assoc.append((column, p_value))
        associated = [(column, p_value) for column, p_value in missing_assoc if p_value < 0.05]
        add_check(
            "Missingness association with target",
            "pass" if not associated else "warn",
            "No missingness indicator is significantly associated with the target at alpha 0.05."
            if not associated
            else ", ".join(f"{column}: p={p_value:.4g}" for column, p_value in associated),
            len(associated),
        )

    return pd.DataFrame(checks)


def analyze_eda(
    df: pd.DataFrame,
    category_threshold: int = 10,
    target_column: str = "loan_status",
) -> Dict[str, object]:
    feature_types = infer_feature_types(df, category_threshold)
    numeric_columns = feature_types["numeric"]
    categorical_columns = feature_types["categorical"]
    return {
        "shape": df.shape,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "numeric_summary": numeric_summary(df, numeric_columns),
        "categorical_summary": categorical_summary(df, categorical_columns),
        "outlier_summary": bonferroni_outlier_summary(df, numeric_columns),
        "data_quality_tests": run_data_quality_tests(df, target_column=target_column),
    }


def run_eda(
    df: pd.DataFrame,
    save_dir: Optional[str] = "eda_plots",
    category_threshold: int = 10,
    show: bool = False,
) -> None:
    """Backward-compatible wrapper that analyzes EDA and renders its report."""
    from eda_report import create_eda_report

    analysis = analyze_eda(df, category_threshold=category_threshold)
    report_path = create_eda_report(df, analysis, save_dir=save_dir, show=show)
    print(f"EDA completed. Plots saved to: {save_dir}. HTML report saved to: {report_path}")


_is_categorical_series = is_categorical_series
_numeric_summary = numeric_summary
_categorical_summary = categorical_summary
_bonferroni_outlier_test = bonferroni_outlier_test
_bonferroni_outlier_summary = bonferroni_outlier_summary
