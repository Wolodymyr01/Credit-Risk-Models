import numpy as np
import pandas as pd


REPRESENTATIVE_STRATA = [
    "loan_status",
    "loan_grade",
    "loan_intent",
    "person_home_ownership",
]


def make_strata_key(df: pd.DataFrame, strata_columns: list[str] | None = None) -> pd.Series:
    columns = strata_columns or REPRESENTATIVE_STRATA
    return df[columns].astype(str).agg(" | ".join, axis=1)


def representative_sample_indices(
    df: pd.DataFrame,
    strata_columns: list[str] | None = None,
    sample_fraction: float = 0.5,
    min_per_stratum: int = 2,
    random_state: int = 42,
) -> pd.Index:
    strata_key = make_strata_key(df, strata_columns)
    sampled_indices = []
    for _, indices in strata_key.groupby(strata_key).groups.items():
        group_indices = pd.Index(indices)
        sample_size = int(round(len(group_indices) * sample_fraction))
        sample_size = max(min(sample_size, len(group_indices)), min(min_per_stratum, len(group_indices)))
        sampled = group_indices.to_series().sample(n=sample_size, random_state=random_state)
        sampled_indices.extend(sampled.tolist())
    return pd.Index(sampled_indices)


def sampling_diagnostics(
    df: pd.DataFrame,
    sample_index: pd.Index,
    strata_columns: list[str] | None = None,
) -> pd.DataFrame:
    strata_key = make_strata_key(df, strata_columns)
    full_counts = strata_key.value_counts().rename("full_count")
    sample_counts = strata_key.loc[sample_index].value_counts().rename("sample_count")
    diagnostics = pd.concat([full_counts, sample_counts], axis=1).fillna(0)
    diagnostics["full_share"] = diagnostics["full_count"] / diagnostics["full_count"].sum()
    diagnostics["sample_share"] = diagnostics["sample_count"] / diagnostics["sample_count"].sum()
    diagnostics["share_diff"] = diagnostics["sample_share"] - diagnostics["full_share"]
    diagnostics["retained_share"] = diagnostics["sample_count"] / diagnostics["full_count"]
    diagnostics = diagnostics.reset_index().rename(columns={"index": "stratum"})
    return diagnostics.sort_values("share_diff", key=lambda values: values.abs(), ascending=False)


def numeric_balance_diagnostics(
    df: pd.DataFrame,
    sample_index: pd.Index,
    numeric_columns: list[str],
) -> pd.DataFrame:
    rows = []
    sample_df = df.loc[sample_index]
    for column in numeric_columns:
        full_values = df[column].dropna()
        sample_values = sample_df[column].dropna()
        pooled_std = np.sqrt((full_values.var(ddof=1) + sample_values.var(ddof=1)) / 2)
        standardized_mean_diff = (
            (sample_values.mean() - full_values.mean()) / pooled_std
            if pooled_std and not np.isnan(pooled_std)
            else 0.0
        )
        rows.append(
            {
                "feature": column,
                "full_mean": full_values.mean(),
                "sample_mean": sample_values.mean(),
                "full_median": full_values.median(),
                "sample_median": sample_values.median(),
                "standardized_mean_diff": standardized_mean_diff,
            }
        )
    return pd.DataFrame(rows).sort_values("standardized_mean_diff", key=lambda values: values.abs(), ascending=False)


def inverse_frequency_weights(
    df: pd.DataFrame,
    strata_columns: list[str] | None = None,
) -> pd.Series:
    strata_key = make_strata_key(df, strata_columns)
    counts = strata_key.value_counts()
    weights = strata_key.map(lambda key: 1.0 / counts[key]).astype(float)
    weights = weights / weights.mean()
    weights.name = "sample_weight"
    return weights


def weight_diagnostics(
    df: pd.DataFrame,
    weights: pd.Series,
    strata_columns: list[str] | None = None,
) -> pd.DataFrame:
    strata_key = make_strata_key(df, strata_columns)
    diagnostics = (
        pd.DataFrame({"stratum": strata_key, "weight": weights})
        .groupby("stratum")
        .agg(
            count=("weight", "count"),
            mean_weight=("weight", "mean"),
            min_weight=("weight", "min"),
            max_weight=("weight", "max"),
            total_weight=("weight", "sum"),
        )
        .reset_index()
    )
    return diagnostics.sort_values("mean_weight", ascending=False)


def effective_sample_size(weights: pd.Series) -> float:
    return float((weights.sum() ** 2) / (weights.pow(2).sum()))
