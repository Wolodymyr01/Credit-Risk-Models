import warnings
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


RAW_LOGIT_CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file",
]

DEFAULT_PAIRWISE_CORRELATION_THRESHOLD = 0.7
LOAN_GRADE_ORDER = {
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
    "E": 5,
    "F": 6,
    "G": 7,
}


def _prepare_raw_logit_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    if "emp_length_missing" not in df.columns:
        df["emp_length_missing"] = df["person_emp_length"].isna().astype(int)
    if "int_rate_missing" not in df.columns:
        df["int_rate_missing"] = df["loan_int_rate"].isna().astype(int)
    if "emp_length" not in df.columns:
        df["emp_length"] = df["person_emp_length"].fillna(0)

    if df["loan_int_rate"].isna().any():
        df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())
    if "loan_grade" in df.columns:
        df["loan_grade_rank"] = df["loan_grade"].map(LOAN_GRADE_ORDER)

    predictors = [
        "person_age",
        "person_income",
        "emp_length",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "cb_person_cred_hist_length",
        "emp_length_missing",
        "int_rate_missing",
        "loan_grade_rank",
        "person_home_ownership",
        "loan_intent",
        "cb_person_default_on_file",
    ]
    predictors = [predictor for predictor in predictors if predictor in df.columns]

    data = df[predictors + ["loan_status"]].dropna()
    data = pd.get_dummies(data, columns=RAW_LOGIT_CATEGORICAL_FEATURES, drop_first=True)
    return data, predictors


def prepare_raw_logit_data(
    df: pd.DataFrame,
    target: str = "loan_status",
) -> Tuple[pd.DataFrame, pd.Series]:
    data, _ = _prepare_raw_logit_features(df)
    return data.drop(columns=[target]), data[target].astype(int)


def fit_raw_logit_model(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[pd.Series] = None,
) -> LogisticRegression:
    model = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X, y, sample_weight=sample_weight)
    return model


def select_uncorrelated_features(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = DEFAULT_PAIRWISE_CORRELATION_THRESHOLD,
) -> tuple[list[str], pd.DataFrame]:
    """Remove one feature from each highly correlated pair.

    When a pair exceeds the threshold, remove the member with the higher
    average absolute correlation to the rest of the design matrix. Target
    association is used only as a tie-breaker.
    """
    numeric_X = X.astype(float)
    target = y.astype(float).loc[X.index]
    feature_target_correlation = numeric_X.apply(lambda column: column.corr(target)).abs().fillna(0.0)
    feature_correlation = numeric_X.corr().abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feature_redundancy = (
        feature_correlation.mask(np.eye(feature_correlation.shape[0], dtype=bool))
        .mean()
        .fillna(0.0)
    )

    candidate_pairs = []
    columns = list(feature_correlation.columns)
    for left_position, left_feature in enumerate(columns):
        for right_feature in columns[left_position + 1 :]:
            correlation = float(feature_correlation.loc[left_feature, right_feature])
            if correlation >= threshold:
                candidate_pairs.append((left_feature, right_feature, correlation))

    removed_features: set[str] = set()
    rows = []
    for left_feature, right_feature, correlation in sorted(candidate_pairs, key=lambda row: row[2], reverse=True):
        if left_feature in removed_features or right_feature in removed_features:
            continue
        left_target_correlation = float(feature_target_correlation[left_feature])
        right_target_correlation = float(feature_target_correlation[right_feature])
        left_redundancy = float(feature_redundancy[left_feature])
        right_redundancy = float(feature_redundancy[right_feature])
        if (
            left_redundancy > right_redundancy
            or (
                np.isclose(left_redundancy, right_redundancy)
                and left_target_correlation < right_target_correlation
            )
        ):
            kept_feature = right_feature
            removed_feature = left_feature
            kept_target_correlation = right_target_correlation
            removed_target_correlation = left_target_correlation
            kept_redundancy = right_redundancy
            removed_redundancy = left_redundancy
        else:
            kept_feature = left_feature
            removed_feature = right_feature
            kept_target_correlation = left_target_correlation
            removed_target_correlation = right_target_correlation
            kept_redundancy = left_redundancy
            removed_redundancy = right_redundancy

        removed_features.add(removed_feature)
        rows.append(
            {
                "removed_feature": removed_feature,
                "kept_feature": kept_feature,
                "pairwise_abs_correlation": correlation,
                "removed_mean_abs_correlation": removed_redundancy,
                "kept_mean_abs_correlation": kept_redundancy,
                "removed_abs_target_correlation": removed_target_correlation,
                "kept_abs_target_correlation": kept_target_correlation,
                "threshold": threshold,
            }
        )

    selected_features = [feature for feature in X.columns if feature not in removed_features]
    return selected_features, pd.DataFrame(rows)


def logistic_model_aic(model: LogisticRegression, X: pd.DataFrame, y: pd.Series) -> float:
    eps = np.finfo(float).eps
    probabilities = np.clip(model.predict_proba(X)[:, 1], eps, 1 - eps)
    target = y.astype(float).to_numpy()
    log_likelihood = float(np.sum(target * np.log(probabilities) + (1 - target) * np.log(1 - probabilities)))
    design = np.column_stack([np.ones(X.shape[0]), X.to_numpy(dtype=float)])
    parameter_count = int(np.linalg.matrix_rank(design))
    return 2 * parameter_count - 2 * log_likelihood


def minimize_aic_features(
    X: pd.DataFrame,
    y: pd.Series,
    initial_features: Iterable[str],
    min_features: int = 1,
    tolerance: float = 1e-6,
) -> tuple[list[str], pd.DataFrame]:
    """Backward-eliminate raw-logit features while AIC improves."""
    current_features = list(initial_features)
    current_model = fit_raw_logit_model(X[current_features], y)
    current_aic = logistic_model_aic(current_model, X[current_features], y)
    rows = []
    step = 1

    while len(current_features) > min_features:
        candidate_rows = []
        for candidate_feature in current_features:
            trial_features = [feature for feature in current_features if feature != candidate_feature]
            trial_model = fit_raw_logit_model(X[trial_features], y)
            trial_aic = logistic_model_aic(trial_model, X[trial_features], y)
            candidate_rows.append((candidate_feature, trial_features, trial_aic))

        feature_to_remove, trial_features, trial_aic = min(candidate_rows, key=lambda row: row[2])
        if trial_aic < current_aic - tolerance:
            rows.append(
                {
                    "step": step,
                    "removed_feature": feature_to_remove,
                    "previous_aic": current_aic,
                    "new_aic": trial_aic,
                    "aic_improvement": current_aic - trial_aic,
                    "remaining_features": len(trial_features),
                }
            )
            current_features = trial_features
            current_aic = trial_aic
            step += 1
        else:
            break

    return current_features, pd.DataFrame(rows)


def build_raw_logit_model(
    df: pd.DataFrame,
    target: str = "loan_status",
    selected_features: Optional[Iterable[str]] = None,
) -> Tuple[LogisticRegression, List[str], pd.Series]:
    """Build the raw logistic-regression benchmark on numeric and dummy variables."""
    data, _ = _prepare_raw_logit_features(df)
    X = data.drop(columns=[target])
    if selected_features is not None:
        X = X[list(selected_features)]
    y = data[target].astype(int)

    model = fit_raw_logit_model(X, y)
    coefficients = pd.Series(model.coef_[0], index=X.columns)
    ordered_predictors = list(coefficients.abs().sort_values(ascending=False).index)
    return model, ordered_predictors, coefficients


def predictor_list_from_coefficients(coefficients: pd.Series, top_n: int = 20) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "coefficient": coefficients,
            "abs_coefficient": coefficients.abs(),
        }
    )
    return summary.sort_values("abs_coefficient", ascending=False).head(top_n)


def dummy_feature_groups(X: pd.DataFrame) -> dict[str, list[str]]:
    """Return one-hot encoded feature families for grouped LR tests."""
    groups = {}
    for feature in RAW_LOGIT_CATEGORICAL_FEATURES:
        prefix = f"{feature}_"
        columns = [column for column in X.columns if column.startswith(prefix)]
        if columns:
            groups[feature] = columns
    return groups
