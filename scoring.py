import warnings
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


SCORING_CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]


def _prepare_scoring_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["emp_length_missing"] = df["person_emp_length"].isna().astype(int)
    df["int_rate_missing"] = df["loan_int_rate"].isna().astype(int)
    df["emp_length"] = df["person_emp_length"].fillna(0)

    grade_median_int_rate = df.groupby("loan_grade")["loan_int_rate"].median()
    df["loan_int_rate"] = df.apply(
        lambda row: grade_median_int_rate[row["loan_grade"]]
        if pd.isna(row["loan_int_rate"]) else row["loan_int_rate"],
        axis=1,
    )

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
        "person_home_ownership",
        "loan_intent",
        "loan_grade",
        "cb_person_default_on_file",
    ]

    df = df[predictors + ["loan_status"]].dropna()

    df = pd.get_dummies(df, columns=SCORING_CATEGORICAL_FEATURES, drop_first=True)
    return df, predictors


def prepare_scoring_data(
    df: pd.DataFrame,
    target: str = "loan_status",
) -> Tuple[pd.DataFrame, pd.Series]:
    data, _ = _prepare_scoring_features(df)
    return data.drop(columns=[target]), data[target].astype(int)


def fit_logit_scoring_model(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[pd.Series] = None,
) -> LogisticRegression:
    model = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X, y, sample_weight=sample_weight)
    return model


def build_logit_scoring_model(
    df: pd.DataFrame,
    target: str = "loan_status",
    selected_features: Optional[Iterable[str]] = None,
) -> Tuple[LogisticRegression, List[str], pd.Series]:
    """Build a credit scoring model using logistic regression (logit link)."""
    data, _ = _prepare_scoring_features(df)
    X = data.drop(columns=[target])
    if selected_features is not None:
        X = X[list(selected_features)]
    y = data[target].astype(int)

    model = fit_logit_scoring_model(X, y)
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
    for feature in SCORING_CATEGORICAL_FEATURES:
        prefix = f"{feature}_"
        columns = [column for column in X.columns if column.startswith(prefix)]
        if columns:
            groups[feature] = columns
    return groups
