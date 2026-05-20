from typing import Dict, Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _prediction_scores(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        score_range = scores.max() - scores.min()
        if score_range == 0:
            return np.full_like(scores, 0.5, dtype=float)
        return (scores - scores.min()) / score_range
    return model.predict(X)


def _log_loss_parts(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, float]:
    eps = np.finfo(float).eps
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_score, dtype=float), eps, 1 - eps)
    ll_model = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    base_rate = np.clip(y.mean(), eps, 1 - eps)
    ll_null = float(np.sum(y * np.log(base_rate) + (1 - y) * np.log(1 - base_rate)))
    return ll_model, ll_null


def _hosmer_lemeshow_test(y_true: pd.Series, y_score: np.ndarray, bins: int = 10) -> Dict[str, object]:
    data = pd.DataFrame({"y_true": y_true.to_numpy(), "y_score": y_score})
    data["group"] = pd.qcut(data["y_score"], bins, labels=False, duplicates="drop")
    grouped = (
        data.groupby("group", observed=True)
        .agg(
            count=("y_true", "count"),
            observed_defaults=("y_true", "sum"),
            expected_defaults=("y_score", "sum"),
            mean_score=("y_score", "mean"),
        )
        .reset_index(drop=True)
    )
    grouped["observed_non_defaults"] = grouped["count"] - grouped["observed_defaults"]
    grouped["expected_non_defaults"] = grouped["count"] - grouped["expected_defaults"]

    eps = np.finfo(float).eps
    statistic = float(
        (
            ((grouped["observed_defaults"] - grouped["expected_defaults"]) ** 2)
            / np.maximum(grouped["expected_defaults"], eps)
        ).sum()
        + (
            ((grouped["observed_non_defaults"] - grouped["expected_non_defaults"]) ** 2)
            / np.maximum(grouped["expected_non_defaults"], eps)
        ).sum()
    )
    degrees_of_freedom = max(int(grouped.shape[0] - 2), 1)
    p_value = float(stats.chi2.sf(statistic, degrees_of_freedom))
    return {
        "statistic": statistic,
        "p_value": p_value,
        "degrees_of_freedom": degrees_of_freedom,
        "groups": grouped,
    }


def _coefficient_significance(
    model,
    X: pd.DataFrame,
    y_score: np.ndarray,
    alpha: float,
) -> pd.DataFrame:
    if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
        return pd.DataFrame()

    design = np.column_stack([np.ones(X.shape[0]), X.to_numpy(dtype=float)])
    weights = np.clip(y_score * (1 - y_score), np.finfo(float).eps, None)
    fisher_information = design.T @ (design * weights[:, None])
    covariance = np.linalg.pinv(fisher_information)

    coefficients = np.concatenate([np.asarray(model.intercept_).ravel()[:1], np.asarray(model.coef_).ravel()])
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    z_scores = np.divide(
        coefficients,
        standard_errors,
        out=np.full_like(coefficients, np.nan, dtype=float),
        where=standard_errors > 0,
    )
    p_values = 2 * stats.norm.sf(np.abs(z_scores))
    confidence_delta = stats.norm.ppf(1 - alpha / 2) * standard_errors

    return pd.DataFrame(
        {
            "term": ["intercept", *X.columns],
            "coefficient": coefficients,
            "std_error": standard_errors,
            "z_score": z_scores,
            "p_value": p_values,
            "ci_lower": coefficients - confidence_delta,
            "ci_upper": coefficients + confidence_delta,
            "significant": p_values < alpha,
        }
    )


def _global_statistical_tests(
    y_true: pd.Series,
    y_score: np.ndarray,
    number_of_predictors: int,
) -> pd.DataFrame:
    ll_model, ll_null = _log_loss_parts(y_true, y_score)
    lr_statistic = max(2 * (ll_model - ll_null), 0.0)
    lr_p_value = float(stats.chi2.sf(lr_statistic, max(number_of_predictors, 1)))

    defaults = y_score[np.asarray(y_true) == 1]
    non_defaults = y_score[np.asarray(y_true) == 0]
    if len(defaults) and len(non_defaults):
        ks_result = stats.ks_2samp(defaults, non_defaults)
        mw_result = stats.mannwhitneyu(defaults, non_defaults, alternative="two-sided")
        ks_statistic = float(ks_result.statistic)
        ks_p_value = float(ks_result.pvalue)
        mann_whitney_statistic = float(mw_result.statistic)
        mann_whitney_p_value = float(mw_result.pvalue)
    else:
        ks_statistic = np.nan
        ks_p_value = np.nan
        mann_whitney_statistic = np.nan
        mann_whitney_p_value = np.nan

    return pd.DataFrame(
        [
            {
                "test": "Likelihood-ratio vs intercept-only",
                "statistic": lr_statistic,
                "p_value": lr_p_value,
                "comment": "Tests whether the fitted indicators improve log-likelihood over a null model.",
            },
            {
                "test": "Kolmogorov-Smirnov score separation",
                "statistic": ks_statistic,
                "p_value": ks_p_value,
                "comment": "Tests whether default and non-default score distributions differ.",
            },
            {
                "test": "Mann-Whitney score ranking",
                "statistic": mann_whitney_statistic,
                "p_value": mann_whitney_p_value,
                "comment": "Tests whether defaults tend to receive higher scores than non-defaults.",
            },
        ]
    )


def _model_fit_statistics(y_true: pd.Series, y_score: np.ndarray, number_of_parameters: int) -> Dict[str, float]:
    ll_model, ll_null = _log_loss_parts(y_true, y_score)
    observations = len(y_true)
    return {
        "log_likelihood": ll_model,
        "null_log_likelihood": ll_null,
        "aic": 2 * number_of_parameters - 2 * ll_model,
        "bic": np.log(observations) * number_of_parameters - 2 * ll_model,
        "mcfadden_r2": 1 - (ll_model / ll_null) if ll_null else np.nan,
    }


def evaluate_model_results(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
    alpha: float = 0.05,
    model_name: str = "Full model",
) -> Dict[str, object]:
    y_score = _prediction_scores(model, X)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    fpr, tpr, roc_thresholds = roc_curve(y, y_score)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y, y_score)
    ks = float(np.max(np.abs(tpr - fpr))) if len(fpr) and len(tpr) else 0.0
    optimal_threshold = float(roc_thresholds[np.argmax(np.abs(tpr - fpr))]) if len(roc_thresholds) else threshold

    score_data = pd.DataFrame(
        {
            "y_true": y.to_numpy(),
            "y_score": y_score,
            "y_pred": y_pred,
        },
        index=y.index,
    )
    score_data["decile"] = pd.qcut(score_data["y_score"], 10, labels=False, duplicates="drop")

    calibration = (
        score_data.groupby("decile", observed=True)
        .agg(
            mean_pred=("y_score", "mean"),
            actual_rate=("y_true", "mean"),
            count=("y_true", "count"),
        )
        .reset_index()
    )

    lift = score_data.sort_values("y_score", ascending=False).reset_index(drop=True)
    lift["rank"] = lift.index + 1
    lift["rank_pct"] = lift["rank"] / lift.shape[0]
    lift_summary = (
        lift.assign(decile=lambda df: pd.qcut(df["rank_pct"], 10, labels=False, duplicates="drop"))
        .groupby("decile", observed=True)
        .agg(
            count=("y_true", "count"),
            defaults=("y_true", "sum"),
            mean_score=("y_score", "mean"),
        )
        .reset_index()
    )
    lift_summary["default_rate"] = lift_summary["defaults"] / lift_summary["count"]
    lift_summary["cumulative_defaults"] = lift_summary["defaults"].cumsum()
    lift_summary["cumulative_capture_rate"] = lift_summary["cumulative_defaults"] / lift_summary["defaults"].sum()

    coefficient_tests = _coefficient_significance(model, X, y_score, alpha)
    non_significant_predictors = []
    if not coefficient_tests.empty:
        non_significant_predictors = coefficient_tests.loc[
            (coefficient_tests["term"] != "intercept") & (~coefficient_tests["significant"]),
            "term",
        ].tolist()

    hosmer_lemeshow = _hosmer_lemeshow_test(y, y_score)
    global_tests = _global_statistical_tests(y, y_score, X.shape[1])
    global_tests = pd.concat(
        [
            global_tests,
            pd.DataFrame(
                [
                    {
                        "test": "Hosmer-Lemeshow calibration",
                        "statistic": hosmer_lemeshow["statistic"],
                        "p_value": hosmer_lemeshow["p_value"],
                        "comment": "Tests grouped calibration; small p-values suggest calibration mismatch.",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    fit_statistics = _model_fit_statistics(y, y_score, X.shape[1] + 1)
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": f1_score(y, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp > 0 else 0.0,
        "roc_auc": roc_auc_score(y, y_score),
        "average_precision": average_precision_score(y, y_score),
        "brier_score": brier_score_loss(y, y_score),
        "ks_statistic": ks,
        "threshold": threshold,
        "optimal_ks_threshold": optimal_threshold,
        **fit_statistics,
    }

    return {
        "model_name": model_name,
        "feature_names": list(X.columns),
        "metrics": metrics,
        "confusion_matrix": pd.DataFrame(
            [[tn, fp], [fn, tp]], columns=["Pred 0", "Pred 1"], index=["Actual 0", "Actual 1"]
        ),
        "roc_curve": pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}),
        "pr_curve": pd.DataFrame({"precision": precision_curve, "recall": recall_curve}).assign(
            threshold=pd.Series(list(pr_thresholds) + [np.nan])
        ),
        "calibration": calibration,
        "lift": lift_summary,
        "scores": score_data,
        "coefficient_significance": coefficient_tests,
        "global_tests": global_tests,
        "hosmer_lemeshow_groups": hosmer_lemeshow["groups"],
        "non_significant_predictors": non_significant_predictors,
    }


def compare_evaluations(evaluations: Iterable[Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for evaluation in evaluations:
        metrics = evaluation["metrics"]
        rows.append(
            {
                "model": evaluation["model_name"],
                "features": len(evaluation["feature_names"]),
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "brier_score": metrics["brier_score"],
                "ks_statistic": metrics["ks_statistic"],
                "aic": metrics["aic"],
                "bic": metrics["bic"],
                "mcfadden_r2": metrics["mcfadden_r2"],
            }
        )
    return pd.DataFrame(rows)
