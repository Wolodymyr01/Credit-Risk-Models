from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProfitLossConfig:
    """Normalized payoff matrix for accept/reject decisions.

    y=1 means default. Predicted 1 means the model flags the application as high risk.
    """

    true_negative: float = 0.20
    false_positive: float = -0.20
    false_negative: float = -1.00
    true_positive: float = 0.00


ThresholdStrategy = Callable[[pd.Series, np.ndarray, ProfitLossConfig], dict[str, float | str]]


def candidate_thresholds(y_score: np.ndarray) -> np.ndarray:
    scores = np.unique(np.asarray(y_score, dtype=float))
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return np.array([0.5])
    return np.unique(np.concatenate(([0.0], scores, [1.0])))


def threshold_confusion_table(y_true: pd.Series, y_score: np.ndarray) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    frame = pd.DataFrame({"threshold": scores, "y_true": y})
    grouped = (
        frame.groupby("threshold", observed=True)
        .agg(positives=("y_true", "sum"), total=("y_true", "count"))
        .reset_index()
        .sort_values("threshold", ascending=False)
    )
    grouped["negatives"] = grouped["total"] - grouped["positives"]
    grouped["tp"] = grouped["positives"].cumsum().astype(int)
    grouped["fp"] = grouped["negatives"].cumsum().astype(int)

    total_positives = int(y.sum())
    total_negatives = int(len(y) - total_positives)
    grouped["fn"] = total_positives - grouped["tp"]
    grouped["tn"] = total_negatives - grouped["fp"]

    extra_rows = []
    if not grouped["threshold"].eq(1.0).any():
        positives_at_one = int(((scores >= 1.0) & (y == 1)).sum())
        negatives_at_one = int(((scores >= 1.0) & (y == 0)).sum())
        extra_rows.append(
            {
                "threshold": 1.0,
                "positives": 0,
                "total": 0,
                "negatives": 0,
                "tp": positives_at_one,
                "fp": negatives_at_one,
                "fn": total_positives - positives_at_one,
                "tn": total_negatives - negatives_at_one,
            }
        )
    if not grouped["threshold"].eq(0.0).any():
        extra_rows.append(
            {
                "threshold": 0.0,
                "positives": 0,
                "total": 0,
                "negatives": 0,
                "tp": total_positives,
                "fp": total_negatives,
                "fn": 0,
                "tn": 0,
            }
        )
    if extra_rows:
        grouped = pd.concat([pd.DataFrame(extra_rows), grouped], ignore_index=True)

    return grouped.drop_duplicates("threshold").sort_values("threshold", ascending=False).reset_index(drop=True)


def confusion_counts_at_threshold(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, int]:
    y = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    return {
        "tn": int(((y == 0) & (y_pred == 0)).sum()),
        "fp": int(((y == 0) & (y_pred == 1)).sum()),
        "fn": int(((y == 1) & (y_pred == 0)).sum()),
        "tp": int(((y == 1) & (y_pred == 1)).sum()),
    }


def profit_at_threshold(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
    profit_loss: ProfitLossConfig,
) -> float:
    counts = confusion_counts_at_threshold(y_true, y_score, threshold)
    return float(
        counts["tn"] * profit_loss.true_negative
        + counts["fp"] * profit_loss.false_positive
        + counts["fn"] * profit_loss.false_negative
        + counts["tp"] * profit_loss.true_positive
    )


def threshold_youden_j(
    y_true: pd.Series,
    y_score: np.ndarray,
    profit_loss: ProfitLossConfig,
) -> dict[str, float | str]:
    del profit_loss
    table = threshold_confusion_table(y_true, y_score)
    sensitivity = table["tp"] / (table["tp"] + table["fn"]).replace(0, np.nan)
    specificity = table["tn"] / (table["tn"] + table["fp"]).replace(0, np.nan)
    table["youden_j"] = (sensitivity + specificity - 1).fillna(0.0)
    row = table.sort_values(["youden_j", "threshold"], ascending=[False, False]).iloc[0]
    best = {"threshold": float(row["threshold"]), "objective_value": float(row["youden_j"])}
    return {"strategy": "youden_j", **best}


def threshold_f1(
    y_true: pd.Series,
    y_score: np.ndarray,
    profit_loss: ProfitLossConfig,
) -> dict[str, float | str]:
    del profit_loss
    table = threshold_confusion_table(y_true, y_score)
    denominator = 2 * table["tp"] + table["fp"] + table["fn"]
    table["f1"] = (2 * table["tp"] / denominator.replace(0, np.nan)).fillna(0.0)
    row = table.sort_values(["f1", "threshold"], ascending=[False, False]).iloc[0]
    best = {"threshold": float(row["threshold"]), "objective_value": float(row["f1"])}
    return {"strategy": "f1", **best}


def threshold_expected_profit_loss(
    y_true: pd.Series,
    y_score: np.ndarray,
    profit_loss: ProfitLossConfig,
) -> dict[str, float | str]:
    table = threshold_confusion_table(y_true, y_score)
    table["profit"] = (
        table["tn"] * profit_loss.true_negative
        + table["fp"] * profit_loss.false_positive
        + table["fn"] * profit_loss.false_negative
        + table["tp"] * profit_loss.true_positive
    )
    row = table.sort_values(["profit", "threshold"], ascending=[False, False]).iloc[0]
    best = {"threshold": float(row["threshold"]), "objective_value": float(row["profit"])}
    return {"strategy": "expected_profit_loss", **best}


THRESHOLD_STRATEGIES: dict[str, ThresholdStrategy] = {
    "youden_j": threshold_youden_j,
    "f1": threshold_f1,
    "expected_profit_loss": threshold_expected_profit_loss,
}


def resolve_threshold_strategy(strategy: str | ThresholdStrategy) -> ThresholdStrategy:
    if callable(strategy):
        return strategy
    try:
        return THRESHOLD_STRATEGIES[strategy]
    except KeyError as exc:
        known = ", ".join(sorted(THRESHOLD_STRATEGIES))
        raise ValueError(f"Unknown threshold strategy '{strategy}'. Expected one of: {known}.") from exc


def compare_threshold_strategies(
    y_true: pd.Series,
    y_score: np.ndarray,
    strategies: dict[str, ThresholdStrategy] | None = None,
    profit_loss: ProfitLossConfig | None = None,
) -> pd.DataFrame:
    profit_loss = profit_loss or ProfitLossConfig()
    strategies = strategies or THRESHOLD_STRATEGIES
    rows = []
    for strategy_name, strategy in strategies.items():
        result = strategy(y_true, y_score, profit_loss)
        threshold = float(result["threshold"])
        counts = confusion_counts_at_threshold(y_true, y_score, threshold)
        total = max(sum(counts.values()), 1)
        profit = profit_at_threshold(y_true, y_score, threshold, profit_loss)
        precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
        recall = counts["tp"] / (counts["tp"] + counts["fn"]) if counts["tp"] + counts["fn"] else 0.0
        specificity = counts["tn"] / (counts["tn"] + counts["fp"]) if counts["tn"] + counts["fp"] else 0.0
        rows.append(
            {
                "strategy": strategy_name,
                "threshold": threshold,
                "objective_value": float(result["objective_value"]),
                "expected_profit": profit,
                "expected_profit_per_case": profit / total,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1_score": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
                **counts,
            }
        )
    return pd.DataFrame(rows)


def recommend_threshold_strategy(threshold_comparison: pd.DataFrame) -> str:
    """Pick a default strategy by comparing decision metrics.

    Expected profit/loss is preferred when it gives the best payoff while staying close
    to the best F1 score. Otherwise, fall back to Youden J as the balanced threshold.
    """
    if threshold_comparison.empty:
        return "youden_j"
    strategies = set(threshold_comparison["strategy"])
    if "expected_profit_loss" in strategies:
        profit_row = threshold_comparison.loc[
            threshold_comparison["strategy"].eq("expected_profit_loss")
        ].iloc[0]
        best_profit = threshold_comparison["expected_profit_per_case"].max()
        best_f1 = threshold_comparison["f1_score"].max()
        if (
            profit_row["expected_profit_per_case"] >= best_profit - 1e-12
            and profit_row["f1_score"] >= 0.98 * best_f1
        ):
            return "expected_profit_loss"
    return "youden_j" if "youden_j" in strategies else str(threshold_comparison.iloc[0]["strategy"])
