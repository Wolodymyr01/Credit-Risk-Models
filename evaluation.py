import os
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        return (scores - scores.min()) / (scores.max() - scores.min())
    return model.predict(X)


def evaluate_model_results(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, object]:
    y_score = _prediction_scores(model, X)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if tn + fp > 0 else 0.0
    roc_auc = roc_auc_score(y, y_score)
    avg_precision = average_precision_score(y, y_score)
    brier = brier_score_loss(y, y_score)
    fpr, tpr, roc_thresholds = roc_curve(y, y_score)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y, y_score)
    ks = float(np.max(np.abs(tpr - fpr))) if len(fpr) and len(tpr) else 0.0
    optimal_threshold = float(roc_thresholds[np.argmax(np.abs(tpr - fpr))]) if len(roc_thresholds) else threshold

    score_data = pd.DataFrame(
        {
            "y_true": y,
            "y_score": y_score,
            "y_pred": y_pred,
        }
    )
    score_data["decile"] = pd.qcut(score_data["y_score"], 10, labels=False, duplicates="drop")

    calibration = (
        score_data.groupby("decile")
        .agg(
            mean_pred=("y_score", "mean"),
            actual_rate=("y_true", "mean"),
            count=("y_true", "count"),
        )
        .reset_index()
    )

    lift = (
        score_data.sort_values("y_score", ascending=False)
        .reset_index(drop=True)
        .assign(rank=lambda df: df.index + 1)
    )
    lift["rank_pct"] = lift["rank"] / lift.shape[0]
    lift_summary = (
        lift.assign(decile=lambda df: pd.qcut(df["rank_pct"], 10, labels=False, duplicates="drop"))
        .groupby("decile")
        .agg(
            count=("y_true", "count"),
            defaults=("y_true", "sum"),
            mean_score=("y_score", "mean"),
        )
        .reset_index()
    )
    lift_summary["default_rate"] = lift_summary["defaults"] / lift_summary["count"]

    evaluation = {
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": specificity,
            "roc_auc": roc_auc,
            "average_precision": avg_precision,
            "brier_score": brier,
            "ks_statistic": ks,
            "threshold": threshold,
            "optimal_ks_threshold": optimal_threshold,
        },
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
    }
    return evaluation


def _plot_line_curve(x, y, title, xlabel, ylabel, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, marker=".", linestyle="-", color="#2c7fb8")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _plot_calibration(calibration: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(calibration["mean_pred"], calibration["actual_rate"], marker="o", linestyle="-", color="#d95f02")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#666666")
    ax.set_title("Calibration Curve")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed default rate")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _plot_score_distribution(scores: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    positives = scores[scores["y_true"] == 1]["y_score"]
    negatives = scores[scores["y_true"] == 0]["y_score"]
    ax.hist(negatives, bins=20, alpha=0.6, label="Non-default", color="#4c72b0", edgecolor="black")
    ax.hist(positives, bins=20, alpha=0.6, label="Default", color="#dd8452", edgecolor="black")
    ax.set_title("Predicted Score Distribution")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def create_html_evaluation_report(
    evaluation: Dict[str, object],
    save_dir: str = "evaluation",
    report_name: str = "evaluation_report.html",
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    image_paths = []

    roc_path = os.path.join(save_dir, "roc_curve.png")
    _plot_line_curve(
        evaluation["roc_curve"]["fpr"],
        evaluation["roc_curve"]["tpr"],
        "ROC Curve",
        "False positive rate",
        "True positive rate",
        roc_path,
    )
    image_paths.append(roc_path)

    pr_path = os.path.join(save_dir, "pr_curve.png")
    _plot_line_curve(
        evaluation["pr_curve"]["recall"],
        evaluation["pr_curve"]["precision"],
        "Precision-Recall Curve",
        "Recall",
        "Precision",
        pr_path,
    )
    image_paths.append(pr_path)

    calib_path = os.path.join(save_dir, "calibration_curve.png")
    _plot_calibration(evaluation["calibration"], calib_path)
    image_paths.append(calib_path)

    score_dist_path = os.path.join(save_dir, "score_distribution.png")
    _plot_score_distribution(evaluation["scores"], score_dist_path)
    image_paths.append(score_dist_path)

    report_path = os.path.join(save_dir, report_name)

    summary = pd.DataFrame(
        evaluation["metrics"].items(), columns=["metric", "value"]
    )
    summary_html = summary.to_html(index=False, classes="metrics-table", border=0)
    confusion_html = evaluation["confusion_matrix"].to_html(classes="confusion-table", border=0)
    calibration_html = evaluation["calibration"].to_html(index=False, classes="calibration-table", border=0)
    lift_html = evaluation["lift"].to_html(index=False, classes="lift-table", border=0)

    html = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<title>Model Evaluation Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 24px; color: #222; }",
        "h1, h2 { color: #333; }",
        "section { margin-bottom: 32px; }",
        "table { border-collapse: collapse; width: 100%; margin-top: 12px; }",
        "table th, table td { border: 1px solid #ccc; padding: 8px; text-align: left; vertical-align: top; }",
        "table th { background: #f5f5f5; }",
        "img { max-width: 100%; height: auto; border: 1px solid #ddd; margin-bottom: 12px; }",
        "figure { margin: 0 0 24px 0; }",
        "figcaption { color: #555; font-size: 0.9em; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Model Evaluation Report</h1>",
        summary_html,
        "<section><h2>Confusion Matrix</h2>",
        confusion_html,
        "</section>",
        "<section><h2>Calibration Summary</h2>",
        calibration_html,
        "</section>",
        "<section><h2>Lift Summary</h2>",
        lift_html,
        "</section>",
    ]

    for image_path in image_paths:
        html.extend([
            "<section>",
            f"<h2>{os.path.basename(image_path).replace('_', ' ').replace('.png','').title()}</h2>",
            f"<img src=\"{os.path.basename(image_path)}\" alt=\"{os.path.basename(image_path)}\">",
            "</section>",
        ])

    html.extend(["</body>", "</html>"])
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    return report_path
