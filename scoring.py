import warnings
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


SCORECARD_NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

SCORECARD_CATEGORICAL_FEATURES = [
    "emp_length_missing",
    "int_rate_missing",
    "person_home_ownership",
    "loan_intent",
    "cb_person_default_on_file",
]

MISSING_BIN = "__MISSING__"
UNKNOWN_WOE = 0.0
DEFAULT_WOE_MERGE_DISTANCE = 0.20
DEFAULT_MAX_NUMERIC_DIRECTION_CHANGES = 1


@dataclass
class FeatureBinning:
    name: str
    kind: str
    edges: Optional[np.ndarray]
    woe_by_bin: dict[object, float]
    unknown_woe: float = UNKNOWN_WOE


def prepare_scorecard_frame(
    df: pd.DataFrame,
    selected_sources: Optional[Iterable[str]] = None,
    target: str = "loan_status",
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare unencoded source variables used by the WoE scorecard transformer."""
    data = df.copy()
    if "emp_length_missing" not in data.columns:
        data["emp_length_missing"] = data["person_emp_length"].isna().astype(int)
    if "int_rate_missing" not in data.columns:
        data["int_rate_missing"] = data["loan_int_rate"].isna().astype(int)
    if "emp_length" not in data.columns:
        data["emp_length"] = data["person_emp_length"].fillna(0)

    if data["loan_int_rate"].isna().any():
        data["loan_int_rate"] = data["loan_int_rate"].fillna(data["loan_int_rate"].median())

    source_features = SCORECARD_NUMERIC_FEATURES + SCORECARD_CATEGORICAL_FEATURES
    if selected_sources is not None:
        selected_set = set(selected_sources)
        source_features = [feature for feature in source_features if feature in selected_set]

    data = data[source_features + [target]].dropna(subset=[target])
    return data[source_features], data[target].astype(int)


def scorecard_sources_from_raw_features(raw_features: Iterable[str]) -> list[str]:
    """Map an adjusted raw-logit feature list back to scorecard source variables."""
    raw_features = set(raw_features)
    sources = []
    for feature in SCORECARD_NUMERIC_FEATURES + SCORECARD_CATEGORICAL_FEATURES:
        if feature in raw_features or any(column.startswith(f"{feature}_") for column in raw_features):
            sources.append(feature)
    return sources


class WoeScorecardTransformer:
    """Fit reusable WoE transformations with quasi-continuous numeric classing."""

    def __init__(
        self,
        numeric_features: Optional[Iterable[str]] = None,
        categorical_features: Optional[Iterable[str]] = None,
        max_numeric_bins: int = 20,
        min_bins: int = 2,
        smoothing: float = 0.5,
        merge_woe_distance: float = DEFAULT_WOE_MERGE_DISTANCE,
        max_numeric_direction_changes: int = DEFAULT_MAX_NUMERIC_DIRECTION_CHANGES,
    ) -> None:
        self.numeric_features = list(numeric_features or SCORECARD_NUMERIC_FEATURES)
        self.categorical_features = list(categorical_features or SCORECARD_CATEGORICAL_FEATURES)
        self.max_numeric_bins = max_numeric_bins
        self.min_bins = min_bins
        self.smoothing = smoothing
        self.merge_woe_distance = merge_woe_distance
        self.max_numeric_direction_changes = max_numeric_direction_changes
        self.binnings_: dict[str, FeatureBinning] = {}
        self.summary_: pd.DataFrame = pd.DataFrame()

    @property
    def feature_names_out_(self) -> list[str]:
        return [f"{feature}_woe" for feature in self.binnings_]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        y = y.astype(int).loc[X.index]
        summaries = []
        self.binnings_ = {}

        for feature in self.numeric_features:
            if feature not in X.columns:
                continue
            labels, edges, fine_bin_count = self._fit_numeric_labels(X[feature])
            if edges is not None:
                edges = self._merge_numeric_edges(X[feature], y, edges)
                edges = self._merge_numeric_edges_for_order(X[feature], y, edges)
                labels = self._apply_numeric_labels(X[feature], edges)
            binning, summary = self._fit_woe_for_labels(feature, "numeric", labels, y, edges)
            coarse_bin_count = int((summary["bin"] != MISSING_BIN).sum())
            summary["fine_bin_count"] = fine_bin_count
            summary["coarse_bin_count"] = coarse_bin_count
            summary["bins_merged"] = max(fine_bin_count - coarse_bin_count, 0)
            self.binnings_[feature] = binning
            summaries.append(summary)

        for feature in self.categorical_features:
            if feature not in X.columns:
                continue
            labels = self._categorical_labels(X[feature])
            binning, summary = self._fit_woe_for_labels(feature, "categorical", labels, y, None)
            summary["fine_bin_count"] = summary.shape[0]
            summary["coarse_bin_count"] = summary.shape[0]
            summary["bins_merged"] = 0
            self.binnings_[feature] = binning
            summaries.append(summary)

        self.summary_ = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = pd.DataFrame(index=X.index)
        for feature, binning in self.binnings_.items():
            if binning.kind == "numeric":
                labels = self._apply_numeric_labels(X[feature], binning.edges)
            else:
                labels = self._categorical_labels(X[feature])
            transformed[f"{feature}_woe"] = labels.map(binning.woe_by_bin).fillna(binning.unknown_woe).astype(float)
        return transformed

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _fit_numeric_labels(self, values: pd.Series) -> tuple[pd.Series, Optional[np.ndarray], int]:
        regular = values.dropna().astype(float)
        if regular.nunique() < self.min_bins:
            labels = values.map(lambda value: MISSING_BIN if pd.isna(value) else str(value))
            return labels, None, int(regular.nunique())

        bin_count = min(self.max_numeric_bins, int(regular.nunique()))
        _, edges = pd.qcut(regular, q=bin_count, retbins=True, duplicates="drop")
        edges = np.unique(edges)
        if len(edges) <= self.min_bins:
            labels = values.map(lambda value: MISSING_BIN if pd.isna(value) else str(value))
            return labels, None, int(regular.nunique())

        edges[0] = -np.inf
        edges[-1] = np.inf
        labels = self._apply_numeric_labels(values, edges)
        return labels, edges, len(edges) - 1

    def _merge_numeric_edges(self, values: pd.Series, y: pd.Series, edges: np.ndarray) -> np.ndarray:
        coarse_edges = edges.copy()
        while len(coarse_edges) - 1 > self.min_bins:
            labels = self._apply_numeric_labels(values, coarse_edges)
            _, summary = self._fit_woe_for_labels("__merge_probe__", "numeric", labels, y, coarse_edges)
            regular = summary[summary["bin"] != MISSING_BIN].reset_index(drop=True)
            distances = regular["woe"].diff().abs().iloc[1:].to_numpy()
            if len(distances) == 0:
                break
            weakest_position = int(np.nanargmin(distances))
            weakest_distance = float(distances[weakest_position])
            if weakest_distance >= self.merge_woe_distance:
                break
            boundary_to_remove = weakest_position + 1
            coarse_edges = np.delete(coarse_edges, boundary_to_remove)
        return coarse_edges

    def _merge_numeric_edges_for_order(self, values: pd.Series, y: pd.Series, edges: np.ndarray) -> np.ndarray:
        ordered_edges = edges.copy()
        while len(ordered_edges) - 1 > self.min_bins:
            labels = self._apply_numeric_labels(values, ordered_edges)
            _, summary = self._fit_woe_for_labels("__order_probe__", "numeric", labels, y, ordered_edges)
            regular = summary[summary["bin"] != MISSING_BIN].reset_index(drop=True)
            if self._direction_change_count(regular["default_rate"]) <= self.max_numeric_direction_changes:
                break
            distances = regular["woe"].diff().abs().iloc[1:].to_numpy()
            if len(distances) == 0:
                break
            weakest_position = int(np.nanargmin(distances))
            boundary_to_remove = weakest_position + 1
            ordered_edges = np.delete(ordered_edges, boundary_to_remove)
        return ordered_edges

    def _apply_numeric_labels(self, values: pd.Series, edges: Optional[np.ndarray]) -> pd.Series:
        if edges is None:
            return values.map(lambda value: MISSING_BIN if pd.isna(value) else str(value))
        codes = pd.cut(values.astype(float), bins=edges, include_lowest=True, labels=False)
        labels = pd.Series(index=values.index, dtype=object)
        for position in range(len(edges) - 1):
            labels[codes == position] = self._numeric_bin_label(edges, position)
        labels[pd.isna(values)] = MISSING_BIN
        return labels.astype(str)

    def _categorical_labels(self, values: pd.Series) -> pd.Series:
        return values.astype(object).where(values.notna(), MISSING_BIN).astype(str)

    def _fit_woe_for_labels(
        self,
        feature: str,
        kind: str,
        labels: pd.Series,
        y: pd.Series,
        edges: Optional[np.ndarray],
    ) -> tuple[FeatureBinning, pd.DataFrame]:
        frame = pd.DataFrame({"bin": labels.astype(str), "target": y})
        grouped = (
            frame.groupby("bin", observed=True)
            .agg(count=("target", "count"), bads=("target", "sum"))
            .reset_index()
        )
        grouped = self._sort_summary_bins(grouped, kind, edges)
        grouped["goods"] = grouped["count"] - grouped["bads"]

        total_goods = grouped["goods"].sum()
        total_bads = grouped["bads"].sum()
        bin_count = max(grouped.shape[0], 1)
        grouped["good_distribution"] = (grouped["goods"] + self.smoothing) / (
            total_goods + self.smoothing * bin_count
        )
        grouped["bad_distribution"] = (grouped["bads"] + self.smoothing) / (
            total_bads + self.smoothing * bin_count
        )
        grouped["woe"] = np.log(grouped["good_distribution"] / grouped["bad_distribution"])
        grouped["iv_component"] = (
            grouped["good_distribution"] - grouped["bad_distribution"]
        ) * grouped["woe"]
        grouped["default_rate"] = grouped["bads"] / grouped["count"]
        grouped["sample_share"] = grouped["count"] / grouped["count"].sum()
        grouped.insert(0, "feature", feature)
        grouped.insert(1, "type", kind)
        grouped["information_value"] = grouped["iv_component"].sum()
        grouped["direction_changes"] = self._direction_change_count(grouped["default_rate"]) if kind == "numeric" else np.nan
        grouped["ordered_feature"] = kind == "numeric"
        grouped["ordering_assessment"] = self._ordering_assessment(grouped, kind)
        grouped["monotonicity_note"] = grouped["ordering_assessment"]
        if kind == "categorical":
            grouped = grouped.sort_values("woe").reset_index(drop=True)
            grouped = self._add_categorical_grouping_proposals(grouped)
        else:
            grouped["proposed_group_id"] = ""
            grouped["proposed_group_members"] = ""
            grouped["proposed_group_default_rate"] = np.nan
            grouped["proposed_group_woe"] = np.nan
            grouped["grouping_rationale"] = ""
        grouped = self._add_woe_distance_columns(grouped)

        woe_by_bin = dict(zip(grouped["bin"], grouped["woe"]))
        binning = FeatureBinning(
            name=feature,
            kind=kind,
            edges=edges,
            woe_by_bin=woe_by_bin,
        )
        return binning, grouped

    def _sort_summary_bins(
        self,
        grouped: pd.DataFrame,
        kind: str,
        edges: Optional[np.ndarray],
    ) -> pd.DataFrame:
        if kind != "numeric" or edges is None:
            return grouped.sort_values("count", ascending=False).reset_index(drop=True)

        order = {
            self._numeric_bin_label(edges, position): position
            for position in range(len(edges) - 1)
        }
        grouped = grouped.copy()
        grouped["_order"] = grouped["bin"].map(order).fillna(len(order))
        return grouped.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    def _numeric_bin_label(self, edges: np.ndarray, position: int) -> str:
        lower = self._format_edge(edges[position])
        upper = self._format_edge(edges[position + 1])
        return f"{position + 1:02d}: ({lower}, {upper}]"

    def _format_edge(self, edge: float) -> str:
        if np.isneginf(edge):
            return "-inf"
        if np.isposinf(edge):
            return "inf"
        return f"{edge:.6g}"

    def _add_woe_distance_columns(self, grouped: pd.DataFrame) -> pd.DataFrame:
        grouped = grouped.copy()
        grouped["next_bin"] = grouped["bin"].shift(-1)
        grouped["woe_distance_to_next_bin"] = (grouped["woe"] - grouped["woe"].shift(-1)).abs()
        grouped["woe_distance_band"] = grouped["woe_distance_to_next_bin"].map(self._woe_distance_band)
        grouped["woe_distance_action"] = grouped["woe_distance_to_next_bin"].map(self._woe_distance_action)
        return grouped

    def _add_categorical_grouping_proposals(self, grouped: pd.DataFrame) -> pd.DataFrame:
        grouped = grouped.copy().reset_index(drop=True)
        group_ids = []
        current_group = 1
        for row_index, row in grouped.iterrows():
            if row_index > 0:
                previous_woe = grouped.loc[row_index - 1, "woe"]
                if abs(row["woe"] - previous_woe) >= self.merge_woe_distance:
                    current_group += 1
            group_ids.append(current_group)

        grouped["proposed_group_id"] = [f"G{group_id}" for group_id in group_ids]
        grouped["proposed_group_members"] = ""
        grouped["proposed_group_default_rate"] = np.nan
        grouped["proposed_group_woe"] = np.nan
        grouped["grouping_rationale"] = ""

        for group_id, group in grouped.groupby("proposed_group_id", sort=False):
            members = " + ".join(group["bin"].astype(str))
            total_count = group["count"].sum()
            default_rate = group["bads"].sum() / total_count if total_count else np.nan
            weighted_woe = np.average(group["woe"], weights=group["count"]) if total_count else np.nan
            rationale = "single distinct category"
            if group.shape[0] > 1:
                rationale = f"grouped by WoE distance < {self.merge_woe_distance:.2f}"
            mask = grouped["proposed_group_id"].eq(group_id)
            grouped.loc[mask, "proposed_group_members"] = members
            grouped.loc[mask, "proposed_group_default_rate"] = default_rate
            grouped.loc[mask, "proposed_group_woe"] = weighted_woe
            grouped.loc[mask, "grouping_rationale"] = rationale
        return grouped

    def _woe_distance_band(self, distance: float) -> str:
        if pd.isna(distance):
            return "last bin"
        if distance < 0.05:
            return "< 0.05"
        if distance < 0.10:
            return "0.05-0.10"
        if distance < 0.20:
            return "0.10-0.20"
        if distance < 0.30:
            return "0.20-0.30"
        return "> 0.30"

    def _woe_distance_action(self, distance: float) -> str:
        if pd.isna(distance):
            return "no next bin"
        if distance < 0.05:
            return "almost always merge"
        if distance < 0.10:
            return "usually merge unless business reason"
        if distance < 0.20:
            return "borderline"
        if distance < 0.30:
            return "usually worth keeping"
        return "strong evidence for separate bins"

    def _direction_change_count(self, values: pd.Series) -> int:
        diffs = np.diff(values.to_numpy(dtype=float))
        non_zero = diffs[np.abs(diffs) > 1e-12]
        if len(non_zero) < 2:
            return 0
        return int(np.sum(np.sign(non_zero[1:]) != np.sign(non_zero[:-1])))

    def _ordering_assessment(self, summary: pd.DataFrame, kind: str) -> str:
        if kind != "numeric":
            return "risk-ordered for grouping proposal"
        regular = summary[summary["bin"] != MISSING_BIN].copy()
        if regular.shape[0] < 3:
            return "too few bins"
        direction_changes = self._direction_change_count(regular["default_rate"])
        if direction_changes == 0:
            return "monotonic"
        if direction_changes <= self.max_numeric_direction_changes:
            return f"{direction_changes} broad turn retained"
        return f"review: {direction_changes} direction changes"


def fit_scorecard_logit_model(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: Optional[pd.Series] = None,
) -> LogisticRegression:
    model = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X, y, sample_weight=sample_weight)
    return model


def build_scorecard_model(
    df: pd.DataFrame,
    selected_sources: Optional[Iterable[str]] = None,
    max_numeric_bins: int = 20,
    merge_woe_distance: float = DEFAULT_WOE_MERGE_DISTANCE,
    max_numeric_direction_changes: int = DEFAULT_MAX_NUMERIC_DIRECTION_CHANGES,
) -> tuple[LogisticRegression, pd.DataFrame, pd.Series, WoeScorecardTransformer, pd.Series]:
    X_raw, y = prepare_scorecard_frame(df, selected_sources=selected_sources)
    numeric_features = [feature for feature in SCORECARD_NUMERIC_FEATURES if feature in X_raw.columns]
    categorical_features = [feature for feature in SCORECARD_CATEGORICAL_FEATURES if feature in X_raw.columns]
    transformer = WoeScorecardTransformer(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        max_numeric_bins=max_numeric_bins,
        merge_woe_distance=merge_woe_distance,
        max_numeric_direction_changes=max_numeric_direction_changes,
    )
    X_woe = transformer.fit_transform(X_raw, y)
    model = fit_scorecard_logit_model(X_woe, y)
    coefficients = pd.Series(model.coef_[0], index=X_woe.columns)
    return model, X_woe, y, transformer, coefficients


def make_scorecard_fold_transformer(
    selected_sources: Iterable[str],
    max_numeric_bins: int = 20,
    merge_woe_distance: float = DEFAULT_WOE_MERGE_DISTANCE,
    max_numeric_direction_changes: int = DEFAULT_MAX_NUMERIC_DIRECTION_CHANGES,
):
    selected_sources = list(selected_sources)

    def factory() -> WoeScorecardTransformer:
        return WoeScorecardTransformer(
            numeric_features=[feature for feature in SCORECARD_NUMERIC_FEATURES if feature in selected_sources],
            categorical_features=[feature for feature in SCORECARD_CATEGORICAL_FEATURES if feature in selected_sources],
            max_numeric_bins=max_numeric_bins,
            merge_woe_distance=merge_woe_distance,
            max_numeric_direction_changes=max_numeric_direction_changes,
        )

    return factory
