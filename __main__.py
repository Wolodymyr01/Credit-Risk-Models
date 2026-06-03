import pandas as pd

from eda import analyze_eda
from eda_report import create_eda_report
from evaluation import (
    compare_evaluations,
    cross_validate_model,
    evaluate_holdout_model,
    evaluate_model_results,
    make_train_test_indices,
    segment_performance,
)
from evaluation_report import create_html_evaluation_report
from sampling import (
    effective_sample_size,
    inverse_frequency_weights,
    numeric_balance_diagnostics,
    representative_sample_indices,
    sampling_diagnostics,
    weight_diagnostics,
)
from raw_logit import (
    dummy_feature_groups,
    fit_raw_logit_model,
    minimize_aic_features,
    predictor_list_from_coefficients,
    prepare_raw_logit_data,
    select_uncorrelated_features,
)
from scoring import (
    fit_scorecard_logit_model,
    make_scorecard_fold_transformer,
    prepare_scorecard_frame,
    scorecard_sources_from_raw_features,
)
from thresholding import ProfitLossConfig


THRESHOLD_STRATEGY = "recommended"
PROFIT_LOSS_CONFIG = ProfitLossConfig(
    true_negative=0.20,
    false_positive=-0.20,
    false_negative=-1.00,
    true_positive=0.00,
)


def clean_credit_risk_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Missing employment length is informative, so keep an indicator and impute the usable numeric input.
    df["emp_length_missing"] = df["person_emp_length"].isna().astype(int)
    df["int_rate_missing"] = df["loan_int_rate"].isna().astype(int)
    df["emp_length"] = df["person_emp_length"].fillna(0)

    # Loan grade is treated as an external risk estimate, so interest-rate imputation avoids grade-level medians.
    df["loan_int_rate"] = df["loan_int_rate"].fillna(df["loan_int_rate"].median())

    # These rows are implausible for consumer-credit modeling and distort age/employment diagnostics.
    df = df[df["person_age"] <= 120]

    income_99th_percentile = df["person_income"].quantile(0.99)
    df = df[df["person_income"] <= income_99th_percentile]
    return df


def adjusted_features_from_significance(full_evaluation: dict, all_features: list[str]) -> tuple[list[str], list[str]]:
    feature_groups = full_evaluation.get("feature_groups", {})
    non_significant_groups = set(full_evaluation.get("non_significant_groups", []))
    grouped_features_to_remove = {
        feature
        for group_name, features in feature_groups.items()
        if group_name in non_significant_groups
        for feature in features
    }
    ungrouped_features_to_remove = set(full_evaluation.get("non_significant_predictors", []))
    removed_features = sorted((grouped_features_to_remove | ungrouped_features_to_remove) & set(all_features))
    adjusted_features = [feature for feature in all_features if feature not in removed_features]
    return adjusted_features, removed_features


def main() -> None:
    credit_risk_df = pd.read_csv("credit_risk_dataset.csv")
    raw_eda = analyze_eda(credit_risk_df)
    raw_eda_report = create_eda_report(credit_risk_df, raw_eda, save_dir="eda_plots/raw")
    print(f"Raw EDA report saved to: {raw_eda_report}")

    cleaned_df = clean_credit_risk_data(credit_risk_df)
    cleaned_eda = analyze_eda(cleaned_df)
    cleaned_eda_report = create_eda_report(cleaned_df, cleaned_eda, save_dir="eda_plots/cleaned")
    print(f"Cleaned EDA report saved to: {cleaned_eda_report}")

    modeling_df = cleaned_df
    X_all, y = prepare_raw_logit_data(modeling_df)
    selected_uncorrelated_features, correlation_screen = select_uncorrelated_features(X_all, y)
    X = X_all[selected_uncorrelated_features]
    scoring_df = modeling_df.loc[X.index]
    if correlation_screen.empty:
        print("\nNo predictor pairs exceeded the pairwise-correlation threshold.")
    else:
        print("\nPredictors removed by pairwise-correlation screen:")
        print(correlation_screen.to_string(index=False))

    train_index, test_index = make_train_test_indices(X, y)
    X_train = X.loc[train_index]
    X_test = X.loc[test_index]
    y_train = y.loc[train_index]
    y_test = y.loc[test_index]
    print(f"\nStratified train/test split: {len(train_index)} train rows, {len(test_index)} test rows.")

    feature_groups = dummy_feature_groups(X)
    raw_model = fit_raw_logit_model(X_train, y_train)
    raw_coefficients = pd.Series(raw_model.coef_[0], index=X_train.columns)
    raw_predictors = list(raw_coefficients.abs().sort_values(ascending=False).index)
    print("Top raw-logit indicators by absolute coefficient:")
    print(predictor_list_from_coefficients(raw_coefficients).to_string())
    print("\nOrdered predictors:")
    print(raw_predictors)

    raw_train_diagnostics = evaluate_model_results(
        raw_model,
        X_train,
        y_train,
        model_name="Raw logit full model",
        feature_groups=feature_groups,
        threshold_strategy=THRESHOLD_STRATEGY,
        profit_loss=PROFIT_LOSS_CONFIG,
    )
    raw_evaluation = evaluate_holdout_model(
        raw_model,
        X_train,
        y_train,
        X_test,
        y_test,
        model_name="Raw logit full model",
        feature_groups=feature_groups,
        threshold_strategy=THRESHOLD_STRATEGY,
        profit_loss=PROFIT_LOSS_CONFIG,
        train_diagnostics=raw_train_diagnostics,
    )
    evaluations = [raw_evaluation]
    if not raw_train_diagnostics["group_likelihood_ratio_tests"].empty:
        print("\nGrouped dummy-variable LR tests:")
        print(raw_train_diagnostics["group_likelihood_ratio_tests"].to_string(index=False))

    adjusted_features, removed_features = adjusted_features_from_significance(raw_train_diagnostics, list(X.columns))
    if removed_features:
        adjusted_model = fit_raw_logit_model(X_train[adjusted_features], y_train)
        adjusted_feature_groups = {
            group_name: [feature for feature in features if feature in adjusted_features]
            for group_name, features in feature_groups.items()
            if any(feature in adjusted_features for feature in features)
        }
        adjusted_train_diagnostics = evaluate_model_results(
            adjusted_model,
            X_train[adjusted_features],
            y_train,
            model_name="Raw logit adjusted model",
            feature_groups=adjusted_feature_groups,
            threshold_strategy=THRESHOLD_STRATEGY,
            profit_loss=PROFIT_LOSS_CONFIG,
        )
        adjusted_evaluation = evaluate_holdout_model(
            adjusted_model,
            X_train[adjusted_features],
            y_train,
            X_test[adjusted_features],
            y_test,
            model_name="Raw logit adjusted model",
            feature_groups=adjusted_feature_groups,
            threshold_strategy=THRESHOLD_STRATEGY,
            profit_loss=PROFIT_LOSS_CONFIG,
            train_diagnostics=adjusted_train_diagnostics,
        )
        evaluations.append(adjusted_evaluation)
        print("\nFeatures removed in adjusted model:")
        print(removed_features)
        if raw_train_diagnostics["non_significant_groups"]:
            print("\nNon-significant dummy groups removed by LR test:")
            print(raw_train_diagnostics["non_significant_groups"])
        if raw_train_diagnostics["non_significant_predictors"]:
            print("\nNon-significant ungrouped indicators removed by Wald test:")
            print(raw_train_diagnostics["non_significant_predictors"])
    else:
        adjusted_model = raw_model
        adjusted_features = list(X.columns)
        adjusted_feature_groups = feature_groups
        adjusted_train_diagnostics = raw_train_diagnostics
        print("\nNo non-significant dummy groups or ungrouped indicators found at alpha 0.05.")

    cross_validation_results = []
    aic_features, aic_steps = minimize_aic_features(X_train, y_train, adjusted_features)
    if aic_steps.empty:
        aic_model = adjusted_model
        modeling_features = adjusted_features
        modeling_feature_groups = adjusted_feature_groups
        print("\nAIC minimization did not remove any additional features.")
    else:
        aic_model = fit_raw_logit_model(X_train[aic_features], y_train)
        modeling_features = aic_features
        modeling_feature_groups = {
            group_name: [feature for feature in features if feature in modeling_features]
            for group_name, features in adjusted_feature_groups.items()
            if any(feature in modeling_features for feature in features)
        }
        aic_train_diagnostics = evaluate_model_results(
            aic_model,
            X_train[modeling_features],
            y_train,
            model_name="Raw logit AIC-minimized model",
            feature_groups=modeling_feature_groups,
            threshold_strategy=THRESHOLD_STRATEGY,
            profit_loss=PROFIT_LOSS_CONFIG,
        )
        aic_evaluation = evaluate_holdout_model(
            aic_model,
            X_train[modeling_features],
            y_train,
            X_test[modeling_features],
            y_test,
            model_name="Raw logit AIC-minimized model",
            feature_groups=modeling_feature_groups,
            threshold_strategy=THRESHOLD_STRATEGY,
            profit_loss=PROFIT_LOSS_CONFIG,
            train_diagnostics=aic_train_diagnostics,
        )
        evaluations.append(aic_evaluation)
        print("\nAIC minimization steps:")
        print(aic_steps.to_string(index=False))

    representative_index = representative_sample_indices(scoring_df.loc[train_index])
    representative_model = fit_raw_logit_model(
        X.loc[representative_index, modeling_features],
        y.loc[representative_index],
    )
    representative_evaluation = evaluate_holdout_model(
        representative_model,
        X.loc[representative_index, modeling_features],
        y.loc[representative_index],
        X_test[modeling_features],
        y_test,
        model_name="Raw logit representative-sample model",
        feature_groups=modeling_feature_groups,
        threshold_strategy=THRESHOLD_STRATEGY,
        profit_loss=PROFIT_LOSS_CONFIG,
    )
    evaluations.append(representative_evaluation)

    def representative_fold_selector(X_train: pd.DataFrame, y_train: pd.Series, fold_index: int) -> pd.Index:
        return representative_sample_indices(
            scoring_df.loc[X_train.index],
            random_state=42 + fold_index,
        )

    cross_validation_results.append(
        cross_validate_model(
            representative_model,
            X[modeling_features],
            y,
            model_name="Raw logit representative-sample model",
            train_index_selector=representative_fold_selector,
            threshold_strategy=THRESHOLD_STRATEGY,
            profit_loss=PROFIT_LOSS_CONFIG,
        )
    )

    sample_weights = inverse_frequency_weights(scoring_df).loc[X.index]
    weighted_model = fit_raw_logit_model(
        X_train[modeling_features],
        y_train,
        sample_weight=sample_weights.loc[train_index],
    )
    weighted_evaluation = evaluate_holdout_model(
        weighted_model,
        X_train[modeling_features],
        y_train,
        X_test[modeling_features],
        y_test,
        model_name="Raw logit weighted model",
        feature_groups=modeling_feature_groups,
        threshold_strategy=THRESHOLD_STRATEGY,
        profit_loss=PROFIT_LOSS_CONFIG,
    )
    evaluations.append(weighted_evaluation)

    scorecard_sources = scorecard_sources_from_raw_features(modeling_features)
    X_scorecard_raw, y_scorecard_raw = prepare_scorecard_frame(
        modeling_df,
        selected_sources=scorecard_sources,
    )
    scorecard_train_index = X_scorecard_raw.index.intersection(train_index)
    scorecard_test_index = X_scorecard_raw.index.intersection(test_index)
    scorecard_transformer = make_scorecard_fold_transformer(scorecard_sources)()
    X_scorecard_train = scorecard_transformer.fit_transform(
        X_scorecard_raw.loc[scorecard_train_index],
        y_scorecard_raw.loc[scorecard_train_index],
    )
    X_scorecard_test = scorecard_transformer.transform(X_scorecard_raw.loc[scorecard_test_index])
    scorecard_model = fit_scorecard_logit_model(
        X_scorecard_train,
        y_scorecard_raw.loc[scorecard_train_index],
    )
    scorecard_coefficients = pd.Series(scorecard_model.coef_[0], index=X_scorecard_train.columns)
    scorecard_evaluation = evaluate_holdout_model(
        scorecard_model,
        X_scorecard_train,
        y_scorecard_raw.loc[scorecard_train_index],
        X_scorecard_test,
        y_scorecard_raw.loc[scorecard_test_index],
        model_name="WoE scorecard model",
        feature_groups={},
        threshold_strategy=THRESHOLD_STRATEGY,
        profit_loss=PROFIT_LOSS_CONFIG,
    )
    scorecard_evaluation["scorecard_bins"] = scorecard_transformer.summary_
    scorecard_evaluation["scorecard_source_features"] = scorecard_sources
    evaluations.append(scorecard_evaluation)
    print("\nScorecard source features:")
    print(scorecard_sources)
    print("\nTop WoE scorecard indicators by absolute coefficient:")
    print(predictor_list_from_coefficients(scorecard_coefficients).to_string())

    sampling_report = {
        "representative": {
            "full_size": int(len(train_index)),
            "sample_size": int(len(representative_index)),
            "sample_share": float(len(representative_index) / len(train_index)),
            "strata": sampling_diagnostics(scoring_df.loc[train_index], representative_index),
            "numeric_balance": numeric_balance_diagnostics(
                scoring_df.loc[train_index],
                representative_index,
                [
                    "person_age",
                    "person_income",
                    "person_emp_length",
                    "loan_amnt",
                    "loan_int_rate",
                    "cb_person_cred_hist_length",
                ],
            ),
        },
        "weighted": {
            "mean_weight": float(sample_weights.mean()),
            "min_weight": float(sample_weights.min()),
            "max_weight": float(sample_weights.max()),
            "effective_sample_size": effective_sample_size(sample_weights),
            "strata": weight_diagnostics(scoring_df, sample_weights),
        },
    }

    comparison = compare_evaluations(evaluations)
    print("\nModel evaluation comparison:")
    print(comparison.to_string(index=False))

    if cross_validation_results:
        cv_summary = pd.concat(
            [result["summary"] for result in cross_validation_results],
            ignore_index=True,
        )
        print("\nCross-validation summary:")
        print(cv_summary.to_string(index=False))

    segment_metrics = segment_performance(
        evaluations,
        scoring_df,
        ["loan_intent", "person_home_ownership"],
    )

    report_path = create_html_evaluation_report(
        evaluations,
        save_dir="model_evaluation",
        comparison=comparison,
        correlation_analysis=correlation_screen,
        cross_validation=cross_validation_results,
        sampling_diagnostics=sampling_report,
        segment_performance=segment_metrics,
    )
    print(f"\nModel evaluation report saved to: {report_path}")


if __name__ == "__main__":
    main()
