import pandas as pd

from eda import analyze_eda
from eda_report import create_eda_report
from evaluation import compare_evaluations, cross_validate_model, evaluate_model_results, segment_performance
from evaluation_report import create_html_evaluation_report
from sampling import (
    effective_sample_size,
    inverse_frequency_weights,
    numeric_balance_diagnostics,
    representative_sample_indices,
    sampling_diagnostics,
    weight_diagnostics,
)
from scoring import (
    build_logit_scoring_model,
    dummy_feature_groups,
    fit_logit_scoring_model,
    predictor_list_from_coefficients,
    prepare_scoring_data,
)


def clean_credit_risk_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Missing employment length is informative, so keep an indicator and impute the usable numeric input.
    df["emp_length_missing"] = df["person_emp_length"].isna().astype(int)
    df["int_rate_missing"] = df["loan_int_rate"].isna().astype(int)
    df["emp_length"] = df["person_emp_length"].fillna(0)

    # Interest-rate missingness is weakly target-related; grade-level medians preserve risk ordering.
    grade_median_int_rate = df.groupby("loan_grade")["loan_int_rate"].median()
    df["loan_int_rate"] = df.apply(
        lambda row: grade_median_int_rate[row["loan_grade"]]
        if pd.isna(row["loan_int_rate"]) else row["loan_int_rate"],
        axis=1,
    )

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

    scoring_model, scoring_predictors, scoring_coefficients = build_logit_scoring_model(cleaned_df)
    print("Top model indicators by absolute coefficient:")
    print(predictor_list_from_coefficients(scoring_coefficients).to_string())
    print("\nOrdered predictors:")
    print(scoring_predictors)

    X, y = prepare_scoring_data(cleaned_df)
    scoring_df = cleaned_df.loc[X.index]
    feature_groups = dummy_feature_groups(X)
    full_evaluation = evaluate_model_results(
        scoring_model,
        X,
        y,
        model_name="Full model",
        feature_groups=feature_groups,
    )
    evaluations = [full_evaluation]
    cross_validation_results = [
        cross_validate_model(scoring_model, X, y, model_name="Full model")
    ]
    if not full_evaluation["group_likelihood_ratio_tests"].empty:
        print("\nGrouped dummy-variable LR tests:")
        print(full_evaluation["group_likelihood_ratio_tests"].to_string(index=False))

    adjusted_features, removed_features = adjusted_features_from_significance(full_evaluation, list(X.columns))
    if removed_features:
        adjusted_model = fit_logit_scoring_model(X[adjusted_features], y)
        adjusted_feature_groups = {
            group_name: [feature for feature in features if feature in adjusted_features]
            for group_name, features in feature_groups.items()
            if any(feature in adjusted_features for feature in features)
        }
        adjusted_evaluation = evaluate_model_results(
            adjusted_model,
            X[adjusted_features],
            y,
            model_name="Adjusted grouped-LR model",
            feature_groups=adjusted_feature_groups,
        )
        evaluations.append(adjusted_evaluation)
        cross_validation_results.append(
            cross_validate_model(
                adjusted_model,
                X[adjusted_features],
                y,
                model_name="Adjusted grouped-LR model",
            )
        )
        print("\nFeatures removed in adjusted model:")
        print(removed_features)
        if full_evaluation["non_significant_groups"]:
            print("\nNon-significant dummy groups removed by LR test:")
            print(full_evaluation["non_significant_groups"])
        if full_evaluation["non_significant_predictors"]:
            print("\nNon-significant ungrouped indicators removed by Wald test:")
            print(full_evaluation["non_significant_predictors"])
    else:
        adjusted_model = scoring_model
        adjusted_features = list(X.columns)
        adjusted_feature_groups = feature_groups
        print("\nNo non-significant dummy groups or ungrouped indicators found at alpha 0.05.")

    representative_index = representative_sample_indices(scoring_df)
    representative_model = fit_logit_scoring_model(
        X.loc[representative_index, adjusted_features],
        y.loc[representative_index],
    )
    representative_evaluation = evaluate_model_results(
        representative_model,
        X[adjusted_features],
        y,
        model_name="Representative-sample adjusted model",
        feature_groups=adjusted_feature_groups,
        run_model_diagnostics=False,
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
            X[adjusted_features],
            y,
            model_name="Representative-sample adjusted model",
            train_index_selector=representative_fold_selector,
        )
    )

    sample_weights = inverse_frequency_weights(scoring_df).loc[X.index]
    weighted_model = fit_logit_scoring_model(
        X[adjusted_features],
        y,
        sample_weight=sample_weights,
    )
    weighted_evaluation = evaluate_model_results(
        weighted_model,
        X[adjusted_features],
        y,
        model_name="Weighted adjusted model",
        feature_groups=adjusted_feature_groups,
        run_model_diagnostics=False,
    )
    evaluations.append(weighted_evaluation)
    cross_validation_results.append(
        cross_validate_model(
            weighted_model,
            X[adjusted_features],
            y,
            model_name="Weighted adjusted model",
            train_weight_provider=lambda X_train, _y_train, _fold_index: inverse_frequency_weights(
                scoring_df.loc[X_train.index]
            ),
        )
    )

    sampling_report = {
        "representative": {
            "full_size": int(scoring_df.shape[0]),
            "sample_size": int(len(representative_index)),
            "sample_share": float(len(representative_index) / scoring_df.shape[0]),
            "strata": sampling_diagnostics(scoring_df, representative_index),
            "numeric_balance": numeric_balance_diagnostics(
                scoring_df,
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

    cv_summary = pd.concat(
        [result["summary"] for result in cross_validation_results],
        ignore_index=True,
    )
    print("\nCross-validation summary:")
    print(cv_summary.to_string(index=False))

    segment_metrics = segment_performance(
        evaluations,
        scoring_df,
        ["loan_grade", "loan_intent", "person_home_ownership"],
    )

    report_path = create_html_evaluation_report(
        evaluations,
        save_dir="model_evaluation",
        comparison=comparison,
        cross_validation=cross_validation_results,
        sampling_diagnostics=sampling_report,
        segment_performance=segment_metrics,
    )
    print(f"\nModel evaluation report saved to: {report_path}")


if __name__ == "__main__":
    main()
