import pandas as pd

from eda import analyze_eda
from eda_report import create_eda_report
from evaluation import compare_evaluations, evaluate_model_results
from evaluation_report import create_html_evaluation_report
from scoring import (
    build_logit_scoring_model,
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
    full_evaluation = evaluate_model_results(scoring_model, X, y, model_name="Full model")
    evaluations = [full_evaluation]

    non_significant = full_evaluation["non_significant_predictors"]
    if non_significant:
        adjusted_features = [feature for feature in X.columns if feature not in non_significant]
        adjusted_model = fit_logit_scoring_model(X[adjusted_features], y)
        adjusted_evaluation = evaluate_model_results(
            adjusted_model,
            X[adjusted_features],
            y,
            model_name="Adjusted significant-indicator model",
        )
        evaluations.append(adjusted_evaluation)
        print("\nNon-significant indicators removed in adjusted model:")
        print(non_significant)
    else:
        print("\nNo non-significant indicators found at alpha 0.05.")

    comparison = compare_evaluations(evaluations)
    print("\nModel evaluation comparison:")
    print(comparison.to_string(index=False))

    report_path = create_html_evaluation_report(
        evaluations,
        save_dir="model_evaluation",
        comparison=comparison,
    )
    print(f"\nModel evaluation report saved to: {report_path}")


if __name__ == "__main__":
    main()
