import pandas as pd
from eda import run_eda

credit_risk_df = pd.read_csv('credit_risk_dataset.csv')
run_eda(credit_risk_df)
print(pd.isna(credit_risk_df).sum())

# Loan interest rate and employment length have missing values. We can create new features to indicate
# whether these values are missing or not and analyse their behaviour with respect to the target variable
# 'loan_status'. This can help us understand if the missingness of these features is related to the likelihood of loan default.
credit_risk_df["emp_length_missing"] = credit_risk_df["person_emp_length"].isna().astype(int)
credit_risk_df["int_rate_missing"] = credit_risk_df["loan_int_rate"].isna().astype(int)

employment_length_missing = credit_risk_df.groupby("emp_length_missing")
print(employment_length_missing["loan_status"].mean())
print(employment_length_missing["person_income"].mean())

# The missingness of employment length seems to be associated with a higher likelihood of loan default
# and lower income. This suggests that individuals with missing employment length data may be at a higher risk
# of defaulting on their loans, possibly due to financial instability or lack of steady employment.
# Missingness is not random and may be informative for our model. It cannot be ignored or imputed without
# considering its potential impact on the target variable. We will fill it with 0 in a combination with missingness indicator.
credit_risk_df["emp_length"] = credit_risk_df["person_emp_length"].fillna(0)
print(credit_risk_df.groupby("int_rate_missing")["loan_status"].mean())

# The missingness of loan interest rate does not seem to be strongly associated with the likelihood of loan default.
# It seems to be random (ETL job issues, not disclosed by the borrower, etc.) and may not provide useful information for our model.
# It looks safe to impute the missing values with median of loan interest rate without worrying about introducing bias related to the target variable.
grade_median_int_rate = credit_risk_df.groupby("loan_grade")["loan_int_rate"].median()
credit_risk_df["loan_int_rate"] = credit_risk_df.apply(
    lambda row: grade_median_int_rate[row["loan_grade"]]
    if pd.isna(row["loan_int_rate"]) else row["loan_int_rate"],
    axis=1,
)

print(pd.isna(credit_risk_df).sum())
# No missing values remain in the dataset after handling the missing data for 'person_emp_length' and 'loan_int_rate'.

print(credit_risk_df[["emp_length", "person_age", "cb_person_cred_hist_length"]].describe())
# Some people in dataset are older than 120 years and have very long employment history. 
# These outliers may be due to data entry errors and should be removed as non-realistic values
credit_risk_df.drop(credit_risk_df[credit_risk_df["person_age"] > 120].index, inplace=True)
run_eda(credit_risk_df)