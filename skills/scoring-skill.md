# Weight of Evidence (WoE) & Quasi-Continuous Transformation Design Skill

## Objective

Design robust, explainable, and stable predictor transformations for scorecard-style models, primarily Logistic Regression.

The goal is not merely maximizing training performance, but balancing:

* Predictive power
* Stability over time
* Interpretability
* Regulatory explainability
* Resistance to overfitting

The resulting transformed variables should be suitable for scorecards, PD models, IFRS9 models, Basel models, fraud models, or other risk-ranking systems.

---

# General Principles

Always prioritize:

1. Stability over maximum in-sample performance.
2. Business plausibility over statistical artifacts.
3. Simplicity when performance differences are negligible.
4. Monotonic relationships whenever reasonable.
5. Future robustness rather than historical fit.

Do not create bins solely because they improve training metrics.

Every bin boundary should have either:

* statistical justification,
* business justification,
* or both.

---

# Input Review Phase

For each variable evaluate:

## Distribution

Inspect:

* Min/max
* Quantiles
* Skewness
* Outliers
* Zero inflation
* Special coded values

Examples:

-999
999
9999
0

may represent:

* Unknown
* Not applicable
* No history
* Missing information

Never assume these values are genuine observations.

---

## Missing Value Analysis

Determine whether missingness is informative.

Questions:

* Does missingness correlate with target?
* Does missingness correlate with customer segments?
* Does missingness have business meaning?

Examples:

Employment Length Missing:

* Student
* Self-employed
* New applicant
* Data unavailable

may itself carry predictive information.

### Decision Rule

If missingness is informative:

Create dedicated missing bin.

If missingness appears random:

Imputation may be considered.

Never automatically impute before investigation.

---

# Fine Classing

Purpose:

Discover the underlying risk pattern.

Recommended methods:

* Quantile bins
* Equal-frequency bins
* Tree-based pre-binning
* Expert-defined bins

Typical count:

10–50 bins.

Fine classing is exploratory and should not be used directly in production.

---

# Coarse Classing

Merge neighboring bins based on:

## Similar Risk

Example:

| Bin | Bad Rate |
| --- | -------- |
| A   | 10.1%    |
| B   | 10.4%    |

Candidate for merge.

---

## Similar WoE

Example:

| Bin | WoE   |
| --- | ----- |
| A   | -0.23 |
| B   | -0.26 |

Candidate for merge.

---

## Sample Adequacy

Avoid bins with:

* Very small populations
* Very few events
* Very few non-events

Typical guidelines:

* ≥5% observations
* ≥30 bads
* ≥100 observations

Adapt thresholds to dataset size.

---

# Weight of Evidence Calculation

For each bin:

WoE = ln(
(% Good in Bin) /
(% Bad in Bin)
)

Requirements:

* Avoid division by zero.
* Apply smoothing when necessary.
* Monitor bins with extremely large WoE values.

Extremely large positive or negative WoE often indicates:

* Tiny sample sizes
* Leakage
* Data quality issues

---

# Monotonicity Assessment

Preferred outcome:

Risk changes consistently in one direction.

Example:

Age increases → default risk decreases.

### Good Pattern

18-25 -> 20%
25-35 -> 15%
35-45 -> 10%
45-55 -> 7%

### Problematic Pattern

18-25 -> 20%
25-35 -> 15%
35-45 -> 18%
45-55 -> 7%

Investigate whether:

* Statistical noise exists
* Business rationale exists
* Interaction effects exist

If no strong justification exists:

Merge bins to restore monotonicity.

---

# Special Values

Always isolate:

* Missing values
* Sentinel values
* Administrative codes

Examples:

-999
-1
999
9999

Create dedicated bins before evaluating regular ranges.

Never combine special values with continuous ranges without justification.

---

# Information Value (IV)

Use IV for variable screening.

General interpretation:

| IV        | Interpretation    |
| --------- | ----------------- |
| <0.02     | Weak              |
| 0.02–0.10 | Useful            |
| 0.10–0.30 | Strong            |
| 0.30–0.50 | Very Strong       |
| >0.50     | Potential Leakage |

Do not select variables solely based on IV.

Always combine:

* IV
* Stability
* Business interpretation
* Correlation review

---

# Quasi-Continuous Transformation

Purpose:

Approximate smooth non-linear risk curves while preserving scorecard explainability.

Instead of:

18-25
25-40
40+

use many narrow intervals:

18-20
20-22
22-24
24-26
...

with corresponding WoE or score values.

Benefits:

* Better ranking power
* Better non-linearity capture
* Improved calibration

Costs:

* Higher complexity
* More maintenance
* Greater instability risk
* More governance effort

Use only when validation evidence supports the added complexity.

---

# Stability Requirements

Evaluate on:

* Out-of-time sample
* Different vintages
* Different customer cohorts

Check:

## Population Stability Index (PSI)

Monitor:

* Variable distribution drift
* Score distribution drift

Typical interpretation:

PSI < 0.10 → Stable

0.10–0.25 → Monitor

> 0.25 → Significant drift

---

## WoE Drift

Compare WoE values across time periods.

Large shifts indicate instability.

Example:

High Income Bin

2024 WoE = 1.20

2025 WoE = 0.35

Requires investigation.

---

# Variable Selection

Prefer variables that are:

* Predictive
* Stable
* Explainable
* Operationally available

Reject variables that are:

* Unstable
* Difficult to explain
* Likely to disappear operationally
* Highly leakage-prone

---

# Multicollinearity Review

WoE does not eliminate correlation.

Review:

* Correlation matrix
* Variance Inflation Factor (VIF)
* Coefficient stability

Common examples:

* Loan Amount
* Monthly Installment
* Debt-to-Income

may remain strongly correlated after WoE transformation.

---

# Leakage Detection

Investigate variables with:

* Extremely high IV
* Extremely high WoE
* Unrealistic predictive power

Potential causes:

* Future information
* Post-decision attributes
* Data preparation errors

Prefer removing suspicious variables.

---

# Governance Requirements

Every transformation must be documented.

Document:

* Bin boundaries
* WoE values
* Sample sizes
* Event rates
* Missing handling
* Business rationale

Transformation logic must be reproducible.

Avoid undocumented manual adjustments.

---

# Validation Checklist

For each transformed variable verify:

* Missing values handled
* Special values handled
* Adequate sample size per bin
* Monotonicity assessed
* WoE calculated correctly
* IV reviewed
* Stability checked
* Leakage assessed
* Correlation reviewed
* Business explanation documented

---

# Model-Level Decision Framework

For each candidate variable:

1. Is missingness informative?

   * Yes → dedicated bin.
   * No → consider imputation.

2. Is relationship monotonic?

   * Yes → preserve monotonic structure.
   * No → investigate before forcing monotonicity.

3. Is the variable stable over time?

   * Yes → keep.
   * No → simplify or remove.

4. Does finer granularity improve validation metrics?

   * Yes → consider quasi-continuous scale.
   * No → use simpler bins.

5. Is model explainability a key requirement?

   * Yes → prefer WoE scorecard.
   * No → consider GAM, spline-based logistic regression, XGBoost, LightGBM, or CatBoost.

---

# Guiding Philosophy

The purpose of WoE engineering is to construct a stable and explainable approximation of the true risk function before fitting the logistic regression model.

When trade-offs exist:

Stability > Interpretability > Predictive Power

unless project objectives explicitly require otherwise.
