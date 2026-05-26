# ================================================
# CREDIT RISK ANALYSIS - UPROSZCZONA WERSJA (jeden plik)
# ================================================

library(tidyverse)
library(ggplot2)
library(pROC)
library(caret)      # do confusionMatrix

# ================================================
# 1. WCZYTANIE DANYCH
# ================================================

df_raw <- read_csv("credit_risk_dataset.csv", show_col_types = FALSE)

cat("Wymiary surowego zbioru:", dim(df_raw), "\n")

# ================================================
# 2. CZYSZCZENIE DANYCH
# ================================================

clean_data <- function(df) {
  df %>%
    mutate(
      # Wskaźniki braków
      emp_length_missing = as.integer(is.na(person_emp_length)),
      int_rate_missing = as.integer(is.na(loan_int_rate)),
      
      # Imputacja
      emp_length = coalesce(person_emp_length, 0),
      loan_int_rate = coalesce(loan_int_rate, median(loan_int_rate, na.rm = TRUE))
    ) %>%
    filter(person_age <= 120) %>%                                      # realistyczny wiek
    filter(person_income <= quantile(person_income, 0.99, na.rm = TRUE)) # usuwamy ekstremalne dochody
}

df <- clean_data(df_raw)

cat("Wymiary po czyszczeniu:", dim(df), "\n")

# ================================================
# 3. PODSTAWOWA EDA
# ================================================

cat("\n=== PODSTAWOWA ANALIZA EDA ===\n")

# Typy zmiennych
numeric_cols <- df %>% select(where(is.numeric)) %>% names()
cat_cols <- df %>% select(where(~ !is.numeric(.))) %>% names()

cat("Zmienne numeryczne:", length(numeric_cols), "\n")
cat("Zmienne kategoryczne:", length(cat_cols), "\n\n")

# Podsumowanie numeryczne
summary_numeric <- df %>%
  select(all_of(numeric_cols)) %>%
  summarise(across(everything(), list(
    mean = ~mean(., na.rm = TRUE),
    median = ~median(., na.rm = TRUE),
    sd = ~sd(., na.rm = TRUE),
    missing_pct = ~mean(is.na(.)) * 100
  ), .names = "{.col}_{.fn}")) %>%
  pivot_longer(everything(), names_to = "var", values_to = "value") %>%
  separate(var, into = c("variable", "stat"), sep = "_") %>%
  pivot_wider(names_from = stat, values_from = value)

print(summary_numeric, n = Inf)

# Rozkład targetu (loan_status)
cat("\nRozkład loan_status:\n")
table(df$loan_status) %>% prop.table() %>% round(4) %>% print()

# ================================================
# 4. PRZYGOTOWANIE DANYCH DO MODELU
# ================================================

prepare_model_data <- function(df) {
  df_model <- df %>%
    select(
      loan_status,
      person_age, person_income, emp_length, loan_amnt, 
      loan_int_rate, loan_percent_income, cb_person_cred_hist_length,
      emp_length_missing, int_rate_missing,
      person_home_ownership, loan_intent, cb_person_default_on_file
    )
  
  # One-hot encoding
  model_matrix <- model.matrix(loan_status ~ ., data = df_model)[, -1]
  
  list(
    X = as.data.frame(model_matrix),
    y = df_model$loan_status
  )
}

data_model <- prepare_model_data(df)

# ================================================
# 5. BUDOWA MODELU LOGISTYCZNEGO
# ================================================

model <- glm(loan_status ~ ., 
             data = cbind(data_model$X, loan_status = data_model$y),
             family = binomial(link = "logit"))

cat("\n=== PODSUMOWANIE MODELU ===\n")
summary(model)

# ================================================
# 6. EWALUACJA MODELU
# ================================================

cat("\n=== EWALUACJA MODELU ===\n")

# Predykcje
pred_prob <- predict(model, type = "response")

# ROC i AUC
roc_obj <- roc(data_model$y, pred_prob, quiet = TRUE)
cat("AUC:", round(auc(roc_obj), 4), "\n")

# Confusion Matrix przy progu 0.5
pred_class <- ifelse(pred_prob > 0.5, 1, 0)
cm <- confusionMatrix(factor(pred_class), factor(data_model$y), positive = "1")
print(cm)

# ================================================
# 7. DODATKOWE ANALIZY (uproszczone)
# ================================================

cat("\n=== DODATKOWE WNIOSKI ===\n")

# Najważniejsze zmienne (wg p-value)
coef_summary <- summary(model)$coefficients %>%
  as.data.frame() %>%
  rownames_to_column("variable") %>%
  arrange(`Pr(>|z|)`)

cat("Najistotniejsze zmienne (top 10):\n")
print(head(coef_summary, 10))

# ================================================
# 8. PROSTE WIZUALIZACJE
# ================================================

# Rozkład prawdopodobieństwa defaultu
ggplot(data.frame(prob = pred_prob, default = factor(data_model$y)), 
       aes(x = prob, fill = default)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  labs(title = "Rozkład przewidywanych prawdopodobieństw defaultu",
       x = "Przewidywane prawdopodobieństwo", y = "Liczba obserwacji") +
  theme_minimal()

# ROC Curve
plot(roc_obj, main = "Krzywa ROC", col = "blue", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")

cat("Model ma AUC =", round(auc(roc_obj), 4), "
