# ================================================
# CREDIT RISK MODELING - WERSJA Z BOGATYMI WIZUALIZACJAMI
# ================================================

library(tidyverse)
library(ggplot2)
library(pROC)
library(caret)
library(car)
library(lmtest)
library(corrplot)
library(vcd)
library(caret)
library(lmtest) 
library(car) 
library(pscl)
library(pROC)
# ================================================
# 1. Wczytanie danych
# ================================================

df_raw <- read_csv("credit_risk_dataset.csv",show_col_types = FALSE)

# ================================================
# 2. Czyszczenie danych
# ================================================

clean_credit_risk_data <- function(df) {
  
  df %>%
    mutate(
      emp_length_missing = as.integer(is.na(person_emp_length)),
      int_rate_missing   = as.integer(is.na(loan_int_rate)),
      
      person_emp_length = coalesce(person_emp_length, 0),
      loan_int_rate = coalesce(
        loan_int_rate,
        median(loan_int_rate, na.rm = TRUE)
      )
    ) %>%
    filter(person_age <= 120) %>%
    filter(
      person_income <= quantile(
        person_income,
        0.99,
        na.rm = TRUE
      )
    )
}

df <- clean_credit_risk_data(df_raw)

#wartości nietypowe dla zmiennych ilościowych
bp1 <- boxplot(df[,1],show.names=TRUE)             
bp2 <- boxplot(df[,2],show.names=TRUE)
bp3 <- boxplot(df[,4],show.names=TRUE)
bp4 <- boxplot(df[,7],show.names=TRUE)
bp5 <- boxplot(df[,8],show.names=TRUE)
bp6 <- boxplot(df[,10],show.names=TRUE)


# ================================================
# 3. Podział train / test
# ================================================

#Sprawdzanie korelacji zmiennych parami, w razie korelacji > 0.7 trzeba odrzucić jedną ze zmiennych 
#Macierz korelacji dla ilościowych zmiennych objaśniających (współczynnik korelacji Pearsona)

round(cor(df[,c(1,2,4,7,8,10,12)]),3) # Wniosek - silna korelacja (0.878) występuje dla zmiennych person_age i cb_person_cred_hist_length, należy usunąć jedną z nich


#Współczynnik korelacji punktowo-dwuseryjnej** dla zmiennej ilościowej i dychotomicznej (zakodowanej 0-1 jako numeryczna) (jest równoważny współczynnikowi korelacji Pearsona)

round(cor(df[,c(1,2,4,7,8,10,12)], df[,c(9,13,14)]),3) #Wniosek - brak silnej korelacji, nie ma potrzeby usuwania zmiennych


# korelacja zmiennych jakościowych, współczynniki kontyngencji, V Cramera - silna korelacja, gdy `V > 0.5`.

assocstats(table(df$person_home_ownership, df$loan_intent))
assocstats(table(df$person_home_ownership, df$loan_grade))
assocstats(table(df$person_home_ownership, df$cb_person_default_on_file))
assocstats(table(df$loan_intent, df$loan_grade))
assocstats(table(df$loan_intent, df$cb_person_default_on_file))  #Wnioski - dla wszystkich par współczynnik V-Creamera jest <0.5, nie ma potrzeby usuwania zmiennych

summary(df$loan_status)

#podzial danych na zbiór uczący (1000 obserwacji) w którym jest tyle samo spłąconych co niespłaconych kredytów)

set.seed(42)
index_0 <- which(df$loan_status==0)
index_1 <- which(df$loan_status==1)

#losowanie po 5000 obserwacji gdzie klient spłacił i 5000 gdzie nie spłacił do zbioru testowego
train_0 <-sample(index_0, 5000, replace=FALSE)
train_1 <-sample(index_1, 5000, replace=FALSE)


train_data_index <- c(train_0,train_1)
train_data_index

#zbiór uczący
train_data <- df[train_data_index,]

#zbiór testowy
test_data <- df[-train_data_index,]
test_data

# ================================================
# 4. Model logistyczny
# ================================================

full_model <- glm(
  loan_status ~ .,
  data = train_data,
  family = binomial()
)

#TESTY


#Testy istotności wszystkich zmiennych niezależnych w modelu: test ilorazu wiarygodności i test Walda
lrtest(full_model)
waldtest(full_model) #zadziała jak usuniemy jedną z silnie skorelowanych zmiennych

#Sprawdzenie, czy zmienne objaśniające nie są współliniowe
vif(full_model)

#test na wartości nietypowe
outlierTest(full_model)

 
# selekcja AIC

model_aic_forward <- step(
  full_model,
  direction = "forward",
  trace = 0
)

model_aic_backward <- step(
  full_model,
  direction = "backward",
  trace = 0
)

model_aic_both <- step(
  full_model,
  direction = "both",
  trace = 0
)

#Funkcja do porównania modelów
ocena_modelu <- function(model) {
  kryterium_AIC <- AIC(model)
  kryterium_BIC <- BIC(model)
  McFadden<- pR2(model)[4]
  Cragg_Uhler<- pR2(model)[6]
  ocena <- data.frame(kryterium_AIC, kryterium_BIC, McFadden, Cragg_Uhler)
  return(ocena)
}

#Porównanie modelów
wyniki_modelów <- rbind(
  model_forward=ocena_modelu(model_aic_forward), 
  model_backward=ocena_modelu(model_aic_backward), 
  model_both=ocena_modelu(model_aic_both))
wyniki_modelów

#zależy nam na na lepszym dopasowaniu do danych więc wybieramy model z jak najmniejszym kryterium AIC i BIC, w tym przypadku to model 
#uzyskany metodą backward lub both (maja takie same wyniki)

#wybieramy model both

#krzywa ROC
krzywa czerwona - ROC wyznaczona na zbiorze uczącym

krzywa niebieska - ROC wyznaczona na zbiorze testowym

rocobj1 <- roc(model_aic_both$y, model_aic_both$fitted.values)
rocobj1_t <- roc(test_data$loan_status, predict(model_aic_both, test_data, type = "response"))
plot(rocobj1, main = "krzywe ROC dla modelu logitowego", col="red")
lines(rocobj1_t, col="blue")

#Pole powierzchni pod krzywą ROC

cat("AUC dla zbioru uczącego\n")
auc(rocobj1) 
cat("\nAUC dla zbioru testowego\n")
auc(rocobj1_t)            #Pole pod krzywą ROC zwiększyło się nieznacznie dla zbioru testowego, wynosi 0.8747 co oznacza dobrą jakość predykcji 


# wyznaczenie punktu odcięcia p* według indeksu Youdena dla krzywej ROC dla modelu logitowego, próba ucząca
p=0.5
p1 <- p
cat("Punkt odcięcia jako proporcja z próby uczącej p*=", p1, "\n")
p2 <- coords(rocobj1, "best", ret="threshold", best.method = "youden", transpose = TRUE)
cat("Punkt odcięcia według indeksu Youdena dla próby uczącej p*=", p2, "\n")

cat("\nPorównanie miar jakości predykcji dla dwóch punktów odcięcia \n")
coords(rocobj1, p1, ret=c("threshold", "acc", "sens", "spec", "ppv", "npv", "youden"), transpose = TRUE)
coords(rocobj1, "best", ret=c("threshold", "acc", "sens", "spec", "ppv", "npv", "youden"), best.method = "youden", transpose = TRUE)

#Funkcja do oceny jakości predykcji dla wyznaczonego według indeksu Youdena punktu odcięcia p2
miary_pred <- function(model, dane, Y, p = p2) {
  tab <- table(obserwowane = Y, przewidywane = ifelse(predict(model, dane, type = "response") > p, 1, 0))
  ACC <- (tab[1,1]+tab[2,2])/sum(tab)
  ER <- (tab[1,2]+tab[2,1])/sum(tab)
  SENS <- (tab[2,2]/(tab[2,2]+tab[2,1]))
  SPEC <- (tab[1,1]/(tab[1,1]+tab[1,2]))
  PPV <-  (tab[2,2]/(tab[2,2]+tab[1,2]))
  NPV <-  (tab[1,1]/(tab[1,1]+tab[2,1]))
  miary <- data.frame(ACC, ER, SENS, SPEC, PPV, NPV)
  return(miary)
}

#Ocena zdolności predykcyjnej na zbiorze uczącym


wyniki_miary_pred_train <- rbind(
  model_logit = miary_pred(model = model_aic_both, dane = train_data,  Y = train_data$loan_status, p2)) 
wyniki_miary_pred_train

#Ocena zdolności predykcyjnej na zbiorze testowym

wyniki_miary_pred_test <- rbind(
  model_logit = miary_pred(model = model_aic_both, dane = test_data,  Y = test_data$loan_status, p2)) 
wyniki_miary_pred_test

round(wyniki_miary_pred_train - wyniki_miary_pred_test,4)

# ================================================
# 5. Wizualizacje
# ================================================

cat("\n=== GENEROWANIE WIZUALIZACJI ===\n")

# Rozkład targetu

p1 <- ggplot(
  df,
  aes(x = factor(loan_status))
) +
  geom_bar(
    fill = "steelblue",
    alpha = 0.8
  ) +
  geom_text(
    stat = "count",
    aes(label = after_stat(count)),
    vjust = -0.5
  ) +
  labs(
    title = "Rozkład zmiennej celu (loan_status)",
    x = "Default (1 = tak)",
    y = "Liczba obserwacji"
  ) +
  theme_minimal()

print(p1)

# ================================================
# Boxploty
# ================================================

numeric_selected <- c(
  "person_age",
  "person_income",
  "loan_amnt",
  "loan_int_rate",
  "loan_percent_income"
)

plots_box <- list()

for (var in numeric_selected) {
  
  p <- ggplot(
    df,
    aes(
      x = factor(loan_status),
      y = .data[[var]],
      fill = factor(loan_status)
    )
  ) +
    geom_boxplot(alpha = 0.7) +
    labs(
      title = paste("Rozkład", var, "względem defaultu"),
      x = "Default",
      y = var
    ) +
    theme_minimal() +
    theme(
      legend.position = "none"
    )
  
  plots_box[[var]] <- p
}

for (p in plots_box) {
  print(p)
}

# ================================================
# Histogramy
# ================================================

p_hist <- df %>%
  select(all_of(numeric_selected)) %>%
  pivot_longer(cols = everything()) %>%
  ggplot(aes(x = value)) +
  geom_histogram(
    bins = 30,
    fill = "steelblue",
    color = "white"
  ) +
  facet_wrap(
    ~name,
    scales = "free"
  ) +
  labs(
    title = "Rozkłady zmiennych numerycznych"
  ) +
  theme_minimal()

print(p_hist)

# ================================================
# Default rate wg kategorii
# ================================================

cat_vars <- c(
  "person_home_ownership",
  "loan_intent",
  "cb_person_default_on_file",
  "loan_grade"
)

plots_cat <- list()

for (var in cat_vars) {
  
  if (var %in% names(df)) {
    
    summary_cat <- df %>%
      group_by(.data[[var]]) %>%
      summarise(
        default_rate = mean(loan_status),
        count = n(),
        .groups = "drop"
      )
    
    p <- ggplot(
      summary_cat,
      aes(
        x = .data[[var]],
        y = default_rate,
        fill = .data[[var]]
      )
    ) +
      geom_col(alpha = 0.8) +
      geom_text(
        aes(
          label = paste0(
            round(default_rate * 100, 1),
            "%"
          )
        ),
        vjust = -0.5
      ) +
      labs(
        title = paste(
          "Default rate wg",
          var
        ),
        x = var,
        y = "Default rate"
      ) +
      theme_minimal() +
      theme(
        axis.text.x = element_text(
          angle = 45,
          hjust = 1
        )
      )
    
    plots_cat[[var]] <- p
  }
}

for (p in plots_cat) {
  print(p)
}

# ================================================
# Heatmap korelacji
# ================================================

numeric_df <- df %>%
  select(where(is.numeric))

cor_mat <- cor(
  numeric_df,
  use = "complete.obs"
)

corrplot(
  cor_mat,
  method = "color",
  type = "upper",
  tl.cex = 0.7,
  tl.col = "black",
  title = "Macierz korelacji",
  mar = c(0, 0, 2, 0)
)

# ================================================
# 6. Ewaluacja modelu
# ================================================

evaluate_model <- function(
    model,
    newdata,
    y_true,
    name
) {
  
  pred_prob <- predict(
    model,
    newdata = newdata,
    type = "response"
  )
  
  roc_obj <- roc(
    y_true,
    pred_prob,
    quiet = TRUE
  )
  
  pred_class <- ifelse(
    pred_prob > 0.5,
    1,
    0
  )
  
  cm <- confusionMatrix(
    factor(pred_class, levels = c(0, 1)),
    factor(y_true, levels = c(0, 1)),
    positive = "1"
  )
  
  null_model <- glm(
    loan_status ~ 1,
    data = train_data,
    family = binomial()
  )
  
  mcfadden <- 1 -
    (
      deviance(model) /
        deviance(null_model)
    )
  
  cat("\n========================\n")
  cat(name, "\n")
  cat("========================\n")
  
  cat(
    "AUC:",
    round(auc(roc_obj), 4),
    "\n"
  )
  
  cat(
    "McFadden R²:",
    round(mcfadden, 4),
    "\n\n"
  )
  
  print(cm)
  
  invisible(
    list(
      auc = auc(roc_obj),
      confusion = cm
    )
  )
}

# ================================================
# 7. Wyniki
# ================================================

res_aic <- evaluate_model(
  model_aic,
  test_data,
  test_data$loan_status,
  "MODEL AIC"
)