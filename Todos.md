Todos

# HIGH PRIORITY

- add hyperparameter tuning

- implement sliding window CV and compare with existing expanding window

- produce plots or summary stats to evaluate chosen methods. Quantify performance variability too
  
- Hybrid modelling (ARIMA + residual ML):
  1. (S)ARIMA baseline: $\hat{y}_t := f(y_{1:t-1})$
     (one-step-ahead forecast from past $y$; can capture trend/seasonality via (S)ARIMA)
  2. residuals: $e_t := y_t - \hat{y}_t$
  3. ML residual model: $\hat{e}_t := h(x_t)$ (train with label $e_t$)
  4. final prediction: $\hat{y}^{\text{final}}_t := \hat{y}_t + \hat{e}_t$

- Stacking (meta-learner):
  - Train $g$ to predict $y_t$ using features $(x_t, \hat{y}_t)$:
    $\hat{y}^{\text{final}}_t := g(x_t, \hat{y}_t)$,
    where $\hat{y}_t$ is obtained from the baseline model in step 1.

- Distillation (teacher–student / surrogate):
  1. teacher: $\hat{y}_t := f(y_{1:t-1}$
  2. student: train $g$ to predict the teacher output from covariates:
     $\tilde{y}_t := g(x_t) \approx \hat{y}_t$ (label is $\hat{y}_t$)



# MEDIUM PRIORITY
- add prophet, arima, sarima
- when adding SARIMAX, linear models, prophet or neural networks, consider correlation-based pruning as a preprocessing step


# LOW PRIORITY
- create endpoints with: a) everything until training -> save pipeline object; b) inference 
