from preparing_data_for_model import prepare_data
from learn_models import (learn_linear_reg_model, check_linear_model_error_RMSE,
                          check_linear_model_error_r2, check_linear_model_error_MAE,
                          learn_logistic_reg_model, check_logistic_reg_model_error,
                          cross_validate_logistic_model)
from analythis.df import FEATURES
from sklearn.metrics import classification_report, f1_score
import numpy as np

features = FEATURES.copy()


# === Линейная регрессия ===
print("=== Тест линейной регрессии ===\n")

X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_data(features, ["CGPA"], "linear")
linear_reg_model = learn_linear_reg_model(X_train, Y_train)

val_rmse = check_linear_model_error_RMSE(linear_reg_model, Y_val, X_val)
print(f"RMSE на validation: {val_rmse:.4f}")

test_rmse = check_linear_model_error_RMSE(linear_reg_model, Y_test, X_test)
print(f"RMSE на test: {test_rmse:.4f}")

val_r2_score = check_linear_model_error_r2(linear_reg_model, Y_val, X_val)
print(f"\nR2 score на validation: {val_r2_score:.6f}")

test_r2_score = check_linear_model_error_r2(linear_reg_model, Y_test, X_test)
print(f"R2 score на test: {test_r2_score:.6f}")


test_mae_score = check_linear_model_error_MAE(linear_reg_model, Y_test, X_test)
print(f"\nMAE score на test: {test_mae_score:.6f}")


# === Логистическая регрессия === 
print("=== Тест логистической регрессии ===\n")

X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_data(features, ["Depression"], "logistic")
logistic_reg_model, X_resampled, Y_resampled = learn_logistic_reg_model(X_train, Y_train)

# 1. Сравнение train / val / test
train_f1 = f1_score(Y_resampled, logistic_reg_model.predict(X_resampled), average="weighted")
val_f1 = check_logistic_reg_model_error(logistic_reg_model, Y_val, X_val)
test_f1 = check_logistic_reg_model_error(logistic_reg_model, Y_test, X_test)

print(f"F1 score на train: {train_f1:.6f}")
print(f"F1 score на validation: {val_f1:.6f}")
print(f"F1 score на test: {test_f1:.6f}")
print(f"Разница train-test: {train_f1 - test_f1:.6f} (< 0.05 — нет переобучения)")

# 2. Кросс-валидация (5-fold)
cv_result = cross_validate_logistic_model(X_train, Y_train, X_val, Y_val, cv=5)
print(f"\nКросс-валидация (5-fold):")
print(f"Scores: {np.round(cv_result['scores'], 6)}")
print(f"Mean: {cv_result['mean']:.4f} +- {cv_result['std']:.4f} (малое std — нет переобучения)")






