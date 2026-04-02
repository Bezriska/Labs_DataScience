from preparing_data_for_model import prepare_data
from learn_models import (learn_linear_reg_model, check_linear_model_error_RMSE,
                          check_linear_model_error_r2, check_linear_model_error_MAE)
from analythis.df import FEATURES
features = FEATURES.copy()


# === Линейная регрессия ===

X_train, X_val, X_test, Y_train, Y_val, Y_test = prepare_data(features, ["CGPA"])
linear_reg_model = learn_linear_reg_model(X_train, Y_train)

# Проверка на validation set
val_rmse = check_linear_model_error_RMSE(linear_reg_model, Y_val, X_val)
print(f"RMSE на validation: {val_rmse:.4f}")

# Финальная проверка на test set
test_rmse = check_linear_model_error_RMSE(linear_reg_model, Y_test, X_test)
print(f"RMSE на test: {test_rmse:.4f}")

val_r2_score = check_linear_model_error_r2(linear_reg_model, Y_val, X_val)
print(f"\nR2 score на validation: {val_r2_score:.6f}")

test_r2_score = check_linear_model_error_r2(linear_reg_model, Y_test, X_test)
print(f"R2 score на test: {test_r2_score:.6f}")


test_mae_score = check_linear_model_error_MAE(linear_reg_model, Y_test, X_test)
print(f"\nMAE score на test: {test_mae_score:.6f}")




