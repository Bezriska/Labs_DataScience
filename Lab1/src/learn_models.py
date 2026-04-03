from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
from sklearn.metrics import root_mean_squared_error, f1_score, r2_score, mean_absolute_error
from imblearn.over_sampling import SMOTE

def learn_linear_reg_model(X_train: pd.DataFrame, Y_train: pd.DataFrame) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model

def check_linear_model_error_RMSE(model: LinearRegression, Y_true, X_val) -> float:
    """Возвращает RMSE для модели регрессии"""
    prediction = model.predict(X_val)
    score = root_mean_squared_error(Y_true, prediction)
    return score

def check_linear_model_error_r2(model: LinearRegression, Y_true: pd.DataFrame, X_val: pd.DataFrame) -> float:

    prediction = model.predict(X_val)
    score = r2_score(Y_true, prediction)
    return score

def check_linear_model_error_MAE(model: LinearRegression, Y_true: pd.DataFrame, X_val: pd.DataFrame) -> float:

    prediction = model.predict(X_val)
    score = mean_absolute_error(Y_true, prediction)
    return score


def learn_logistic_reg_model(X_train: pd.DataFrame, Y_train: pd.DataFrame) -> LogisticRegression:

    sm = SMOTE(random_state=52)
    X_resampled, Y_resampled = sm.fit_resample(X_train, Y_train.values.ravel())

    model = LogisticRegression()
    model.fit(X_resampled, Y_resampled)
    return model


def check_logistic_reg_model_error(model: LogisticRegression, Y_true: pd.DataFrame, X_val: pd.DataFrame):

    prediction = model.predict(X_val)
    score = f1_score(Y_true.values.ravel(), prediction, average="weighted")
    return score

