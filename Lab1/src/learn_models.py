from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error, f1_score, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

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


def learn_logistic_reg_model(X_train: pd.DataFrame, Y_train: pd.DataFrame):

    sm = SMOTE(random_state=52)
    X_resampled, Y_resampled = sm.fit_resample(X_train, Y_train.values.ravel())

    model = LogisticRegression(max_iter=1000)
    model.fit(X_resampled, Y_resampled)
    return model, X_resampled, Y_resampled


def check_logistic_reg_model_error(model: LogisticRegression, Y_true: pd.DataFrame, X_val: pd.DataFrame):

    prediction = model.predict(X_val)
    score = f1_score(Y_true.values.ravel(), prediction, average="weighted")
    return score


def cross_validate_logistic_model(X_train: pd.DataFrame, Y_train: pd.DataFrame,
                                   X_val: pd.DataFrame, Y_val: pd.DataFrame,
                                   cv: int = 5) -> dict:
    """Проводит кросс-валидацию логистической регрессии с SMOTE.

    Returns:
        dict с ключами: scores, mean, std
    """
    pipeline = Pipeline([
        ("smote", SMOTE(random_state=52)),
        ("model", LogisticRegression(max_iter=1000))
    ])
    X_cv = pd.concat([X_train, X_val])
    Y_cv = pd.concat([Y_train, Y_val])
    scores = cross_val_score(pipeline, X_cv, Y_cv.values.ravel(), cv=cv, scoring="f1_weighted")
    return {"scores": scores, "mean": scores.mean(), "std": scores.std()}

