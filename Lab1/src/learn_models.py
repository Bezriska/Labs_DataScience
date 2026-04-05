from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error, f1_score, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


def learn_linear_reg_model(X_train: pd.DataFrame, Y_train: pd.DataFrame) -> LinearRegression:
    """Обучает модель линейной регрессии

    Args:
        X_train: Матрица признаков обучающей выборки
        Y_train: Вектор целевых значений обучающей выборки

    Returns:
        Обученная модель LinearRegression
    """
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return model


def check_linear_model_error_RMSE(
    model: LinearRegression,
    Y_true: pd.DataFrame,
    X_val: pd.DataFrame
) -> float:
    """Вычисляет RMSE для модели линейной регрессии

    Args:
        model: Обученная модель LinearRegression
        Y_true: Реальные значения целевой переменной
        X_val: Матрица признаков для предсказания

    Returns:
        Значение RMSE
    """
    prediction = model.predict(X_val)
    score = root_mean_squared_error(Y_true, prediction)
    return score


def check_linear_model_error_r2(
    model: LinearRegression,
    Y_true: pd.DataFrame,
    X_val: pd.DataFrame
) -> float:
    """Вычисляет R² для модели линейной регрессии

    Args:
        model: Обученная модель LinearRegression
        Y_true: Реальные значения целевой переменной
        X_val: Матрица признаков для предсказания

    Returns:
        Значение R² (от -inf до 1, чем ближе к 1 — тем лучше)
    """
    prediction = model.predict(X_val)
    score = r2_score(Y_true, prediction)
    return score


def check_linear_model_error_MAE(
    model: LinearRegression,
    Y_true: pd.DataFrame,
    X_val: pd.DataFrame
) -> float:
    """Вычисляет MAE для модели линейной регрессии

    Args:
        model: Обученная модель LinearRegression
        Y_true: Реальные значения целевой переменной
        X_val: Матрица признаков для предсказания

    Returns:
        Значение MAE
    """
    prediction = model.predict(X_val)
    score = mean_absolute_error(Y_true, prediction)
    return score


def learn_logistic_reg_model(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame
) -> tuple[LogisticRegression, pd.DataFrame, np.ndarray]:
    """Обучает модель логистической регрессии с предварительным балансированием через SMOTE

    SMOTE применяется только к обучающей выборке, что исключает утечку данных

    Args:
        X_train: Матрица признаков обучающей выборки
        Y_train: Вектор целевых значений обучающей выборки

    Returns:
        Кортеж из обученной модели, признаков и меток после SMOTE
    """
    sm = SMOTE(random_state=52)
    X_resampled, Y_resampled = sm.fit_resample(X_train, Y_train.values.ravel())

    model = LogisticRegression(max_iter=1000)
    model.fit(X_resampled, Y_resampled)
    return model, X_resampled, Y_resampled


def check_logistic_reg_model_error(
    model: LogisticRegression,
    Y_true: pd.DataFrame,
    X_val: pd.DataFrame
) -> float:
    """Вычисляет F1 weighted для модели логистической регрессии

    Args:
        model: Обученная модель LogisticRegression
        Y_true: Реальные значения целевой переменной
        X_val: Матрица признаков для предсказания

    Returns:
        Значение F1 weighted (от 0 до 1, чем ближе к 1 — тем лучше)
    """
    prediction = model.predict(X_val)
    score = f1_score(Y_true.values.ravel(), prediction, average="weighted")
    return score


def cross_validate_logistic_model(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    Y_val: pd.DataFrame,
    cv: int = 5
) -> dict[str, float | np.ndarray]:
    """Проводит k-fold кросс-валидацию логистической регрессии с SMOTE

    SMOTE встроен в Pipeline, поэтому применяется только к тренировочным фолдам
    на каждой итерации, что исключает утечку данных

    Args:
        X_train: Матрица признаков обучающей выборки
        Y_train: Вектор целевых значений обучающей выборки
        X_val: Матрица признаков валидационной выборки
        Y_val: Вектор целевых значений валидационной выборки
        cv: Количество фолдов кросс-валидации

    Returns:
        Словарь с ключами scores (массив), mean (среднее) и std (стандартное отклонение)
    """
    pipeline = Pipeline([
        ("smote", SMOTE(random_state=52)),
        ("model", LogisticRegression(max_iter=1000))
    ])
    X_cv = pd.concat([X_train, X_val])
    Y_cv = pd.concat([Y_train, Y_val])
    scores = cross_val_score(pipeline, X_cv, Y_cv.values.ravel(), cv=cv, scoring="f1_weighted")
    return {"scores": scores, "mean": scores.mean(), "std": scores.std()}

