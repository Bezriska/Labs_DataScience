from typing import Callable
from .df import FEATURES
import pandas as pd

features = FEATURES


def delete_over_24_h(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет строки, где суммарное время активностей превышает 24 часа в сутки

    Args:
        df: Датафрейм с признаками Social_Media_Hours, Study_Hours, Sleep_Duration

    Returns:
        Датафрейм без физически невозможных строк
    """
    mask = (df["Social_Media_Hours"] + df["Study_Hours"] + df["Sleep_Duration"]) >= 24
    return df[~mask]


def add_feature_is_person_recovery(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет признак Is_Recovery на основе длительности сна

    Оптимальный сон: 7-9 часов. Значение True означает, что студент
    спит достаточно для полноценного восстановления

    Args:
        df: Датафрейм с признаком Sleep_Duration

    Returns:
        Датафрейм с новым булевым признаком Is_Recovery
    """
    df["Is_Recovery"] = (df["Sleep_Duration"] > 7) & (df["Sleep_Duration"] <= 9)
    return df


def add_feature_is_procrastination(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет признак Is_Procrastination на основе времени в соцсетях

    Прокрастинация определяется как более 5 часов в соцсетях в день

    Args:
        df: Датафрейм с признаком Social_Media_Hours

    Returns:
        Датафрейм с новым булевым признаком Is_Procrastination
    """
    df["Is_Procrastination"] = df["Social_Media_Hours"] > 5
    return df


def add_feature_high_stress(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет признак High_Stress на основе уровня стресса

    Высокий стресс определяется как Stress_Level > 6

    Args:
        df: Датафрейм с признаком Stress_Level

    Returns:
        Датафрейм с новым булевым признаком High_Stress
    """
    df["High_Stress"] = (df["Stress_Level"] > 6)
    return df


def add_feature_low_physical_activity(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет признак Low_Physical_Activity на основе физической активности

    Низкая активность определяется как менее 30 минут в день

    Args:
        df: Датафрейм с признаком Physical_Activity

    Returns:
        Датафрейм с новым булевым признаком Low_Physical_Activity
    """
    df["Low_Physical_Activity"] = df["Physical_Activity"] < 30
    return df


def add_feature_sleep_deficit(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет признак Sleep_Deficit на основе длительности сна

    Дефицит сна определяется как менее 6 часов в сутки

    Args:
        df: Датафрейм с признаком Sleep_Duration

    Returns:
        Датафрейм с новым булевым признаком Sleep_Deficit
    """
    df["Sleep_Deficit"] = df["Sleep_Duration"] < 6
    return df


def add_feature_sleep_stress_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет признак Sleep_Stress_Ratio — отношение сна к стрессу

    При нулевом значении Stress_Level результат заменяется на 0
    во избежание деления на ноль

    Args:
        df: Датафрейм с признаками Sleep_Duration и Stress_Level

    Returns:
        Датафрейм с новым вещественным признаком Sleep_Stress_Ratio
    """
    stress = df["Stress_Level"].replace(0, pd.NA)
    df["Sleep_Stress_Ratio"] = (df["Sleep_Duration"] / stress).fillna(0)
    return df


def add_feature_social_media_plus_study_sleep_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет признак Daily_Hours_Sleep_Ratio

    Вычисляет отношение суммы часов учёбы и соцсетей к продолжительности сна.
    При нулевом Sleep_Duration результат заменяется на 0

    Args:
        df: Датафрейм с признаками Study_Hours, Social_Media_Hours, Sleep_Duration

    Returns:
        Датафрейм с новым вещественным признаком Daily_Hours_Sleep_Ratio
    """
    sleep = df["Sleep_Duration"].replace(0, pd.NA)
    df["Daily_Hours_Sleep_Ratio"] = ((df["Study_Hours"] + df["Social_Media_Hours"]) / sleep).fillna(0)
    return df


def apply_many_feature(df: pd.DataFrame, funcs: list[Callable[[pd.DataFrame], pd.DataFrame]]) -> pd.DataFrame:
    """Последовательно применяет список функций feature engineering к датафрейму

    Args:
        df: Исходный датафрейм
        funcs: Список функций, каждая из которых принимает и возвращает pd.DataFrame

    Returns:
        Датафрейм со всеми добавленными признаками

    Raises:
        TypeError: Если элемент списка не является callable
    """
    for func in funcs:
        if callable(func):
            df = func(df)
        else:
            raise TypeError("Применяемая функция должна быть Callable")
    return df


# === Признаки для логистической регрессии ===

def add_feature_is_high_cgpa(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет признак Is_High_CGPA на основе академической успеваемости

    Высокий CGPA определяется как значение выше 3.4

    Args:
        df: Датафрейм с признаком CGPA

    Returns:
        Датафрейм с новым булевым признаком Is_High_CGPA
    """
    df["Is_High_CGPA"] = df["CGPA"] > 3.4
    return df


def add_feature_stress_physical_act_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет признак Stress_Physical_Ratio — отношение стресса к физической активности

    При нулевом Physical_Activity результат заменяется на 0
    во избежание деления на ноль

    Args:
        df: Датафрейм с признаками Stress_Level и Physical_Activity

    Returns:
        Датафрейм с новым вещественным признаком Stress_Physical_Ratio
    """
    activity = df["Physical_Activity"].replace(0, pd.NA)
    df["Stress_Physical_Ratio"] = (df["Stress_Level"] / activity).fillna(0)
    return df


