from typing import Callable
from .df import FEATURES
import pandas as pd

features = FEATURES

def add_feature_is_person_recovery(df: pd.DataFrame):
    """
    Добавляет признак Is_Recovery на основе длительности сна.
    Оптимальный сон: 7-9 часов.

    Args:
        df (pd.DataFrame): Dataset

    Returns:
        df (pd.DataFrame): Dataset with new feature
    """
    df["Is_Recovery"] = (df["Sleep_Duration"] > 7) & (df["Sleep_Duration"] <= 9)
    return df


def add_feature_is_procrastination(df: pd.DataFrame):
    """Добавляет признак Is_Procrastination на основе количества часов в соц сетях

    Args:
        df (pd.DataFrame): Dataset

    Returns:
        df (pd.DataFrame): Dataset with new feature
    """
    df["Is_Procrastination"] = df["Social_Media_Hours"] > 5
    return df


def add_feature_high_stress(df: pd.DataFrame):
    """Добавляет признак High_Stress

    Args:
        df (pd.DataFrame): Dataset

    Returns:
        pd.DataFrame: Dataset with new feature
    """
    df["High_Stress"] = (df["Stress_Level"] > 6)
    return df


def add_feature_low_physical_activity(df: pd.DataFrame):

    df["Low_Physical_Activity"] = df["Physical_Activity"] < 30
    return df


def add_feature_sleep_deficit(df: pd.DataFrame):

    df["Sleep_Deficit"] = df["Sleep_Duration"] < 6
    return df


def add_feature_sleep_stress_ratio(df: pd.DataFrame):

    stress = df["Stress_Level"].replace(0, pd.NA)
    df["Sleep_Stress_Ratio"] = (df["Sleep_Duration"] / stress).fillna(0)
    return df


def add_feature_social_media_plus_study_sleep_ratio(df: pd.DataFrame):

    sleep = df["Sleep_Duration"].replace(0, pd.NA)
    df["Daily_Hours_Sleep_Ratio"] = ((df["Study_Hours"] + df["Social_Media_Hours"]) / sleep).fillna(0)
    return df


def apply_many_feature(df: pd.DataFrame, funcs: list[Callable]):
    """Применяет последовательно несколько функций для добавления признаков
    
    Args:
        df (pd.DataFrame): Dataset
        funcs (list[Callable]): Список функций для применения
        
    Returns:
        pd.DataFrame: Dataset со всеми новыми признаками
    """
    for func in funcs:
        if callable(func):
            df = func(df)
        else:
            raise TypeError("Применяемая функция должна быть Callable")

    return df

# === Фичи для логистической регрессии ===

def add_feature_is_high_cgpa(df: pd.DataFrame):

    df["Is_High_CGPA"] = df["CGPA"] > 3.4
    return df


def add_feature_stress_physical_act_ratio(df: pd.DataFrame):

    activity = df["Physical_Activity"].replace(0, pd.NA)
    df["Stress_Physical_Ratio"] = (df["Stress_Level"] / activity).fillna(0)
    return df


