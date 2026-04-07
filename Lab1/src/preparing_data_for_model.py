import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from analythis.feature_engineering import (add_feature_high_stress, add_feature_is_person_recovery,
                                           add_feature_is_procrastination,
                                            add_feature_low_physical_activity, add_feature_sleep_deficit,
                                            apply_many_feature, add_feature_sleep_stress_ratio,
                                            add_feature_social_media_plus_study_sleep_ratio, add_feature_is_high_cgpa,
                                            add_feature_stress_physical_act_ratio, delete_over_24_h)

funcs = [add_feature_high_stress, add_feature_is_person_recovery,
         add_feature_is_procrastination, add_feature_low_physical_activity,
         add_feature_sleep_deficit, add_feature_sleep_stress_ratio, add_feature_social_media_plus_study_sleep_ratio]


def split_data(df: pd.DataFrame, target_variable: list[str]) -> list[pd.DataFrame]:
    """Разбивает датафрейм на выборки train, val, test в соотношении 60/20/20

    Args:
        df: Исходный датафрейм с признаками и целевой переменной
        target_variable: Список названий целевых столбцов

    Returns:
        Список [X_train, X_val, X_test, Y_train, Y_val, Y_test]
    """
    X = df.drop(columns=target_variable)
    Y = df[target_variable]

    X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.2, random_state=52)
    X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.25, random_state=52)

    return [X_train, X_val, X_test, Y_train, Y_val, Y_test]


def prepare_data(
    df: pd.DataFrame,
    target_variable: list[str],
    reg_type: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Полный пайплайн подготовки данных: кодирование, feature engineering, разбивка, масштабирование

    StandardScaler обучается только на X_train, что исключает утечку данных
    Для reg_type="logistic" дополнительно добавляются Is_High_CGPA и Stress_Physical_Ratio

    Args:
        df: Исходный датафрейм
        target_variable: Список названий целевых столбцов
        reg_type: Тип модели: "linear" или "logistic"

    Returns:
        Кортеж (X_train, X_val, X_test, Y_train, Y_val, Y_test)
    """
    deleted_24h = delete_over_24_h(df)
    encoded_df = pd.get_dummies(deleted_24h, columns=["Gender", "Department"])
    featured_df = apply_many_feature(encoded_df, funcs)
    if reg_type == "logistic":
        featured_df = add_feature_is_high_cgpa(featured_df)
        featured_df = add_feature_stress_physical_act_ratio(featured_df)

    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(featured_df, target_variable)

    numeric_cols = ['Age', 'CGPA', 'Sleep_Duration', 'Study_Hours',
                    'Social_Media_Hours', 'Physical_Activity', 'Stress_Level']
    cols_to_scale = [col for col in numeric_cols if col in X_train.columns]

    scaler = StandardScaler()
    scaler.fit(X_train[cols_to_scale])

    X_train[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
    X_val[cols_to_scale] = scaler.transform(X_val[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


    








