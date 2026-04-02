import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from analythis.feature_engineering import (add_feature_high_stress, add_feature_is_person_recovery,
                                           add_feature_is_procrastination,
                                            add_feature_low_physical_activity, add_feature_sleep_deficit,
                                            apply_many_feature)

funcs = [add_feature_high_stress, add_feature_is_person_recovery,
         add_feature_is_procrastination, add_feature_low_physical_activity, add_feature_sleep_deficit]


def split_data(df: pd.DataFrame, target_variable: list[str]) -> list[pd.DataFrame]:

    X = df.drop(columns=target_variable)
    Y = df[target_variable]
    
    
    X_temp, X_test, Y_temp, Y_test = train_test_split(X, Y, test_size=0.2, random_state=52)
    X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size=0.25, random_state=52)

    return [X_train, X_val, X_test, Y_train, Y_val, Y_test]


def prepare_data(df: pd.DataFrame, target_variable: list[str]) -> list[pd.DataFrame]:

    encoded_df = pd.get_dummies(df, columns=["Gender", "Department"])
    featured_df = apply_many_feature(encoded_df, funcs)

    X_train, X_val, X_test, Y_train, Y_val, Y_test = split_data(featured_df, target_variable)
    
    numeric_cols = ['Age', 'CGPA', 'Sleep_Duration', 'Study_Hours', 
                    'Social_Media_Hours', 'Physical_Activity', 'Stress_Level']
    cols_to_scale = [col for col in numeric_cols if col in X_train.columns]
    
    scaler = StandardScaler()
    scaler.fit(X_train[cols_to_scale])
    
    X_train[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
    X_val[cols_to_scale] = scaler.transform(X_val[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    
    # Применяем fit только на X_train для предотвращения утечки данных
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


    








