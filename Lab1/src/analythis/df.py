import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display



path = '/Users/maxim/.cache/kagglehub/datasets/aldinwhyudii/student-depression-and-lifestyle-100k-data/versions/1'

df = pd.read_csv(f'{path}/student_lifestyle_100k.csv')

DATAFRAME = df

# Удаляем Student_ID из-за ненадобности
FEATURES = df.drop(columns=["Student_ID"])

def sns_plot(df: pd.DataFrame, feature: str, bins: int = 2, color: str = "blue", is_bool: bool = False):
    plt.figure(figsize=(10, 5))

    display(df[feature].describe())

    sns.histplot(
        data=df,
        x=feature,
        bins=bins,
        color=color,
        edgecolor="black"
    )

    plt.title(f"Распределение признака '{feature}'", fontsize=16)
    plt.xlabel(feature, fontsize=16)
    plt.ylabel("Количество", fontsize=16)
    if is_bool:
        plt.xticks([0, 1], ["False", "True"])

    plt.show()

"""
Data columns (total 11 columns):
 #   Column              Non-Null Count   Dtype  
---  ------              --------------   -----  
 0   Student_ID          100000 non-null  int64  
 1   Age                 100000 non-null  int64  
 2   Gender              100000 non-null  str    
 3   Department          100000 non-null  str    
 4   CGPA                100000 non-null  float64
 5   Sleep_Duration      100000 non-null  float64
 6   Study_Hours         100000 non-null  float64
 7   Social_Media_Hours  100000 non-null  float64
 8   Physical_Activity   100000 non-null  int64  
 9   Stress_Level        100000 non-null  int64  
 10  Depression          100000 non-null  bool   
dtypes: bool(1), float64(4), int64(4), str(2)
memory usage: 7.7 MB
"""

# print(FEATURES.isna().describe())
# print(FEATURES.duplicated().describe())
# print(df.duplicated(subset=["Student_ID"]).describe())

# ВЫВОД: нет дубликатов и пустых


    # gender_stats = FEATURES.groupby('Gender').agg({
    #     'CGPA': 'mean',
    #     'Study_Hours': 'mean',
    #     'Sleep_Duration': 'mean',
    #     'Social_Media_Hours': 'mean',
    #     'Stress_Level': 'mean'
    # }).reset_index()

    # gender_long = gender_stats.melt(id_vars='Gender', 
    #                                  var_name='Metric', 
    #                                  value_name='Value')

    # plt.figure(figsize=(12, 6))
    # sns.barplot(data=gender_long, x='Metric', y='Value', hue='Gender', palette='Set2')
    # plt.title('Сравнение показателей по полу', fontsize=14)
    # plt.xlabel('Показатели', fontsize=12)
    # plt.ylabel('Среднее значение', fontsize=12)
    # plt.xticks(rotation=45, ha='right')
    # plt.legend(title='Пол')
    # plt.tight_layout()
    # plt.show()

    # Вывод: распределение одинаковое



# total = FEATURES["Social_Media_Hours"] + FEATURES["Sleep_Duration"] + FEATURES["Study_Hours"]
# print((total >= 24).sum())

# Вывод: больше 24 часов в сутках есть у 89 человек -> удаляем