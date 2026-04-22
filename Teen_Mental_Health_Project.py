#                    Course: INT375

#       TEEN MENTAL HEALTH DATASET - COLLEGE PROJECT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# STEP 1: LOAD THE DATASET

df = pd.read_csv("Teen_Mental_Health_Dataset (1).csv")
print("Data Loaded Successfully\n")
print("First 10 rows:\n", df.head(10), "\n")
print("Last 5 rows:\n", df.tail(), "\n")
print("DataFrame Info:\n")
df.info()
print("\nDescriptive Statistics:\n", df.describe(), "\n")
print("Shape of DataFrame:\n", df.shape, "\n")
print("Column Names:\n", df.columns, "\n")


# STEP 2: CHECKING MISSING VALUES


print("Missing values in each column:\n", df.isnull().sum(), "\n")

# Fill missing values
# continuous -> mean, discrete -> median, categorical -> mode
df_filled = df.fillna({
    'age': df['age'].mean(),
    'sleep_hours': df['sleep_hours'].mean(),
    'daily_social_media_hours': df['daily_social_media_hours'].median(),
    'stress_level': df['stress_level'].median(),
    'platform_usage': df['platform_usage'].mode()[0],
    'social_interaction_level': df['social_interaction_level'].mode()[0]
})
print("After filling missing values:\n", df_filled.head(), "\n")




# STEP 3: REMOVE DUPLICATES

print("Duplicate rows before removal:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicate rows after removal:", df.duplicated().sum(), "\n")


# STEP 4: OUTLIER DETECTION AND HANDLING

numeric_cols = ['daily_social_media_hours', 'sleep_hours', 'screen_time_before_sleep',
                'academic_performance', 'physical_activity',
                'stress_level', 'anxiety_level', 'addiction_level']

# IQR Method
print("IQR Method")
df_clean = df.copy()

for col in numeric_cols:
    Q1 = np.percentile(df_clean[col], 25)
    Q3 = np.percentile(df_clean[col], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
    print(f"{col}: Lower={lower_bound:.2f}, Upper={upper_bound:.2f}, Outliers={len(outliers)}")
    df_clean[col] = df_clean[col].apply(
        lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x)
    )



# STEP 5: GROUP BY AND PIVOT TABLE



grouped_gender = df_clean.groupby('gender')['anxiety_level'].mean()
print("Mean Anxiety Level by Gender:\n", grouped_gender, "\n")
 
grouped_platform = df_clean.groupby('platform_usage')['addiction_level'].mean()
print("Mean Addiction Level by Platform:\n", grouped_platform, "\n")
 
grouped_depression = df_clean.groupby('depression_label')['anxiety_level'].mean()
print("Mean Anxiety Level by Depression Label:\n", grouped_depression, "\n")
 
 
pivot_depression = df_clean.pivot_table(
    values='anxiety_level',
    index='depression_label',
    columns='gender',
    aggfunc='mean'
)
print("Pivot Table - Anxiety Level by Depression Label & Gender:\n", pivot_depression, "\n")
pivot_table = df_clean.pivot_table(
    values='stress_level',
    index='social_interaction_level',
    columns='gender',
    aggfunc='mean'
)
print("Pivot Table - Stress Level by Social Interaction & Gender:\n", pivot_table, "\n")


# STEP 6: SAVE CLEANED DATASET


df_clean.to_csv("Teen_Mental_Health_Cleaned.csv", index=False)
print("Cleaned data saved to: Teen_Mental_Health_Cleaned.csv\n")



# OBJECTIVES


# OBJECTIVE 1 - Visualization (Bar Chart)
# Question: Which social media platform is most popular among teenagers?
# Attribute: platform_usage


print("\n OBJECTIVE 1: Platform Usage Distribution")

platform_count = df_clean['platform_usage'].value_counts()
print("Platform Usage Count:\n", platform_count, "\n")

plt.figure()
plt.bar(platform_count.index, platform_count.values)
plt.title("Platform Usage Among Teenagers")
plt.xlabel("Platform")
plt.ylabel("Number of Teenagers")
plt.show()






# OBJECTIVE 2 - Visualization (Correlation Heatmap)
# Question: Which mental health attributes are most strongly correlated with each other?
# Attribute: all numeric columns


print("OBJECTIVE 2: Correlation Heatmap")


numeric_df = df_clean.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()
print("Correlation Matrix:\n", corr_matrix, "\n")

plt.figure(figsize=(10, 7))
sns.heatmap(corr_matrix, annot=True)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()



# OBJECTIVE 3 - Simple Linear Regression
# Question: Does screen time before sleep predict sleep hours?
# Attributes: screen_time_before_sleep (X) -> sleep_hours (Y)

print("\n OBJECTIVE 3: SLR - Screen Time Before Sleep vs Sleep Hours")


X2 = df_clean[['screen_time_before_sleep']]
y2 = df_clean['sleep_hours']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=0)

model2 = LinearRegression()
model2.fit(X2_train, y2_train)
y2_pred = model2.predict(X2_test)

mse2 = mean_squared_error(y2_test, y2_pred)
r2_2 = r2_score(y2_test, y2_pred)

print(f"Intercept: {model2.intercept_:.2f}")
print(f"Coefficient: {model2.coef_[0]:.2f}")
print(f"MSE: {mse2:.4f}")
print(f"R2 Score: {r2_2:.4f}\n")

plt.figure()
plt.scatter(X2, y2, color='green', alpha=0.4, label='Actual Data')
plt.plot(X2, model2.predict(X2), color='red', linewidth=2, label='Regression Line')
plt.xlabel("Screen Time Before Sleep (hrs)")
plt.ylabel("Sleep Hours")
plt.title("SLR: Screen Time Before Sleep vs Sleep Hours")
plt.legend()
plt.grid(True)
plt.show()

sample2 = pd.DataFrame([[3.0]], columns=['screen_time_before_sleep'])
print(f"Predicted Sleep Hours for 3 hrs screen time: {model2.predict(sample2)[0]:.2f}")



# OBJECTIVE 4 - Z-Test
# Question: Is the average anxiety level of teenagers significantly different from 5?
# Attribute: anxiety_level
# H0: Mean anxiety = 5
# H1: Mean anxiety != 5 (Two-Tailed)


print("\n OBJECTIVE 4: Z-Test - Anxiety Level vs Population Mean 5")


sample_mean = df_clean['anxiety_level'].mean()
population_mean = 5
population_std = df_clean['anxiety_level'].std(ddof=1)
n = len(df_clean['anxiety_level'])
alpha = 0.05

standard_error = population_std / np.sqrt(n)
z_score = (sample_mean - population_mean) / standard_error
p_value = 2 * (1 - norm.cdf(abs(z_score)))

print(f"Sample Mean: {sample_mean:.4f}")
print(f"Population Mean (H0): {population_mean}")
print(f"Z-Score: {z_score:.4f}")
print(f"P-Value: {p_value:.4f}")

if p_value < alpha:
    print("Conclusion: Reject H0. Anxiety level is significantly different from 5.\n")
else:
    print("Conclusion: Fail to reject H0. No significant difference from 5.\n")




# OBJECTIVE 5 - Visualization (Line Plot)
# Question: How does mean stress level change across different ages of teenagers?
# Attribute: age, stress_level
 
print("OBJECTIVE 5: Mean Stress Level by Age")

 
mean_stress_age = df_clean.groupby('age')['stress_level'].mean()
print("Mean Stress Level by Age:\n", mean_stress_age, "\n")
 
plt.figure()
plt.plot(mean_stress_age.index, mean_stress_age.values, marker='o', color='tomato')
plt.title("Mean Stress Level by Age")
plt.xlabel("Age")
plt.ylabel("Mean Stress Level")
plt.grid(True)
plt.show()
