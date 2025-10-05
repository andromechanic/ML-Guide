# Chapter 9 Handling Missing Data

### Concept Overview

In our first look, we treated missing data as a technical problem to be solved. Now, let's approach it like a detective. Before we can decide *how* to handle missing values, we need to understand *why* they might be missing in the first place. The reason for the missingness can influence which strategy (dropping vs. imputing) is most appropriate.

Broadly, missing data can be categorized into three types:

1.  **Missing Completely at Random (MCAR):** The fact that a value is missing is completely unrelated to any other data. It's a purely random event. For example, a respondent in a survey accidentally skipping a question. This is the easiest case to handle.
2.  **Missing at Random (MAR):** The missingness is related to some other **observed** feature in the dataset. For example, in a health survey, men might be less likely to answer a question about diet. In this case, the missingness in the 'diet\_score' column is not random, but it can be explained by the 'gender' column, which we have.
3.  **Missing Not at Random (MNAR):** The missingness is related to the value itself. For example, people with very high salaries might be less likely to disclose their income. Here, the reason the 'salary' value is missing is because the salary is high. This is the trickiest type of missing data to handle as it can introduce significant bias.

While it's often difficult to know the exact type, thinking about these categories helps you make more informed decisions. In this part, we'll also learn to visualize the pattern of missing data, which can provide valuable clues.

-----

### Code and Analysis

Let's create a DataFrame and then dive into advanced ways of identifying and visualizing where the data is missing.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    'age': [25, 32, np.nan, 45, 38, 29, 51, np.nan],
    'salary': [70000, 110000, 85000, np.nan, 95000, 72000, 150000, 120000],
    'experience_years': [2, 10, 5, 20, 12, 4, 25, 15],
    'department': ['Sales', 'IT', 'Sales', np.nan, 'IT', 'HR', 'Finance', 'IT']
}
df = pd.DataFrame(data)

print("--- Original DataFrame ---")
print(df)
```

#### Advanced Identification

We know `.isnull().sum()` gives us a count. Let's also calculate the percentage of missing data, which is often more useful.

```python
# Count missing values
missing_counts = df.isnull().sum()
print("\n--- Missing Value Counts ---")
print(missing_counts)

# Calculate percentage of missing values
total_rows = len(df)
missing_percentage = (missing_counts / total_rows) * 100
print("\n--- Missing Value Percentage (%) ---")
print(missing_percentage)
```

**Interpretation:** Seeing the percentage is often more insightful. `age` and `salary` are both missing 25% of their values. If this were a very large dataset, a 1% missing rate might be trivial, while a 25% rate is significant and requires careful handling.

-----

#### Visualizing Missing Data

A heatmap is an excellent way to see the pattern of missing data at a glance.

```python
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax)
ax.set_title('Visualizing Missing Data')
plt.show()
```

**Interpretation:** The heatmap provides a clear visual map of our missing data. Each column is represented on the x-axis, and each row on the y-axis. The yellow lines indicate where data is missing (`True`). We can see that row 2 is missing an `age` value, and row 3 is missing both a `salary` and a `department`. This kind of plot is invaluable for spotting patterns in larger datasets.

-----

#### Advanced Dropping with `thresh`

The `.dropna()` method has a useful parameter called `thresh` (threshold), which gives you more control over which rows or columns to drop.

```python
# The thresh parameter specifies the minimum number of NON-missing values required for a row/column to be kept.
# Let's keep rows that have at least 3 non-missing values.
df_thresh = df.dropna(thresh=3)

print("\n--- DataFrame after dropping rows with less than 3 non-NaN values ---")
print(df_thresh)
```

**Interpretation:** Row 3, which had only 2 non-missing values (`experience_years` and its own index), was dropped. Row 2, which had 3 non-missing values, was kept. This is more flexible than the default `.dropna()`, which would have dropped both rows.

-----


We've now taken a more nuanced look at missing data. We understand that the *reason* for missingness matters, and we've learned how to quantify and visualize it more effectively.

  * Calculating the **percentage** of missing data helps us understand the scale of the problem.
  * A **heatmap** gives us an intuitive visual overview of where the gaps in our data are.
  * The `thresh` parameter in `.dropna()` provides more granular control over data removal.

In the next part, we will move beyond simple pandas `fillna()` and learn how to perform imputation in a more structured and robust way using scikit-learn's `SimpleImputer`, which is the standard approach in a machine learning workflow.    


We've now seen *why* data might be missing and how to find it. While pandas' `.fillna()` is great for quick analysis, a proper machine learning workflow requires a more robust and principled approach to imputation. This is where scikit-learn comes in.

-----

## Imputation with Scikit-Learn's SimpleImputer

### Concept Overview

In a machine learning project, one of the most important rules is to **prevent data leakage**. This means that any information from your test set must not be used to prepare or train your model. If you calculate the mean of a column using the entire dataset and then use it to fill missing values before splitting your data, you have "leaked" information from the future (the test set) into your past (the training set). This leads to an unrealistically optimistic evaluation of your model's performance.

To solve this, we need to:

1.  **Split** our data into training and testing sets first.
2.  **Learn** the imputation value (e.g., the mean) from the **training data only**.
3.  **Apply** that learned value to fill missing values in **both** the training and the test sets.

Doing this manually is tedious. Scikit-learn provides a "transformer" object called `SimpleImputer` that automates this process perfectly. It follows the standard `.fit()` and `.transform()` pattern:

  * `.fit()`: Learns the parameter (the mean, median, or mode) from the training data.
  * `.transform()`: Uses the learned parameter to fill the missing values.

This ensures a clean, reproducible, and leak-proof workflow.

-----

### Code and Analysis

First, let's set up our data and, crucially, perform a train-test split.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

data = {
    'age': [25, 32, np.nan, 45, 38, 29, 51, np.nan, 48, 33],
    'salary': [70000, 110000, 85000, np.nan, 95000, 72000, 150000, 120000, 130000, np.nan],
    'department': ['Sales', 'IT', 'Sales', np.nan, 'IT', 'HR', 'Finance', 'IT', 'Finance', 'Sales'],
    'signed_up': [1, 1, 0, 1, 0, 0, 1, 1, 1, 0] # Our dummy target variable
}
df = pd.DataFrame(data)

# Separate features (X) from the target (y)
X = df.drop('signed_up', axis=1)
y = df['signed_up']

# Split the data BEFORE any preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- X_train (before imputation) ---")
print(X_train)
print("\n--- X_test (before imputation) ---")
print(X_test)
```

#### Imputing Numerical Features

We'll use the `mean` strategy for `age` and `salary`.

```python
# 1. Create the imputer object
imputer_mean = SimpleImputer(strategy='mean')

# 2. Fit the imputer on the training data ONLY
imputer_mean.fit(X_train[['age', 'salary']])

# 3. Transform both the training and test data
X_train_imputed_num = imputer_mean.transform(X_train[['age', 'salary']])
X_test_imputed_num = imputer_mean.transform(X_test[['age', 'salary']])

# The output is a NumPy array, let's put it back into a DataFrame to view it
X_train['age'] = X_train_imputed_num[:, 0]
X_train['salary'] = X_train_imputed_num[:, 1]
X_test['age'] = X_test_imputed_num[:, 0]
X_test['salary'] = X_test_imputed_num[:, 1]

print("\n--- X_train (after numerical imputation) ---")
print(X_train)
```

#### Imputing Categorical Features

We'll use the `most_frequent` (mode) strategy for `department`.

```python
# 1. Create the imputer object
imputer_mode = SimpleImputer(strategy='most_frequent')

# 2. Fit and transform on the training data
# Note: scikit-learn expects 2D data, so we use [['department']]
X_train['department'] = imputer_mode.fit_transform(X_train[['department']])

# 3. ONLY transform the test data
X_test['department'] = imputer_mode.transform(X_test[['department']])

print("\n--- Final Imputed X_train ---")
print(X_train)
print("\n--- Final Imputed X_test ---")
print(X_test)
```

-----

### Output Explanation

  * **Initial Split:** Our initial `X_train` has missing values for `age`, `salary`, and `department`. The `X_test` also has a missing `salary`.
  * **Numerical Imputation:** The `NaN` value in `X_train`'s `age` column (row 2) was filled with the mean of the other ages in the training set.
  * **Final Result:** In the final output, all `NaN` values in both `X_train` and `X_test` have been filled. Crucially, the missing salary in `X_test` (row 3) was filled using the mean salary calculated *only from the original 8 training rows*. We have successfully imputed our data without leaking any information from the test set.

-----

### Practical Notes

  * **The `fit_transform` Shortcut:** For the training set, you can combine the `.fit()` and `.transform()` steps into a single, convenient call: `X_train_imputed = imputer.fit_transform(X_train)`. However, you must remember to **only** use `.transform()` on the test set.
  * **Why this is Crucial:** This `fit` on train, `transform` on both methodology is the absolute standard for all preprocessing steps in machine learning (including scaling, which we'll see next). It ensures that your model evaluation is honest and reflects how the model would perform on truly new, unseen data.
  * **Keeping Track of Columns:** Notice that the output of `.transform()` is a NumPy array without column names. When you have many columns, it's important to have a system to put the processed arrays back into a DataFrame with the correct column names. We will explore more advanced methods for this (like `ColumnTransformer`) later in the book.

We've now mastered the standard approach for imputation using `SimpleImputer`. While this is a fast and effective baseline, its main limitation is that it's **univariate**—it only considers the values within the same column to make its guess. In this final part, we'll explore more sophisticated **multivariate** techniques.

-----

## A Look at Advanced Imputation Methods

### Concept Overview

Multivariate imputation is a more advanced strategy that uses the relationships **between** features to make a more educated guess for a missing value. The core idea is that the other columns in your dataset can provide valuable clues about what a missing value is likely to be. 🧠

For example, if you know a person's job title is "Manager" and their years of experience is 15, you can make a much better guess for their missing "salary" than just taking the overall average salary of everyone in the company.

We'll introduce two popular multivariate methods available in scikit-learn:

1.  **K-Nearest Neighbors (KNN) Imputation:** This method finds the 'k' most similar rows (the "nearest neighbors") in the dataset based on the other available features. It then imputes the missing value using the average or mode of those neighbors.
2.  **Iterative Imputation (MICE):** This is a more powerful technique. For a column with missing values, it treats that column as the target `y` and all other columns as features `X`. It then trains a regression model to predict the missing values. It does this for each column with missing values and repeats the process in rounds, refining its imputations until the values stabilize.

These methods can often provide more accurate imputations than simple strategies, especially when the missingness is related to other features (the MAR case).

-----

### Code and Analysis

Advanced imputers require all data to be numerical. Let's create a sample DataFrame and perform a simple numerical encoding on the `department` column first.

```python
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, IterativeImputer

data = {
    'department': ['Sales', 'IT', 'Sales', 'IT', 'HR'],
    'age': [25, 32, np.nan, 45, 29],
    'experience_years': [2, 10, 5, 20, 4]
}
df = pd.DataFrame(data)

# --- Preprocessing: Encode categorical features ---
# Create a mapping from department name to a number
dept_map = {'Sales': 0, 'IT': 1, 'HR': 2}
df['department_encoded'] = df['department'].map(dept_map)
df_numeric = df.drop('department', axis=1)

print("--- DataFrame Ready for Imputation (with NaN) ---")
print(df_numeric)
```

#### K-Nearest Neighbors (KNN) Imputation

Let's fill the missing `age` using the 2 nearest neighbors.

```python
# Create a copy to work with
df_knn = df_numeric.copy()

# 1. Create the KNNImputer object
imputer_knn = KNNImputer(n_neighbors=2)

# 2. Fit and transform the data
df_imputed_knn = imputer_knn.fit_transform(df_knn)

# Convert back to a DataFrame to view
df_imputed_knn = pd.DataFrame(df_imputed_knn, columns=df_knn.columns)

print("\n--- DataFrame after KNN Imputation (k=2) ---")
print(df_imputed_knn)
```

**How was the value calculated?**
The missing age was in row 2. `KNNImputer` looks at the other features (`department_encoded` and `experience_years`) to find the two most similar rows. The two most similar rows to `[department=0, experience=5]` are row 0 `[0, 2]` and row 4 `[2, 4]`. The ages of these two neighbors are 25 and 29. The average is `(25 + 29) / 2 = 27`.

-----

#### Iterative Imputation (MICE)

This method uses a model to predict the missing value.

```python
# Note: IterativeImputer is still considered experimental, so we need a special import
from sklearn.experimental import enable_iterative_imputer

# Create a copy to work with
df_iterative = df_numeric.copy()

# 1. Create the IterativeImputer object
imputer_iterative = IterativeImputer(max_iter=10, random_state=42)

# 2. Fit and transform the data
df_imputed_iterative = imputer_iterative.fit_transform(df_iterative)

# Convert back to a DataFrame to view
df_imputed_iterative = pd.DataFrame(df_imputed_iterative, columns=df_iterative.columns)

print("\n--- DataFrame after Iterative Imputation ---")
print(df_imputed_iterative)
```

-----

### Output Explanation

  * **KNN Imputation:** The `NaN` value for age in row 2 was replaced with `27.0`. As shown in the breakdown, this value is the direct average of the ages of its two "closest" neighbors in the dataset.
  * **Iterative Imputation:** The `NaN` value was replaced with `28.25`. This number wasn't calculated from a simple average. Instead, the imputer trained a model where `age` was the target and `department_encoded` and `experience_years` were the features. The value `28.25` is the prediction from that internal model.

-----

### Practical Notes

  * **Start Simple, Then Get Fancy:** `SimpleImputer` is fast, easy, and often a good-enough solution. Always start there as your baseline. If you have a strong reason to believe that a multivariate approach will be better (e.g., strong correlations in your data) and you are willing to accept the extra computational cost, then you can experiment with `KNNImputer` or `IterativeImputer`.
  * **Preprocessing is Required:** A key takeaway is that these advanced imputers require all data to be numerical. This means you must handle your categorical encoding **before** you can use them, which adds a step to your preprocessing pipeline.
  * **Choosing 'k' for KNN:** The number of neighbors (`k`) is an important parameter. A small `k` might be influenced by noise, while a large `k` might smooth over interesting local patterns. It's often treated as a hyperparameter that you can tune. A value of 3 or 5 is a common starting point.