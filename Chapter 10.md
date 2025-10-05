# Chapter 10: Feature Scaling

### Concept Overview

Imagine you have a dataset with two features: a person's `age` (ranging from 18 to 70) and their `income` (ranging from \\$20,000 to \\$200,000). Many machine learning algorithms, especially those that calculate distances between data points (like K-Nearest Neighbors or SVMs) or use gradient descent (like Linear Regression), would be biased towards the `income` feature simply because its values are much larger. The algorithm would mistakenly think that changes in income are far more important than changes in age.

**Feature scaling** is the process of transforming your numerical features to a common scale, without distorting the differences in the ranges of values or losing information. It's like converting different currencies to a single standard (e.g., US Dollars) before comparing them. This ensures that each feature contributes approximately proportionately to the final result. ⚖️

There are two primary methods of feature scaling we will cover:

1.  **Standardization (`StandardScaler`):** This method rescales the data so that it has a **mean of 0** and a **standard deviation of 1**. The resulting values are not bounded to a specific range.
2.  **Normalization (`MinMaxScaler`):** This method rescales the data to a fixed range, usually **0 to 1**.

-----

### Code Implementation

Just like with imputation, feature scaling must be done **after** the train-test split to prevent data leakage. We will learn the scaling parameters from the training set only and apply that same transformation to the test set.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Sample data with features on different scales
data = {
    'age': [25, 32, 45, 38, 51, 29, 48, 33],
    'salary': [70000, 110000, 85000, 95000, 150000, 72000, 130000, 92000],
    'purchased': [0, 1, 0, 1, 1, 0, 1, 0] # Target variable
}
df = pd.DataFrame(data)

# Separate features (X) and target (y)
X = df[['age', 'salary']]
y = df['purchased']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print("--- Original X_train ---")
print(X_train)
```

#### Standardization with `StandardScaler`

This is the most common scaling technique.

```python
# 1. Create the scaler object
scaler_std = StandardScaler()

# 2. Fit the scaler on the training data ONLY
scaler_std.fit(X_train)

# 3. Transform both the training and test data
X_train_scaled = scaler_std.transform(X_train)
X_test_scaled = scaler_std.transform(X_test)

# The output is a NumPy array, let's view the scaled training data
print("\n--- X_train after Standardization ---")
print(X_train_scaled)
```

#### Normalization with `MinMaxScaler`

This scales the data to a 0-1 range.

```python
# 1. Create the scaler object
scaler_minmax = MinMaxScaler()

# 2. Fit the scaler on the training data ONLY
scaler_minmax.fit(X_train)

# 3. Transform both the training and test data
X_train_normalized = scaler_minmax.transform(X_train)
X_test_normalized = scaler_minmax.transform(X_test)

# View the normalized training data
print("\n--- X_train after Normalization ---")
print(X_train_normalized)
```

-----

### Output Explanation

  * **Standardization:** The `X_train_scaled` output shows our `age` and `salary` features transformed. The values are now centered around 0, and some are negative. For example, the first row's age of 29 is below the average age in the training set, so its scaled value is negative (-1.01).
  * **Normalization:** The `X_train_normalized` output shows the same data transformed differently. All values are now strictly between 0 and 1. The lowest age in the training set (25) is mapped to 0, and the highest age (48) is mapped to 1. All other ages are scaled proportionally in between.

In both cases, the scalers learned the parameters (mean/std for `StandardScaler`, min/max for `MinMaxScaler`) from `X_train` and then applied that *exact same transformation* to `X_test`, preventing any data leakage.

-----

### Practical Notes

  * **Which Scaler Should I Use?**
      * **`StandardScaler`** is the go-to choice for most applications. It's robust and works well with algorithms that assume a zero-centered, Gaussian-like distribution of features. It is also less affected by outliers than `MinMaxScaler`.
      * **`MinMaxScaler`** is useful when you need your feature values to be bounded within a specific range (like 0-1 for image processing or certain neural network architectures). However, it can be sensitive to outliers; a single very large or very small value can skew the scaling of all other data points.
  * **Which Models Need Scaling?** It's crucial to know when scaling is required.
      * **Models that DO need scaling:** Linear Regression, Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Principal Component Analysis (PCA), and Neural Networks.
      * **Models that DO NOT need scaling:** Tree-based algorithms like Decision Trees, Random Forests, and Gradient Boosting. These models work by splitting features based on their rank, not their magnitude, so they are immune to the scale of the data.
  * **Scale Features, Not the Target:** You should only ever apply scaling to your input features (`X`). You should **never** scale your target variable (`y`).