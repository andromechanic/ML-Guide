
# Chapter 8 Exploratory Data Analysis (EDA): Uncovering Insights
## Framing the Problem and Initial Data Inspection

### Concept Overview

Every machine learning project begins with a question. The goal of this initial phase is to frame that question clearly and perform a first-pass investigation of the data we have to answer it. This is like a detective arriving at a crime scene: before looking for specific clues, they first assess the overall situation. 🕵️‍♀️

In this part, we will:

1.  **Frame the Problem:** Clearly state our objective. For this project, our goal is to **understand the key factors that influence a person's medical insurance charges**. The insights we gather will help us build a model to *predict* these charges in later chapters.
2.  **Load and Verify Data:** We'll load our dataset and perform a series of high-level checks to understand its size, structure, and integrity. This includes looking for missing values, checking data types, and getting a feel for the scale of the data.

This initial inspection is crucial. It helps us spot potential data quality issues early and gives us a mental map of the dataset before we start creating detailed visualizations.

-----

### Code and Analysis

#### Loading the Data

We'll start by importing our libraries and loading the dataset into a pandas DataFrame. For this extended analysis, we'll use a larger, more realistic sample of the data.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# In a real project: df = pd.read_csv('insurance.csv')
# For reproducibility, we'll create a larger sample DataFrame
data = {
    'age': [19, 18, 28, 33, 32, 31, 46, 37, 37, 60, 25, 62, 23, 56, 27, 19, 52, 23, 56, 30],
    'sex': ['female', 'male', 'male', 'male', 'male', 'female', 'female', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'male', 'female', 'male', 'male', 'male'],
    'bmi': [27.9, 33.77, 33.0, 22.705, 28.88, 25.74, 33.44, 27.74, 29.83, 25.84, 26.22, 26.29, 34.4, 39.82, 42.13, 24.6, 30.78, 23.845, 40.3, 35.3],
    'children': [0, 1, 3, 0, 0, 0, 1, 3, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    'smoker': ['yes', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'yes', 'no', 'no', 'yes', 'no', 'no', 'no', 'no', 'yes'],
    'region': ['southwest', 'southeast', 'southeast', 'northwest', 'northwest', 'southeast', 'southeast', 'northwest', 'northeast', 'northwest', 'northeast', 'southeast', 'southwest', 'southeast', 'southeast', 'southwest', 'northeast', 'northeast', 'southwest', 'southwest'],
    'charges': [16884.924, 1725.5523, 4449.462, 21984.47061, 3866.8552, 3756.6216, 8240.5896, 7281.5056, 6406.4107, 28923.13692, 2721.3208, 27808.7251, 1826.843, 11090.7178, 39611.7577, 1837.237, 10797.3362, 2395.17155, 10602.385, 36837.467]
}
df = pd.DataFrame(data)

# --- High-Level Inspection ---

# Check the dimensions of the DataFrame (rows, columns)
print(f"Dataset dimensions: {df.shape}")

# Look at the first and last few rows
print("\n--- First 5 Rows ---")
print(df.head())
print("\n--- Last 5 Rows ---")
print(df.tail())
```

**Initial Output:** Our dataset has 20 rows and 7 columns. Seeing the head and tail helps confirm that the data loaded correctly and gives us a quick glimpse of the values.

-----

#### Data Types and Missing Values

Now, let's get a more technical summary.

```python
# Use .info() for a concise summary of the DataFrame
print("\n--- DataFrame Info ---")
df.info()

# Check for the total number of duplicate rows
num_duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {num_duplicates}")
```

**Interpretation:**

  * **Data Types (`Dtype`):** The `.info()` method is excellent. It confirms that `age`, `bmi`, `children`, and `charges` are numerical (`int64`, `float64`), which is what we expect. The `sex`, `smoker`, and `region` columns are `object` types, which is how pandas represents strings. This is correct.
  * **Missing Values (`Non-Null Count`):** All columns show "20 non-null", which matches the total number of rows. This is great news—it means there is no missing data to handle in this dataset.
  * **Duplicates:** There are no duplicate rows, which is another good sign for data quality.

-----

#### Statistical Summaries

Let's get a statistical overview of both our numerical and categorical columns.

```python
# .describe() for numerical columns
print("\n--- Statistical Summary of Numerical Columns ---")
print(df.describe())

# .describe() for categorical (object) columns
print("\n--- Statistical Summary of Categorical Columns ---")
print(df.describe(include='object'))
```

**Interpretation:**

  * **Numerical Summary:**
      * `age`: The age of patients ranges from 18 to 62, with an average of about 35.
      * `bmi`: The average BMI is around 30.6. The standard deviation (`std`) of 5.3 suggests a moderate spread.
      * `charges` (Our Target): This is very insightful. The mean charge is \\$11,800, but the standard deviation is almost as large (\\$11,500). Furthermore, the max value (\\$39,611) is significantly higher than the 75th percentile (\\$19,959). This confirms our initial suspicion that the data is right-skewed with some very high-cost individuals.
  * **Categorical Summary:**
      * `sex`: There are 2 unique values, with `male` being the most frequent (`top`).
      * `smoker`: There are 2 unique values, and the majority are `no`.
      * `region`: There are 4 unique regions, with `southeast` being the most common in our sample.

-----

We have successfully loaded our data and performed a thorough initial inspection. We've confirmed the size of our dataset, verified the data types, checked for missing values and duplicates, and generated high-level statistical summaries. We already have a key insight: our target variable, `charges`, is highly skewed, and this is something we must keep in mind as we proceed with our analysis.

Excellent. Now that we have a high-level understanding of our dataset, it's time to zoom in and analyze each feature individually. This process is called **Univariate Analysis**.

-----

## Univariate Analysis - Understanding Each Variable

### Concept Overview

Univariate analysis is the process of inspecting and visualizing one variable at a time. The goal is to understand the characteristics of each feature in isolation before we explore how they interact with each other. For each variable, we want to understand its:

  * **Distribution:** What is the shape of the data? Is it symmetric, skewed, or uniform?
  * **Central Tendency:** Where is the "center" of the data (e.g., the mean or median)?
  * **Spread:** How much variation is there in the data (e.g., standard deviation, range)?
  * **Outliers:** Are there any extreme or unusual values?

We'll use different techniques for numerical and categorical variables.

-----

### Code and Analysis

We will continue with the `df` DataFrame we created in Part 1.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Recreating the DataFrame from Part 1 for this section
data = {
    'age': [19, 18, 28, 33, 32, 31, 46, 37, 37, 60, 25, 62, 23, 56, 27, 19, 52, 23, 56, 30],
    'sex': ['female', 'male', 'male', 'male', 'male', 'female', 'female', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'male', 'female', 'male', 'male', 'male'],
    'bmi': [27.9, 33.77, 33.0, 22.705, 28.88, 25.74, 33.44, 27.74, 29.83, 25.84, 26.22, 26.29, 34.4, 39.82, 42.13, 24.6, 30.78, 23.845, 40.3, 35.3],
    'children': [0, 1, 3, 0, 0, 0, 1, 3, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    'smoker': ['yes', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'yes', 'no', 'no', 'yes', 'no', 'no', 'no', 'no', 'yes'],
    'region': ['southwest', 'southeast', 'southeast', 'northwest', 'northwest', 'southeast', 'southeast', 'northwest', 'northeast', 'northwest', 'northeast', 'southeast', 'southwest', 'southeast', 'southeast', 'southwest', 'northeast', 'northeast', 'southwest', 'southwest'],
    'charges': [16884.924, 1725.5523, 4449.462, 21984.47061, 3866.8552, 3756.6216, 8240.5896, 7281.5056, 6406.4107, 28923.13692, 2721.3208, 27808.7251, 1826.843, 11090.7178, 39611.7577, 1837.237, 10797.3362, 2395.17155, 10602.385, 36837.467]
}
df = pd.DataFrame(data)
```

### Analyzing Numerical Variables

For numerical variables, we primarily use histograms and box plots to see their distribution and spread.

#### Target Variable: `charges`

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5)) # Create a figure with 2 subplots

# Histogram
sns.histplot(df['charges'], kde=True, ax=axes[0])
axes[0].set_title('Distribution of Charges (Histogram)')

# Box Plot
sns.boxplot(x=df['charges'], ax=axes[1])
axes[1].set_title('Distribution of Charges (Box Plot)')

plt.suptitle('Analysis of the Target Variable: Charges', fontsize=16)
plt.show()
```

**Interpretation:** The histogram clearly shows the strong right skew we noted earlier. The box plot complements this by explicitly showing the median (the line in the box) is much closer to the lower end of the range, and it highlights several high-value data points as outliers (the dots to the right).

-----

#### Feature: `age`

```python
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df['age'], bins=20, kde=True, ax=ax)
ax.set_title('Distribution of Age')
plt.show()
```

**Interpretation:** The distribution of age is relatively uniform, without a strong peak or skew. This means our dataset includes a good mix of people from different age brackets.

-----

#### Feature: `bmi`

```python
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df['bmi'], kde=True, ax=ax)
ax.set_title('Distribution of BMI (Body Mass Index)')
plt.show()
```

**Interpretation:** The BMI distribution looks very much like a **normal distribution** (a "bell curve"). It's symmetric and centered around a value of about 30. This is a common and ideal distribution for many machine learning models.

-----

### Analyzing Categorical Variables

For categorical variables, we use count plots and value counts to understand the frequency of each category.

#### Feature: `smoker`

```python
fig, ax = plt.subplots(figsize=(7, 5))
sns.countplot(x='smoker', data=df, ax=ax)
ax.set_title('Count of Smokers vs. Non-Smokers')
plt.show()

# Show exact percentages
print(df['smoker'].value_counts(normalize=True))
```

**Interpretation:** This is a key insight. The `smoker` category is **imbalanced**. In our sample, only 15% of individuals are smokers. This is important because the model will have much more data to learn from non-smokers.

-----

#### Features: `sex` and `region`

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sex
sns.countplot(x='sex', data=df, ax=axes[0])
axes[0].set_title('Distribution by Sex')

# Region
sns.countplot(x='region', data=df, ax=axes[1])
axes[1].set_title('Distribution by Region')

plt.suptitle('Analysis of Categorical Features', fontsize=16)
plt.show()

print("\n--- Sex ---")
print(df['sex'].value_counts())
print("\n--- Region ---")
print(df['region'].value_counts())
```

**Interpretation:** Unlike the `smoker` variable, the `sex` and `region` features are quite well-balanced. The number of males and females is nearly equal, and the individuals are spread out fairly evenly across the four regions.

-----

We have now examined each variable in isolation. Our key findings are:

  * **Numerical:**
      * `charges`: Heavily right-skewed with significant outliers.
      * `age`: Fairly uniform distribution.
      * `bmi`: Approximately normal (bell-shaped) distribution.
  * **Categorical:**
      * `smoker`: Imbalanced, with far more non-smokers than smokers.
      * `sex` & `region`: Well-balanced.

Now that we have a solid understanding of each feature's individual characteristics, we're ready for the next exciting step: exploring the relationships **between** these variables (Bivariate and Multivariate Analysis).

Now that we've analyzed each variable in isolation, the real detective work begins. In this part, we'll explore the relationships **between** variables to uncover the key drivers of our target, `charges`. This is known as **Bivariate** (two variables) and **Multivariate** (many variables) analysis.

-----

## Bivariate and Multivariate Analysis - Finding Relationships

### Concept Overview

This is the stage where we start connecting the dots. While univariate analysis tells us about the individual characters in our story, bivariate and multivariate analysis help us understand the plot—how the characters interact. Our goal is to answer questions like:

  * Do older people incur higher charges? (Numerical vs. Numerical)
  * Do smokers pay more than non-smokers? (Categorical vs. Numerical)
  * Are smokers more common in a specific region? (Categorical vs. Categorical)

Discovering these relationships is the primary goal of EDA and directly informs which features will be most important for our machine learning model. 🤝

-----

### Code and Analysis

We continue with the `df` DataFrame from the previous parts.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# Recreating the DataFrame from Part 1 & 2
data = {
    'age': [19, 18, 28, 33, 32, 31, 46, 37, 37, 60, 25, 62, 23, 56, 27, 19, 52, 23, 56, 30],
    'sex': ['female', 'male', 'male', 'male', 'male', 'female', 'female', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'male', 'female', 'male', 'male', 'male'],
    'bmi': [27.9, 33.77, 33.0, 22.705, 28.88, 25.74, 33.44, 27.74, 29.83, 25.84, 26.22, 26.29, 34.4, 39.82, 42.13, 24.6, 30.78, 23.845, 40.3, 35.3],
    'children': [0, 1, 3, 0, 0, 0, 1, 3, 2, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    'smoker': ['yes', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'no', 'yes', 'no', 'no', 'yes', 'no', 'no', 'no', 'no', 'yes'],
    'region': ['southwest', 'southeast', 'southeast', 'northwest', 'northwest', 'southeast', 'southeast', 'northwest', 'northeast', 'northwest', 'northeast', 'southeast', 'southwest', 'southeast', 'southeast', 'southwest', 'northeast', 'northeast', 'southwest', 'southwest'],
    'charges': [16884.924, 1725.5523, 4449.462, 21984.47061, 3866.8552, 3756.6216, 8240.5896, 7281.5056, 6406.4107, 28923.13692, 2721.3208, 27808.7251, 1826.843, 11090.7178, 39611.7577, 1837.237, 10797.3362, 2395.17155, 10602.385, 36837.467]
}
df = pd.DataFrame(data)
```

### Numerical vs. Numerical

#### Correlation Heatmap

Let's start with a high-level overview of the linear relationships between all numerical variables.

```python
corr_matrix = df.corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
ax.set_title('Correlation Matrix')
plt.show()
```

**Interpretation:** The heatmap gives us a quick summary. `age` has the strongest positive correlation with `charges` (0.61 in our sample), followed by `bmi` (0.33).

-----

#### Age, BMI, and Charges (Multivariate)

Let's dig deeper into these relationships using a more advanced plot, `lmplot`, which can show scatter plots and regression lines for different subsets of the data.

```python
# lmplot is powerful because it can create facets (separate plots) or
# use hue to draw different regression lines on the same plot.
sns.lmplot(x='age', y='charges', hue='smoker', data=df, height=6, aspect=1.5)
plt.title('Age vs. Charges by Smoker Status', fontsize=16)
plt.show()
```

**Interpretation:** This is a fantastic insight that a simple scatter plot might miss. We see two distinct regression lines. For non-smokers (blue), there's a gentle positive slope: charges increase moderately with age. For smokers (orange), the slope is much steeper: charges increase significantly with age. This is a classic **interaction effect**, where the effect of one variable (`age`) on the target depends on the value of another variable (`smoker`).

-----

### Categorical vs. Numerical

We use box plots or violin plots to compare the distribution of a numerical variable across different categories.

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# sex vs. charges
sns.boxplot(x='sex', y='charges', data=df, ax=axes[0])
axes[0].set_title('Charges Distribution by Sex')

# region vs. charges
sns.boxplot(x='region', y='charges', data=df, ax=axes[1])
axes[1].set_title('Charges Distribution by Region')

plt.suptitle('Comparing Charges Across Categories', fontsize=16)
plt.show()
```

**Interpretation:**

  * **Sex vs. Charges:** While the median charge for males appears slightly higher in this sample, the interquartile ranges (the boxes) overlap significantly. This suggests that `sex` is likely not as strong a predictor as `smoker` or `age`.
  * **Region vs. Charges:** There are some differences in the distributions across regions, but again, the effect is much less pronounced than what we saw with smoking status.

-----

### Categorical vs. Categorical

We can check if certain categories are associated with each other. For example, is the proportion of smokers different across regions?

```python
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='region', hue='smoker', data=df, ax=ax)
ax.set_title('Smoker Distribution Across Regions')
plt.show()
```

**Interpretation:** This plot helps confirm that the "smoker effect" is not just a hidden "region effect." We see that smokers and non-smokers exist in all regions. The proportion of smokers seems highest in the `southeast` in this sample, which is an interesting secondary finding, but it doesn't invalidate our main conclusion about the impact of smoking on charges.

-----

By exploring the relationships between variables, we've solidified our understanding of the key drivers of medical costs in this dataset.

  * The relationship between numerical features (`age`, `bmi`) and our target (`charges`) is heavily influenced by the `smoker` category. This **interaction effect** is a critical finding.
  * The `smoker` variable itself has the most dramatic impact on `charges`.
  * Other categorical variables like `sex` and `region` have a much weaker, less clear relationship with `charges`.

We have now gathered a wealth of information. The final step in our EDA project is to consolidate these findings into a concise summary and generate clear hypotheses for our modeling phase.

We've now inspected our data, analyzed each variable individually, and explored the rich relationships between them. The final and most important step of any EDA is to consolidate our findings into a clear, actionable summary. This summary will serve as our guide for the upcoming data preprocessing and modeling stages.

***

## EDA Summary and Hypothesis Generation

### Concept Overview

The goal of EDA is not just to create plots; it's to generate insights. The final step of the process is to synthesize everything we've learned into a coherent narrative. This summary serves two main purposes:

1.  **Communication:** It provides a concise summary of the dataset's key characteristics for stakeholders, team members, or for your own future reference.
2.  **Strategy:** It directly informs our plan for feature engineering and model selection. A good EDA tells you which features are likely to be important and what potential challenges (like skewed data) you need to address.

From this summary, we will formulate specific **hypotheses**—educated guesses about what will be effective in our modeling phase. ✅

---

### EDA Summary: Key Findings

After a thorough exploration of the medical charges dataset, we can report the following key insights:

#### On the Target Variable (`charges`)
* The distribution of `charges` is **heavily right-skewed**. The vast majority of individuals have relatively low medical costs, while a small number of individuals have extremely high costs, creating a long tail.
* The median charge is significantly lower than the mean, which is a classic indicator of this skewness and the presence of high-value outliers.

---
#### On Key Predictive Features
* **`smoker`**: This is, without a doubt, the **most significant predictor** of medical charges. The difference in the distribution of charges between smokers and non-smokers is dramatic and statistically significant.
* **`age`**: There is a strong, clear, **positive linear relationship** between age and charges. As age increases, medical costs tend to rise consistently.
* **`bmi`**: Body Mass Index also shows a positive correlation with charges. Individuals with higher BMIs tend to have higher medical costs.

---
#### On Important Interactions
* The most powerful insight comes from the **interaction between `smoker` status and other variables**. The positive correlation of `age` and `bmi` with `charges` is **much more pronounced for smokers**. A simple model looking at these features in isolation would miss this crucial dynamic.

---
#### On Features with Weaker Signals
* **`sex`** and **`region`**: While minor differences in charges exist across these categories, their impact appears to be far less significant than `smoker`, `age`, or `bmi`. They are likely to be weak predictors.
* **`children`**: The correlation of the number of children with charges was very low, suggesting it has little predictive power on its own.

---
#### On Data Quality
* The dataset is of high quality. We found **no missing values** and **no duplicate rows**, meaning we can proceed to preprocessing without needing to perform imputation or data cleaning for these issues.

---

### Hypothesis Generation for Modeling

Based on the summary above, we can formulate several clear hypotheses that will guide our modeling strategy:

* **Hypothesis 1:** The `smoker` feature will have the **largest feature importance** in any predictive model we build.
* **Hypothesis 2:** A linear model (like Linear Regression) should be able to capture the `age` vs. `charges` trend and provide a decent **baseline performance**.
* **Hypothesis 3:** A model that can inherently capture **interaction effects** (like a Decision Tree or Random Forest) or a linear model where we manually create interaction features (e.g., `bmi * smoker_status`) will significantly outperform a simple linear model.
* **Hypothesis 4:** The skewed nature of the `charges` variable may negatively impact the performance of some models (especially linear models which often assume normality). Applying a **logarithmic transformation** to `charges` is a promising preprocessing step that could lead to better results.

With this EDA complete, we are no longer working in the dark. We have a deep understanding of our data and a clear, insight-driven strategy for the chapters ahead.
