#  Chapter 6: Data Visualization with Matplotlib and Seaborn


### Concept Overview

In our first look at visualization, we saw how to create basic plots to quickly understand our data. Now, we'll learn how to enhance these plots to extract more information and tailor them to our specific questions. A well-customized plot can reveal subtle patterns and tell a much clearer story. 🎨

This part focuses on two key areas:

1.  **Adding a Third Dimension with `hue`**: We'll explore the `hue` parameter in more detail. This powerful feature allows us to encode a third, categorical variable onto a 2D plot using color. This is one of the easiest ways to add depth to your analysis.
2.  **The Object-Oriented API for Customization**: Instead of using `plt.title()` and `plt.xlabel()`, we will introduce Matplotlib's more powerful **object-oriented (OO) API**. The standard practice is to create a **Figure** (`fig`) and one or more **Axes** (`ax`) objects. The `fig` is the overall window or page, and each `ax` is a specific plot within that figure. This approach gives you much more control and is the standard for any complex visualization.

-----

### Code Implementation

As always, we start with our standard imports and load the "tips" dataset.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure settings for seaborn plots
sns.set_theme(style="whitegrid")

# Load the sample dataset
tips_df = sns.load_dataset('tips')
```

#### Enhancing a Scatter Plot with `hue` and `palette`

Let's revisit our scatter plot of `total_bill` vs. `tip`, but this time, let's see if the relationship differs by the day of the week.

```python
# The object-oriented approach: plt.subplots() returns a figure and an axes object
fig, ax = plt.subplots(figsize=(10, 6))

# Create the scatter plot on the specific axes `ax`
sns.scatterplot(data=tips_df, x='total_bill', y='tip', hue='day', palette='deep', ax=ax)

# Use the `ax` object to set titles and labels (the OO way)
ax.set_title('Bill vs. Tip, Colored by Day of the Week', fontsize=16)
ax.set_xlabel('Total Bill ($)', fontsize=12)
ax.set_ylabel('Tip ($)', fontsize=12)

plt.show()
```

-----

#### Enhancing a Histogram with `hue`

We can also use `hue` on a histogram to compare the distributions of a numerical variable across different categories.

```python
fig, ax = plt.subplots(figsize=(10, 6))

# `multiple="stack"` will stack the bars for each category
sns.histplot(data=tips_df, x='total_bill', hue='time', multiple="stack", palette='viridis', ax=ax)

ax.set_title('Distribution of Total Bills by Meal Time', fontsize=16)
ax.set_xlabel('Total Bill ($)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)

plt.show()
```

-----

### Output Explanation

  * **Enhanced Scatter Plot:** The plot still shows the positive relationship between the bill and the tip. However, by adding `hue='day'`, we've colored each point according to the day of the week it was recorded. This allows us to see if there are day-specific patterns. For example, we can see the cluster of high-value bills from Saturday and Sunday. We also used the `palette='deep'` parameter to select a different color scheme.
  * **Enhanced Histogram:** Instead of one distribution, we now see two stacked distributions for the `total_bill`. The different colors represent the meal time (`Lunch` or `Dinner`). This stacked histogram clearly shows us not only the overall distribution of bills but also that dinner bills tend to be higher and more numerous than lunch bills.

-----

### Practical Notes

  * **Embrace the Object-Oriented API (`fig, ax`)**: While `plt.title()` is quick for a single plot, it becomes cumbersome when you have multiple plots in one figure (subplots). Learning to use the `fig, ax = plt.subplots()` pattern from the beginning is a best practice that will scale to any visualization task you encounter. Think of `ax` as your canvas for a single plot.
  * **Choosing a Palette**: Aesthetics matter. A good color palette can make your chart more readable and impactful. Seaborn comes with many built-in palettes. You can find a full list in the Seaborn documentation. Common choices include `'deep'`, `'muted'`, `'bright'`, `'pastel'`, `'colorblind'`, and sequential palettes like `'viridis'` or `'plasma'` for data with an inherent order.
  * **`hue` is for Categories**: The `hue` parameter is designed to work with **categorical** data (or numerical data with very few unique values). Using `hue` with a continuous variable that has hundreds of unique values will result in an unreadable plot with a messy legend.


Here is the next part of our deep dive into data visualization, where we'll explore more sophisticated plots for understanding distributions and categorical data.

-----

## Visualizing Distributions and Categorical Data

### Concept Overview

While histograms and box plots are the workhorses of distribution analysis, Seaborn provides several other powerful plot types that can reveal deeper insights into your data's structure. These plots help us answer more nuanced questions about how our data is spread out and how it differs across categories.

In this part, we'll explore three advanced plot types:

  * **KDE Plot (`sns.kdeplot`)**: A Kernel Density Estimate plot creates a smooth curve to represent the distribution of a numerical variable. Think of it as a smoothed-out histogram that can be easier to interpret, especially when comparing multiple distributions.
  * **Violin Plot (`sns.violinplot`)**: This powerful plot is a hybrid of a box plot and a KDE plot. It shows the standard summary statistics (like the median and quartiles) of a box plot, but also the full, smoothed distribution of the data on either side. 🎻
  * **Bar Plot (`sns.barplot`)**: A bar plot looks similar to a count plot, but it's used for a different purpose. Instead of showing counts, it displays an **aggregate value** (by default, the mean) of a numerical variable for each category.

-----

### Code Implementation

We'll continue using our standard setup and the "tips" dataset.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
tips_df = sns.load_dataset('tips')
```

#### KDE Plot: For Smooth Distributions

Let's visualize the distribution of tip amounts and compare it between smokers and non-smokers.

```python
fig, ax = plt.subplots(figsize=(10, 6))

sns.kdeplot(data=tips_df, x='tip', hue='smoker', fill=True, common_norm=False, palette='crest', ax=ax)

ax.set_title('Distribution of Tips for Smokers vs. Non-Smokers', fontsize=16)
ax.set_xlabel('Tip ($)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)

plt.show()
```

-----

#### Violin Plot: Combining a Box Plot and KDE Plot

Let's compare the distribution of total bills by day, but with more detail than a standard box plot.

```python
fig, ax = plt.subplots(figsize=(10, 6))

sns.violinplot(data=tips_df, x='day', y='total_bill', hue='sex', split=True, palette='pastel', ax=ax)

ax.set_title('Distribution of Total Bill by Day and Sex', fontsize=16)
ax.set_xlabel('Day of the Week', fontsize=12)
ax.set_ylabel('Total Bill ($)', fontsize=12)

plt.show()
```

-----

#### Bar Plot: Visualizing an Aggregation

This plot doesn't show counts; it shows the result of a calculation (like the mean). What was the average tip amount on each day?

```python
fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(data=tips_df, x='day', y='tip', palette='flare', ax=ax)

ax.set_title('Average Tip Amount by Day of the Week', fontsize=16)
ax.set_xlabel('Day of the Week', fontsize=12)
ax.set_ylabel('Average Tip ($)', fontsize=12)

plt.show()
```

-----

### Output Explanation

  * **KDE Plot:** The resulting plot shows two smooth curves. We can see that the most common tip amount (the peak of the density curve) is slightly lower for smokers. The `fill=True` argument shades the area under the curve, making it easier to see the shape of each distribution.
  * **Violin Plot:** This plot is rich with information. The width of each "violin" represents the density of data points at that bill amount. The white dot inside is the median. The thick black bar is the interquartile range (like the "box" in a box plot). By setting `split=True`, we've shown the distribution for males on one side of the violin and females on the other, allowing for a dense and detailed comparison.
  * **Bar Plot:** The height of each bar represents the **average tip** for that day, not the number of customers. The small black lines at the top of each bar are **confidence intervals**, which give us an idea of the uncertainty in our estimate of the mean. Based on this plot, it appears the average tip is highest on Sunday.

-----

### Practical Notes

  * **Violin Plot vs. Box Plot:** When should you use one over the other?
      * Use a **Box Plot** for a clean, simple summary of key statistics. It's less cluttered.
      * Use a **Violin Plot** when the shape of the distribution is important. For example, a violin plot can reveal if a distribution is bimodal (has two peaks), which a box plot would completely hide.
  * **Bar Plot vs. Count Plot: A CRITICAL Distinction:** 💡
      * `sns.countplot(data=df, x='day')` answers: "How many customers visited each day?"
      * `sns.barplot(data=df, x='day', y='tip')` answers: "What was the **average tip** on each day?"
        Mistaking one for the other is a common beginner error. A bar plot requires both `x` and `y` variables, while a count plot only requires one.
  * **Changing the Estimator:** The bar plot calculates the mean by default. You can change this with the `estimator` parameter. For example, to plot the total (sum) of tips for each day, you would use `estimator=np.sum`.

Here is the final part of our deep dive into data visualization. This section covers powerful plots designed to give you a high-level overview of your entire dataset and the complex relationships within it.

-----

## Relationship and Matrix Plots

### Concept Overview

So far, we've focused on visualizing one or two variables at a time. To get the full picture, however, we need plots that can summarize the relationships across our entire dataset. These "big picture" plots are essential for understanding feature interactions and are a key part of the feature selection process in machine learning.

We will cover three powerful plot types:

  * **Regression Plot (`sns.regplot`)**: This plot goes a step beyond a simple scatter plot by automatically fitting and drawing a linear regression model line. It's the quickest way to visualize the strength and direction of a linear relationship between two variables. 🔗
  * **Pair Plot (`sns.pairplot`)**: This is a fantastic exploratory tool that creates a matrix of plots in a single command. It shows a scatter plot for every pair of numerical variables in your dataset and a histogram or KDE plot for each individual variable along the diagonal.
  * **Heatmap (`sns.heatmap`)**: A heatmap is a graphical representation of a matrix where values are depicted by color. Its most important use in machine learning is to visualize a **correlation matrix**, allowing you to see the correlation between every variable at a single glance. 🔥

-----

### Code Implementation

We'll continue using our standard setup and the "tips" dataset.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
tips_df = sns.load_dataset('tips')
```

#### Regression Plot: Visualizing Linear Relationships

Let's add a regression line to our `total_bill` vs. `tip` scatter plot.

```python
fig, ax = plt.subplots(figsize=(10, 6))

sns.regplot(data=tips_df, x='total_bill', y='tip', ax=ax)

ax.set_title('Regression Plot of Total Bill vs. Tip', fontsize=16)
ax.set_xlabel('Total Bill ($)', fontsize=12)
ax.set_ylabel('Tip ($)', fontsize=12)

plt.show()
```

-----

#### Pair Plot: The Ultimate Exploratory Tool

Let's create a matrix of plots for all numerical variables, colored by the `time` of the meal.

```python
# This one command creates a grid of several plots
sns.pairplot(data=tips_df, hue='time', palette='Set1')

plt.suptitle('Pair Plot of Restaurant Tip Data', y=1.02) # Add a title for the entire figure
plt.show()
```

-----

#### Heatmap: Visualizing a Correlation Matrix

This is a crucial plot for feature selection. First, we calculate the correlation matrix, then we plot it.

```python
# Calculate the correlation matrix for numerical columns only
correlation_matrix = tips_df.corr(numeric_only=True)
print("--- Correlation Matrix ---")
print(correlation_matrix)

# Create the heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)

ax.set_title('Correlation Matrix of Tip Data', fontsize=16)
plt.show()
```

-----

### Output Explanation

  * **Regression Plot:** The plot includes the familiar scatter plot of data points. Added on top is a solid blue line representing the best-fit linear regression model. The shaded light blue area around the line is the **confidence interval** for the regression estimate. This plot visually confirms the strong positive linear relationship between the bill and the tip.
  * **Pair Plot:** The output is a grid of plots.
      * The plots on the **diagonal** are histograms (or KDEs) showing the distribution of each single variable (`total_bill`, `tip`, `size`).
      * The **off-diagonal** plots are scatter plots showing the relationship between every pair of variables. For example, the plot in the first row, second column shows `total_bill` on the y-axis and `tip` on the x-axis. The `hue` parameter has colored each point by meal time.
  * **Heatmap:** The heatmap visualizes the correlation matrix. The `annot=True` parameter writes the correlation coefficient in each cell. The color bar on the right shows the mapping from colors to values.
      * **Interpretation:** Cells with a high positive value (e.g., `total_bill` and `tip` at 0.68) are colored warm (red), indicating a strong positive correlation. Cells with values near 0 are neutral (white), indicating little to no linear correlation. Strong negative correlations would be colored cool (blue).

-----

### Practical Notes

  * **`regplot` vs. `lmplot`**: Seaborn has another, more powerful function called `lmplot()` for creating regression plots. The main difference is that `lmplot` uses its own "FacetGrid" structure, making it very easy to create separate regression plots for different categories (e.g., one plot for smokers and another for non-smokers).
  * **Pair Plots are for Exploration, Not Presentation**: A pair plot is an excellent "first look" tool for a data scientist to quickly scan for relationships. However, it can be dense and overwhelming for a non-technical audience. For presentations, it's better to pick the most interesting relationship you found in the pair plot and create a larger, cleaner, individual plot.
  * **Correlation Is Not Causation:** 💡 This is one of the most important principles in statistics and data science. A heatmap might show that `total_bill` and `tip` are highly correlated. This **does not prove** that a higher bill *causes* a higher tip (though in this case it's likely). Two variables could be correlated because they are both influenced by a third, unobserved "confounding" variable. Always be critical when interpreting correlation.

