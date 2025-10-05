
# Chapter 5: Pandas for Data Manipulation and Analysis

### Concept Overview

While NumPy is the perfect tool for handling numerical arrays, most real-world data isn't just a grid of numbers. It has labels, mixed data types, and is often messy. This is where **pandas** shines. Pandas is the most popular Python library for data cleaning, manipulation, and analysis. It's built on NumPy, so it's fast, but it provides a much more flexible and user-friendly interface for working with tabular data.

The name "pandas" is derived from "panel data," an econometrics term for multidimensional datasets. The library introduces two core data structures that you will use constantly:

  * **Series:** A one-dimensional, labeled array. Think of it as a single column in a spreadsheet with a name.
  * **DataFrame:** A two-dimensional, labeled data structure, much like a spreadsheet or a SQL table. It's the primary object in pandas. A DataFrame is essentially a collection of Series that share the same index. 📊

For the vast majority of machine learning projects, your first step will be to load your data into a pandas DataFrame.

-----

### Code Implementation

By convention, pandas is always imported with the alias `pd`.

```python
import pandas as pd
import numpy as np
```

#### Creating a DataFrame

You can create a DataFrame from many sources, including a Python dictionary.

```python
# Create a dictionary where keys are column names and values are lists of data
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 32, 28, 45],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston'],
    'Salary': [70000, 110000, 85000, 130000]
}

# Create the DataFrame
df = pd.DataFrame(data)

print("DataFrame created from a dictionary:")
print(df)
```

#### Inspecting the Data

Once you have a DataFrame (usually by loading it from a file), these are the first commands you'll run.

```python
# Display the first 3 rows
print("\n--- First 3 Rows (.head(3)) ---")
print(df.head(3))

# Get a concise summary of the DataFrame
print("\n--- DataFrame Info (.info()) ---")
df.info()

# Get quick statistical summary of numerical columns
print("\n--- Statistical Summary (.describe()) ---")
print(df.describe())

# Get the dimensions of the DataFrame (rows, columns)
print(f"\nDataFrame shape: {df.shape}")
```

#### Selecting Data (Columns and Rows)

This is one of the most fundamental skills in pandas.

```python
# --- Selecting Columns ---

# Select a single column (returns a pandas Series)
ages = df['Age']
print("\n--- Selecting the 'Age' column (a Series) ---")
print(type(ages))
print(ages)

# Select multiple columns (returns a new DataFrame)
# Note the double square brackets [[...]]
subset = df[['Name', 'Salary']]
print("\n--- Selecting 'Name' and 'Salary' columns (a DataFrame) ---")
print(subset.head())


# --- Selecting Rows ---

# .loc[] is used for label-based indexing
# Get the row with index label 2
print("\n--- Selecting row with index 2 (.loc[2]) ---")
print(df.loc[2])

# .iloc[] is used for integer-position-based indexing
# Get the row at the 3rd position (index 2)
print("\n--- Selecting row at position 2 (.iloc[2]) ---")
print(df.iloc[2])
```

-----

### Output Explanation

  * **Creating a DataFrame:** The output shows a neatly formatted table. The dictionary keys became the column headers, and the lists of values filled the rows. The bold numbers on the left (0, 1, 2, 3) are the **Index**, which pandas creates automatically to label the rows.
  * **Inspecting the Data:**
      * `.head(3)` displays the first three records, which is useful for getting a quick peek at the data without printing the entire table.
      * `.info()` provides a technical summary. It tells us we have 4 entries (rows), 4 columns, the name and data type (`Dtype`) of each column, and that there are no missing (`non-null`) values.
      * `.describe()` automatically calculates key statistics like the mean, standard deviation (`std`), min, and max for all numerical columns (`Age` and `Salary`).
  * **Selecting Data:**
      * When we select a single column with `df['Age']`, the output is a **Series**.
      * When we select multiple columns with `df[['Name', 'Salary']]`, the output is a smaller **DataFrame**.
      * `.loc[2]` and `.iloc[2]` give the same result in this case because our index labels are the same as the integer positions. The output is a Series representing all the data in that row.

-----

### Practical Notes

  * **Your Go-To Tool for EDA:** Pandas is the primary tool for **Exploratory Data Analysis (EDA)**. The vast majority of your time at the beginning of a project will be spent in a Jupyter Notebook using pandas to load, clean, visualize, and understand your dataset.
  * **Reading from Files is More Common:** While we created a DataFrame manually, you will almost always load data from a file. The most common function is `pd.read_csv('your_file.csv')`, which is powerful and has many options for handling different file formats.
  * **The Index is Powerful:** The row labels on the left (the **Index**) are more than just labels. They are used by pandas to align data during operations, which is a powerful feature we'll explore later. For now, just know that every row and column in a pandas object has a label.



## Common DataFrame Operations

### Concept Overview

Once you've loaded and inspected your data, the real work of data wrangling begins. This involves cleaning the data, transforming it, and extracting insights. Pandas provides a rich set of functions to perform these tasks efficiently.

In this section, we will cover the essential operations that form the backbone of nearly every data analysis project:

  * **Conditional Filtering:** Selecting rows that meet specific criteria (e.g., all employees with a salary over $90,000).
  * **Creating and Modifying Columns:** Adding new columns derived from existing data (e.g., creating a "Salary in Thousands" column).
  * **Handling Missing Data:** Finding and deciding how to treat missing values (`NaN`), which are common in real-world datasets.
  * **Grouping and Aggregating:** The powerful "split-apply-combine" pattern using `.groupby()` to summarize data by category (e.g., calculating the average salary per department).

-----

### Code Implementation

Let's create a new sample DataFrame for these examples. This one includes a categorical column and a missing value to make our scenarios more realistic.

```python
import pandas as pd
import numpy as np

data = {
    'Department': ['Sales', 'IT', 'Sales', 'HR', 'IT', 'HR'],
    'Employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank'],
    'YearsExperience': [5, 10, 3, 8, 4, np.nan],
    'Salary': [70000, 110000, 65000, 95000, 82000, 78000]
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)
print("-" * 35)
```

#### Conditional Filtering

Use boolean expressions inside square brackets `[]` to filter rows.

```python
# Single condition: Find all employees in the 'IT' department
it_dept = df[df['Department'] == 'IT']
print("\n--- Employees in the IT Department ---")
print(it_dept)

# Multiple conditions: Find employees in Sales with a salary > $68,000
# Use & for AND, | for OR. Each condition must be in parentheses.
high_earning_sales = df[(df['Department'] == 'Sales') & (df['Salary'] > 68000)]
print("\n--- High-Earning Sales Employees ---")
print(high_earning_sales)
```

#### Creating New Columns

You can create new columns by simple assignment.

```python
# Create a new column 'Salary_in_Thousands'
df['Salary_in_Thousands'] = df['Salary'] / 1000

# Create a 'Seniority' column based on a condition
# np.where is a great tool for this: np.where(condition, value_if_true, value_if_false)
df['Seniority'] = np.where(df['YearsExperience'] >= 5, 'Senior', 'Junior')

print("\n--- DataFrame with New Columns ---")
print(df)
```

#### Handling Missing Data

First find, then decide whether to drop or fill.

```python
# Check for missing values in each column
print("\n--- Count of Missing Values ---")
print(df.isnull().sum())

# Option 1: Drop rows with any missing values
df_dropped = df.dropna()
print("\n--- DataFrame after dropping NaNs ---")
print(df_dropped)

# Option 2: Fill missing values (e.g., with the mean of the column)
mean_experience = df['YearsExperience'].mean()
df_filled = df.fillna(value={'YearsExperience': mean_experience})
print("\n--- DataFrame after filling NaNs with the mean ---")
print(df_filled)
```

#### Grouping and Aggregating (`.groupby()`)

This is one of the most powerful features of pandas.

```python
# Calculate the average salary for each department
# This splits the data by department, calculates the mean for each group, and combines the results.
avg_salary_by_dept = df.groupby('Department')['Salary'].mean()

print("\n--- Average Salary by Department ---")
print(avg_salary_by_dept)

# You can perform other aggregations too, like .sum(), .count(), .max(), etc.
employee_count_by_dept = df.groupby('Department')['Employee'].count()
print("\n--- Count of Employees by Department ---")
print(employee_count_by_dept)
```

-----

### Output Explanation

  * **Conditional Filtering:** The outputs are new DataFrames containing only the rows that satisfied our boolean conditions. Notice how the original index is preserved.
  * **Creating New Columns:** The DataFrame is now wider, with two new columns. `'Salary_in_Thousands'` was calculated for every row. The `'Seniority'` column was populated based on the `YearsExperience` for each employee. Note that Frank is 'Junior' because his experience is `NaN`, which doesn't satisfy the `>= 5` condition.
  * **Missing Data:** `.isnull().sum()` shows us that `YearsExperience` has 1 missing value. The `df_dropped` output is missing the row for Frank. In `df_filled`, Frank's `NaN` value for experience has been replaced with `6.0`, which is the calculated mean of the other values in that column.
  * **Grouping:** The output of `.groupby()` is a new Series. The **Index** of the Series contains the unique group names (`HR`, `IT`, `Sales`), and the **values** are the aggregated results (e.g., the mean salary or the employee count) for each group.

-----

### Practical Notes

  * **The Power of `groupby`:** The "split-apply-combine" pattern enabled by `.groupby()` is a cornerstone of data analysis. You can split your data into any number of logical groups, apply virtually any function to them (mean, sum, count, max, etc.), and combine the results into a neat summary.
  * **`inplace=True`:** Many pandas methods, like `dropna()` and `fillna()`, have an `inplace=True` parameter. If you set it, the DataFrame will be modified directly, and nothing will be returned. It's often safer, especially for beginners, to avoid `inplace=True` and instead assign the result to a new variable (e.g., `df_clean = df.dropna()`). This prevents you from accidentally losing raw data.
  * **Vectorization is Still Key:** Operations like creating the `'Salary_in_Thousands'` column are vectorized. Pandas applies the division to the entire `'Salary'` Series at once, which is far more efficient than iterating through rows with a Python `for` loop. Always look for a vectorized pandas solution before writing a loop.



## Combining DataFrames and Advanced Operations

### Concept Overview

In real-world projects, your data is often spread across multiple files. You might have one file with employee information and another with department details. To perform a meaningful analysis, you need to combine them. Pandas provides two primary methods for this, inspired by database operations:

  * **Concatenating (`pd.concat`)**: This is like stacking papers on top of one another. It's used to append rows from one DataFrame to another. 🖇️
  * **Merging (`pd.merge`)**: This is a more sophisticated operation, similar to a `JOIN` in SQL. It combines DataFrames by linking rows that share a common value in a key column (e.g., matching an `employee_id` across two tables).

Additionally, we'll cover other essential operations for data exploration and transformation:

  * **Finding Unique Values and Counts:** Quickly summarizing the categories within a column.
  * **Applying Custom Functions (`.apply`)**: Performing complex, row-by-row transformations that go beyond simple arithmetic.

-----

### Code Implementation

Let's set up a few sample DataFrames to work with.

```python
import pandas as pd

# DataFrame with employee information
employees_df = pd.DataFrame({
    'employee_id': ['E1', 'E2', 'E3', 'E4', 'E5'],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'dept_id': ['D1', 'D2', 'D1', 'D3', 'D2']
})

# DataFrame with department information
departments_df = pd.DataFrame({
    'dept_id': ['D1', 'D2', 'D3', 'D4'],
    'dept_name': ['Sales', 'IT', 'HR', 'Finance']
})

# A second DataFrame of employees for concatenation
new_hires_df = pd.DataFrame({
    'employee_id': ['E6', 'E7'],
    'name': ['Frank', 'Grace'],
    'dept_id': ['D1', 'D4']
})
```

#### Concatenating DataFrames

This is used to stack DataFrames.

```python
# Stack the original employees and new hires DataFrames
all_employees_df = pd.concat([employees_df, new_hires_df], ignore_index=True)

print("--- Concatenated DataFrame ---")
print(all_employees_df)
```

#### Merging (Joining) DataFrames

This is used to combine DataFrames based on a common key.

```python
# Merge employees with their department names
# 'on' specifies the key column. 'how='inner'' means only keep rows where
# the dept_id exists in BOTH DataFrames.
employee_depts_df = pd.merge(employees_df, departments_df, on='dept_id', how='inner')

print("\n--- Merged (Inner Join) DataFrame ---")
print(employee_depts_df)
```

#### Unique Values and Value Counts

These are essential for exploring categorical columns.

```python
# Use the merged DataFrame for this example
print("\n--- Unique Values and Counts ---")

# Get the number of unique departments
num_unique_depts = employee_depts_df['dept_name'].nunique()
print(f"Number of unique departments: {num_unique_depts}")

# Get the array of unique department names
unique_depts_array = employee_depts_df['dept_name'].unique()
print(f"Unique department names: {unique_depts_array}")

# Get the frequency of each department
dept_counts = employee_depts_df['dept_name'].value_counts()
print("\nEmployee counts per department:")
print(dept_counts)
```

#### Applying a Custom Function (`.apply`)

Use this when you need to perform a complex operation on a column.

```python
# Let's categorize employees based on the length of their name
def name_length_category(name):
    if len(name) > 5:
        return 'Long Name'
    else:
        return 'Short Name'

# Apply this function to the 'name' column to create a new column
employee_depts_df['name_category'] = employee_depts_df['name'].apply(name_length_category)

# You can achieve the same thing with a concise lambda function
# employee_depts_df['name_category'] = employee_depts_df['name'].apply(lambda x: 'Long Name' if len(x) > 5 else 'Short Name')

print("\n--- DataFrame after using .apply() ---")
print(employee_depts_df)
```

-----

### Output Explanation

  * **Concatenating:** The `all_employees_df` DataFrame contains all the rows from both `employees_df` and `new_hires_df`, one stacked on top of the other. The `ignore_index=True` argument re-created a clean index from 0 to 6.
  * **Merging:** The `employee_depts_df` now includes the `dept_name` for each employee. Pandas matched the rows where the `dept_id` was the same in both original DataFrames. Notice that David (dept\_id D3) is included, but the Finance department (dept\_id D4) is not, because no employee was assigned to it in the `employees_df`.
  * **Unique Values and Counts:** The code correctly identifies 3 unique departments represented in our merged data. The `value_counts()` output is a Series showing that the Sales and IT departments each have 2 employees, and HR has 1, sorted by frequency.
  * **Applying a Function:** The final DataFrame has a new `name_category` column. For each row, pandas took the value in the `name` column, passed it to our `name_length_category` function, and placed the returned value ('Long Name' or 'Short Name') into the new column.

-----

### Practical Notes

  * **Choose Your Join Wisely:** We used `how='inner'`, which is the most common type of merge. It only includes keys present in **both** DataFrames. Another very common type is `how='left'`, which keeps **all** rows from the left DataFrame (`employees_df` in our case) and fills in `NaN` for any rows that don't have a match in the right DataFrame.
  * **`.apply()` Can Be Slow:** While incredibly flexible, `.apply()` is essentially a loop. If you can achieve the same result with a built-in, vectorized pandas or NumPy function (like we did with `np.where` in the last part), that solution will almost always be significantly faster. Use `.apply()` when a vectorized option isn't available.
  * **Method Chaining:** Pandas is designed for "method chaining," where you can link multiple operations together in a clean, readable way. For example, you could get the top 2 most common departments with a single line: `employee_depts_df['dept_name'].value_counts().head(2)`.


## Working with Data Types and Pivot Tables

### Concept Overview

Real-world data is rarely clean. Dates are often stored as text, and string columns can contain inconsistent formatting or multiple pieces of information mashed together. Before you can analyze or model your data, you must transform these columns into the correct format.

Pandas provides special tools called **accessors** that unlock a wide range of functions for specific data types:

  * **`.str` accessor:** Used on a Series of strings, it allows you to apply vectorized string methods to every element at once (e.g., convert to lowercase, find a substring, or split by a delimiter). 📝
  * **`.dt` accessor:** Used on a Series of datetime objects, it allows you to easily extract time-based properties like the year, month, day of the week, and more. 🗓️

Finally, we'll look at **Pivot Tables**. If you've ever used a spreadsheet program like Excel, you'll be familiar with this powerful concept. A pivot table is a data summarization tool that reshapes or "pivots" your data, allowing you to view it from different perspectives by turning unique row values into columns.

-----

### Code Implementation

Let's create a new DataFrame representing sales orders, which has data types that need cleaning.

```python
import pandas as pd

data = {
    'order_date': ['2025-01-05', '2025-01-06', '2025-01-05', '2025-01-07', '2025-01-06'],
    'product_code': ['PROD-A_123', 'PROD-B_456', 'PROD-A_789', 'PROD-C_101', 'PROD-B_112'],
    'region': ['North', 'South', 'North', 'North', 'West'],
    'sales': [150, 200, 50, 300, 250]
}
sales_df = pd.DataFrame(data)
print("--- Original DataFrame ---")
print(sales_df)
sales_df.info()
```

#### Working with Text Data (`.str` Accessor)

The `product_code` column has mixed case and contains two pieces of information. Let's clean it up.

```python
# Convert the whole column to lowercase
sales_df['product_code_lower'] = sales_df['product_code'].str.lower()

# Extract the product category (the part before the '_')
# .split() returns a list of strings, .str.get(0) gets the first element
sales_df['product_category'] = sales_df['product_code'].str.split('_').str.get(0)

print("\n--- DataFrame with Cleaned Text Columns ---")
print(sales_df)
```

#### Working with Datetime Data (`.dt` Accessor)

The `order_date` column is currently a generic `object`. We need to convert it to a datetime object.

```python
# Convert the 'order_date' column to datetime objects
sales_df['order_date'] = pd.to_datetime(sales_df['order_date'])

# Now that it's a datetime object, we can use the .dt accessor
sales_df['order_month'] = sales_df['order_date'].dt.month
sales_df['order_day_of_week'] = sales_df['order_date'].dt.day_name()

print("\n--- DataFrame with Extracted Date Parts ---")
print(sales_df.head())
print("\n--- New Dtypes ---")
sales_df.info()
```

#### Creating a Pivot Table

Let's summarize total sales by region and day of the week.

```python
# index: The column to use for the new DataFrame's index (rows)
# columns: The column whose unique values will become the new columns
# values: The column of data to aggregate
# aggfunc: The aggregation function to apply (e.g., 'sum', 'mean')
sales_summary = sales_df.pivot_table(index='region', 
                                     columns='order_day_of_week', 
                                     values='sales', 
                                     aggfunc='sum',
                                     fill_value=0) # Fills NaN with 0

print("\n--- Pivot Table: Total Sales by Region and Day ---")
print(sales_summary)
```

-----

### Output Explanation

  * **Original DataFrame:** The `.info()` output shows that `order_date` and `product_code` are both of type `object`, which is pandas' default for text.
  * **Text Data:** The DataFrame now has two new columns. `product_code_lower` is the lowercase version of the original. `product_category` contains only the "PROD-A", "PROD-B", etc., part of the code, which we successfully extracted using string splitting.
  * **Datetime Data:** The second `.info()` output confirms that the `order_date` column is now of type `datetime64[ns]`. Because of this change, we were able to use the `.dt` accessor to easily create the `order_month` and `order_day_of_week` columns.
  * **Pivot Table:** The output is a new, reshaped DataFrame. Each unique `region` is a row, and each unique `order_day_of_week` is a column. The values in the cells represent the sum of sales for that specific region/day combination. For example, the North region had \\$200 in sales on Sunday (`150 + 50`). The value `0` for South on Sunday indicates there were no sales for that combination.

-----

### Practical Notes

  * **Always Check Your `Dtypes`:** This is a crucial first step in any data cleaning process. Use `.info()` to see what data types pandas has inferred. You cannot use the `.dt` accessor on an `object` column, even if it looks like a date. You must convert it first with `pd.to_datetime()`.
  * **The Power of `.str.extract()`:** For extracting text that matches a specific pattern (e.g., pulling a number out of a string), the `.str.split()` method can be limited. The `.str.extract()` method, which uses **regular expressions**, is an incredibly powerful tool for advanced text parsing.
  * **Pivoting vs. Groupby:** A pivot table is often a more readable presentation of a `groupby` operation. The command `df.groupby(['region', 'order_day_of_week'])['sales'].sum()` would produce the same information, but in a hierarchical Series format. A pivot table presents it in a more intuitive, spreadsheet-like grid.


## Data Cleaning and Advanced Aggregation

### Concept Overview

This final part of our pandas deep dive focuses on the practical tasks of tidying up your dataset and performing complex summaries. We'll cover four key areas:

  * **Sorting Data (`.sort_values()`):** A straightforward but essential task for ordering your DataFrame based on the values in one or more columns. This is useful for inspection and for preparing data for certain types of plots.
  * **Handling Duplicates:** Real-world data often contains duplicate entries. Identifying and removing these is a critical data cleaning step to prevent your model from being biased. 🧹
  * **Binning (Discretization):** The process of converting a continuous numerical variable into discrete categorical "bins". For example, turning a specific `age` into an `Age Group` like "Young", "Adult", or "Senior". This is a powerful feature engineering technique.
  * **Advanced Aggregation (`.agg()`):** Going beyond a single `groupby()` summary (like `.mean()`), the `.agg()` method allows you to apply multiple aggregation functions at once, and even apply different functions to different columns, giving you a rich, detailed summary in a single command.

-----

### Code Implementation

Let's create a sample DataFrame of customer data. We'll intentionally add a duplicate row to demonstrate the cleaning process.

```python
import pandas as pd

data = {
    'region': ['North', 'South', 'North', 'West', 'South', 'North'],
    'age': [35, 42, 28, 51, 42, 28],
    'visits': [10, 15, 8, 20, 12, 8],
    'spend': [500, 750, 400, 1200, 600, 400]
}
customers_df = pd.DataFrame(data)

# Add a fully duplicate row for demonstration
duplicate_row = pd.DataFrame([['North', 28, 8, 400]], columns=customers_df.columns)
customers_df = pd.concat([customers_df, duplicate_row], ignore_index=True)

print("--- Original DataFrame (with duplicates) ---")
print(customers_df)
```

#### Sorting Data

Let's sort our data to better understand customer spend.

```python
# Sort by a single column in descending order
sorted_by_spend = customers_df.sort_values(by='spend', ascending=False)
print("\n--- Sorted by Spend (Descending) ---")
print(sorted_by_spend)
```

#### Handling Duplicates

First, we find the duplicates, then we remove them.

```python
# Identify which rows are duplicates (returns a boolean Series)
# The first occurrence is marked as False
print("\n--- Identifying Duplicate Rows ---")
print(customers_df.duplicated())

# Drop the duplicate rows
customers_cleaned_df = customers_df.drop_duplicates(keep='first')
print("\n--- DataFrame After Dropping Duplicates ---")
print(customers_cleaned_df)
```

#### Binning Numerical Data (`pd.cut`)

Let's convert the continuous `age` column into categorical `age_group`.

```python
# Define the edges of our bins and the labels
bins = [0, 30, 50, 100]
labels = ['Young', 'Adult', 'Senior']

# Create the new column
customers_cleaned_df['age_group'] = pd.cut(customers_cleaned_df['age'], bins=bins, labels=labels, right=False)
print("\n--- DataFrame with 'age_group' Column ---")
print(customers_cleaned_df)
```

#### Advanced Aggregation with `.agg()`

Let's create a complex summary of our customer data.

```python
# Group by region and apply multiple functions to the 'spend' column
spend_summary = customers_cleaned_df.groupby('region')['spend'].agg(['sum', 'mean', 'count'])
print("\n--- Spend Summary by Region ---")
print(spend_summary)


# Group by region and apply different functions to different columns
# Pass a dictionary to .agg() where keys are columns and values are functions
mixed_summary = customers_cleaned_df.groupby('region').agg({
    'spend': 'mean',
    'visits': 'sum',
    'age': ['min', 'max']
})
print("\n--- Mixed Summary by Region ---")
print(mixed_summary)
```

-----

### Output Explanation

  * **Sorting Data:** The DataFrame is reordered, with the highest-spending customer at the top.
  * **Handling Duplicates:** The `.duplicated()` method correctly identifies row 6 as a duplicate of row 2. After `.drop_duplicates()`, the resulting DataFrame is one row shorter, with the duplicate entry removed.
  * **Binning:** The new `age_group` column has been added. Each customer's age has been mapped to one of our defined categories. For example, the customer with age 28 is labeled 'Young', while the customer with age 51 is 'Senior'. The `right=False` argument means the interval is inclusive of the left edge (e.g., `[0, 30)`).
  * **Advanced Aggregation:**
      * The first summary shows the total (`sum`), average (`mean`), and number (`count`) of transactions for the `spend` column, neatly grouped by region.
      * The second, more complex summary shows the average spend, the total number of visits, and both the minimum and maximum age for customers in each region, all in one clean table.

-----

### Practical Notes

  * **Reset Your Index:** After sorting, dropping rows, or filtering, your DataFrame's index can become disordered (e.g., 0, 1, 3, 5...). It's often good practice to create a clean, new index by chaining `.reset_index(drop=True)` at the end of your operation. The `drop=True` part prevents the old index from being added as a new column.
  * **Why Bin Data?** Binning is a powerful feature engineering technique. It can help machine learning models by capturing non-linear relationships, and it makes the data more robust by reducing the impact of small measurement errors. It's also very useful for visualization.
  * **The Power of `.agg()`:** When you need to create a summary table with multiple statistics, `groupby().agg()` is the best tool for the job. It's far more efficient than calculating each summary statistic separately and then trying to combine them manually.


