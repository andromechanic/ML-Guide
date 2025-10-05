# Chapter 4: NumPy for High-Performance Numerical Computing

### Concept Overview

In the previous introduction, we saw that NumPy's speed comes from its core data structure, the **N-dimensional array**, and its ability to perform **vectorized** operations. Now, we'll dive deeper into how we create these arrays and explore the full range of mathematical operations we can apply to them.

This part focuses on two key areas:

1.  **Advanced Array Creation:** We'll move beyond simply converting lists and explore NumPy's powerful built-in functions for generating arrays from scratch. This includes creating arrays of random numbers, which is fundamental for initializing model weights, splitting data, and creating sample datasets.

2.  **Universal Functions (Ufuncs):** We've already seen simple vectorization with `+` and `*`. Ufuncs are the engine that makes this possible. They are high-performance functions that operate on NumPy arrays in an element-by-element fashion. This means you can apply complex mathematical functions (like square root, exponential, or logarithm) to every single element in an array without writing a slow Python `for` loop.

-----

### Code Implementation

As always, we begin by importing NumPy.

```python
import numpy as np
```

#### Advanced Array Creation

Here are the most common methods for creating arrays that you'll use in machine learning.

```python
# Create an array from a range, similar to Python's range()
arr_arange = np.arange(10)
print(f"np.arange(10):\n{arr_arange}")

# Create an array of evenly spaced numbers over an interval
# Useful for creating axes for plots
arr_linspace = np.linspace(0, 10, 5) # Start, Stop, Number of points
print(f"\nnp.linspace(0, 10, 5):\n{arr_linspace}")

# --- Creating arrays of random numbers ---

# An array of a given shape with random values from a uniform distribution [0, 1)
arr_rand = np.random.rand(2, 3) # 2 rows, 3 columns
print(f"\nnp.random.rand(2, 3):\n{arr_rand}")

# An array with random values from the standard normal ("Gaussian") distribution
arr_randn = np.random.randn(2, 3)
print(f"\nnp.random.randn(2, 3):\n{arr_randn}")

# An array of random integers
# Low (inclusive), High (exclusive), Size
arr_randint = np.random.randint(0, 100, 5) 
print(f"\nnp.random.randint(0, 100, 5):\n{arr_randint}")
```

#### Universal Functions (Ufuncs) in Action

Let's apply some common mathematical ufuncs to an array.

```python
# Start with a simple array
data_array = np.arange(1, 6)
print(f"Original array:\n{data_array}")

# Apply various ufuncs
sqrt_array = np.sqrt(data_array)
exp_array = np.exp(data_array) # e^(element)
log_array = np.log(data_array) # Natural log

print(f"\nAfter np.sqrt():\n{sqrt_array}")
print(f"\nAfter np.exp():\n{exp_array}")
print(f"\nAfter np.log():\n{log_array}")

# Ufuncs also work between two arrays (binary ufuncs)
arr1 = np.array([10, 20, 30])
arr2 = np.array([2, 5, 6])
max_array = np.maximum(arr1, arr2) # Element-wise maximum
print(f"\nElement-wise maximum of {arr1} and {arr2}:\n{max_array}")
```

-----

### Output Explanation

  * **Array Creation:** The output demonstrates the versatility of NumPy's creation functions.
      * `np.arange` is a straightforward way to create a sequence of integers.
      * `np.linspace` is different; it creates a specified *number* of points (5 in our case) that are perfectly evenly spaced between the start and stop values, including both endpoints.
      * The random functions generate arrays populated with random numbers following different statistical distributions, a task essential for many ML algorithms. Note that your random numbers will be different from the example output.
  * **Ufuncs:** The output shows how each universal function was applied to every element in `data_array` individually. `np.sqrt` calculated the square root of each number, `np.exp` calculated the exponential, and so on. The `np.maximum` function compared the elements of `arr1` and `arr2` at each position and returned the larger of the two for each position. All of this is executed without any explicit Python loops, resulting in clean and fast code.

-----

### Practical Notes

  * **Reproducibility with Random Seeds:** When you use random number generation for tasks like splitting data or initializing models, you want the results to be the same every time you run the code. You can ensure this by setting a **random seed**. Add `np.random.seed(42)` (the number 42 is a convention, but any integer will work) at the beginning of your script to get the same "random" numbers every time.
  * **Full List of Ufuncs:** NumPy offers a huge library of universal functions, including trigonometric (`np.sin`, `np.cos`), statistical (`np.mean`, `np.std`), and rounding (`np.round`, `np.ceil`) functions. You don't need to memorize them, but it's good to know they exist so you can look them up when needed.
  * **Data Types Matter:** Ufuncs and other NumPy operations are fastest when working on arrays with a specific numerical data type (like `float64` or `int64`). If your array contains mixed types (which forces its `dtype` to be `object`), you lose most of NumPy's performance benefits.



## Indexing, Slicing, and Reshaping

### Concept Overview

Creating arrays is the first step, but the real work in data science involves accessing, filtering, and restructuring your data. NumPy provides a powerful and flexible syntax for these tasks that is far more capable than standard Python list indexing.

In this part, we will master four essential techniques for manipulating array data:

  * **Indexing:** Selecting a single element from an array using its position (e.g., the element in the 2nd row and 3rd column).
  * **Slicing:** Extracting entire subsections of an array, like the first three rows or the last two columns.
  * **Boolean Indexing:** A powerful method for filtering your data based on a condition, allowing you to select all elements that meet a certain criteria (e.g., all values greater than 50).
  * **Reshaping:** Changing the dimensions of an array without changing its data. This is a common requirement when preparing data for machine learning algorithms that expect a specific input shape.

-----

### Code Implementation

Let's start by creating a sample 2D array to work with for all our examples.

```python
import numpy as np

# Create a 3x4 array (3 rows, 4 columns) with numbers 0 through 11
arr = np.arange(12).reshape(3, 4)
print("Original Array:")
print(arr)
print("-" * 20)
```

#### Indexing and Slicing

The syntax is `[row_slice, column_slice]`. The colon `:` means "all".

```python
# Get a single element (row 1, column 2)
element = arr[1, 2]
print(f"Element at [1, 2]: {element}\n")

# Get an entire row (row 0, all columns)
row_0 = arr[0, :]
print(f"Row 0: {row_0}\n")

# Get an entire column (all rows, column 1)
col_1 = arr[:, 1]
print(f"Column 1: {col_1}\n")

# Get a sub-array (rows 0-1, columns 1-2)
# The slice 1:3 means index 1 up to (but not including) index 3
sub_array = arr[0:2, 1:3]
print("Sub-array of rows 0-1 and columns 1-2:")
print(sub_array)
```

#### Boolean Indexing (Filtering)

This is the primary way to select data based on its value.

```python
# First, create a boolean condition
# This returns a boolean array of the same shape as 'arr'
condition = arr > 5
print("Boolean mask (arr > 5):")
print(condition)
print("-" * 20)

# Use the boolean array to index the original array
# This returns a 1D array containing only the elements that were True
filtered_arr = arr[condition]
print(f"Elements greater than 5: {filtered_arr}")
```

#### Reshaping Arrays

Change the structure while keeping the data the same.

```python
print(f"Original shape: {arr.shape}")

# Reshape to 4 rows and 3 columns
reshaped_arr_1 = arr.reshape(4, 3)
print("\nReshaped to (4, 3):")
print(reshaped_arr_1)

# Reshape to 2 rows, 6 columns
reshaped_arr_2 = arr.reshape(2, 6)
print("\nReshaped to (2, 6):")
print(reshaped_arr_2)

# Use -1 to have NumPy automatically calculate one dimension
# This is extremely useful!
flattened_arr = arr.reshape(-1) # Flattens to a 1D array
print(f"\nFlattened with reshape(-1): {flattened_arr}")
```

-----

### Output Explanation

  * **Indexing and Slicing:** The output shows how we can precisely grab a single number (`6`), a 1D array representing a row (`[0 1 2 3]`), a 1D array for a column (`[1 5 9]`), or a 2D sub-array (`[[1 2], [5 6]]`) from our original matrix.
  * **Boolean Indexing:** The first output is a boolean "mask" of `True`/`False` values, indicating where the condition `arr > 5` is met. When this mask is used as an index, it effectively "pulls out" only the values from the original array where the mask was `True`, resulting in a new 1D array: `[ 6 7 8 9 10 11]`.
  * **Reshaping:** The outputs show our original 12 numbers rearranged into different grid structures. `reshape(4, 3)` creates a taller matrix, while `reshape(2, 6)` creates a wider one. The `reshape(-1)` command is a handy shortcut to produce a flat, 1D array, which is a very common operation. The total number of elements (12) must remain the same for a reshape to be valid.

-----

### Practical Notes

  * **Slices are Views, Not Copies:** 💡 This is a critical concept. When you take a slice of a NumPy array, you are creating a **view** of the original data, not a new copy. **This means if you modify the slice, the original array will also be modified\!**

    ```python
    original_arr = np.array([0, 1, 2, 3, 4])
    slice_view = original_arr[2:4] # Creates a view of [2, 3]
    slice_view[:] = 99 # Modify all elements in the slice

    print(f"The slice: {slice_view}")
    print(f"The ORIGINAL array was changed: {original_arr}") # Output: [ 0  1 99 99  4]
    ```

    If you need an independent copy, you must explicitly use the `.copy()` method: `slice_copy = original_arr[2:4].copy()`.

  * **Filtering and Assignment:** You can use boolean indexing to modify parts of an array that meet a condition. For example, `arr[arr < 5] = 0` would find all elements less than 5 and replace them with 0. This is extremely useful for data cleaning tasks like handling outliers.



## Aggregations and the Axis Concept

### Concept Overview

A fundamental task in data analysis is to summarize, or **aggregate**, data into a single, meaningful value. This could be finding the sum, average, maximum, or standard deviation of your dataset. NumPy provides fast and efficient functions for performing these aggregations on arrays.

While aggregating an entire array is simple, the real power comes from performing aggregations along specific **axes** (dimensions). This allows you to ask more nuanced questions, such as "What is the average value for each column?" or "What is the maximum value in each row?"

Understanding the `axis` parameter is one of the most crucial (and sometimes tricky) concepts in NumPy:

  * **`axis=0`**: This refers to the **row** axis. When you perform an operation along `axis=0`, you are collapsing the rows and performing the calculation for each **column**. Think of it as operating "down the rows".
  * **`axis=1`**: This refers to the **column** axis. When you perform an operation along `axis=1`, you are collapsing the columns and performing the calculation for each **row**. Think of it as operating "across the columns".

-----

### Code Implementation

Let's create a sample array to demonstrate these concepts.

```python
import numpy as np

# Create a 3x5 array (3 rows, 5 columns)
data = np.arange(1, 16).reshape(3, 5)
print("Original Data:")
print(data)
print("-" * 25)
```

#### Basic Aggregations (on the entire array)

This reduces the entire array to a single value.

```python
# Sum of all elements in the array
total_sum = data.sum()
print(f"Sum of all elements: {total_sum}")

# Mean (average) of all elements
total_mean = data.mean()
print(f"Mean of all elements: {total_mean}")

# Standard deviation of all elements
total_std = data.std()
print(f"Standard deviation: {total_std:.2f}") # Format to 2 decimal places

# Maximum value in the array
total_max = data.max()
print(f"Maximum value: {total_max}")
```

#### Aggregations Along an Axis

This is where we specify `axis=0` or `axis=1`.

```python
# Get the sum of each COLUMN (collapsing the rows)
sum_by_column = data.sum(axis=0)
print(f"\nSum of each column (axis=0): {sum_by_column}")

# Get the sum of each ROW (collapsing the columns)
sum_by_row = data.sum(axis=1)
print(f"Sum of each row (axis=1):    {sum_by_row}")

# Get the mean of each COLUMN
mean_by_column = data.mean(axis=0)
print(f"\nMean of each column (axis=0): {mean_by_column}")
```

-----

### Output Explanation

  * **Original Data:** Our sample is a 3x5 grid of numbers from 1 to 15.
  * **Basic Aggregations:** These functions work as expected, reducing all 15 numbers into one summary statistic. The sum of numbers 1 through 15 is 120, the mean is 8.0, and the max is 15.
  * **Aggregations Along an Axis:** This is the key part.
      * `data.sum(axis=0)`: The output is `[18 21 24 27 30]`. This result was calculated "down the columns":
          * First element is `1 + 6 + 11 = 18`
          * Second element is `2 + 7 + 12 = 21`
          * ...and so on for each of the 5 columns.
      * `data.sum(axis=1)`: The output is `[15 40 65]`. This result was calculated "across the rows":
          * First element is `1 + 2 + 3 + 4 + 5 = 15`
          * Second element is `6 + 7 + 8 + 9 + 10 = 40`
          * Third element is `11 + 12 + 13 + 14 + 15 = 65`
      * The same logic applies to `data.mean(axis=0)`, which gives us the average of each column.

-----

### Practical Notes

  * **Feature Normalization (A Core ML Task):** A very common data preprocessing step is to scale your features. One way to do this is "standardization," where you subtract the mean and divide by the standard deviation. Critically, this must be done **per feature** (i.e., per column). This is a perfect use case for axis-based aggregations:
    ```python
    # Assume 'data' is your feature matrix (samples x features)
    # Calculate mean and std for each feature (column)
    feature_means = data.mean(axis=0)
    feature_stds = data.std(axis=0)

    # Normalize the data
    normalized_data = (data - feature_means) / feature_stds

    # print("\nFirst 5 columns of normalized data (rounded):")
    # print(np.round(normalized_data[:,:5], 2))
    ```
  * **`np.function()` vs. `array.method()`:** You can perform aggregations in two ways: `np.sum(data)` or `data.sum()`. They are largely equivalent, but the method-based approach (`data.sum()`) is often preferred for code readability as part of a chain of operations.
  * **Keeping Dimensions:** Sometimes you want the output of an aggregation to have the same number of dimensions as the input. You can achieve this with `keepdims=True`. For example, `data.sum(axis=0)` produces a 1D array of shape `(5,)`. In contrast, `data.sum(axis=0, keepdims=True)` would produce a 2D array of shape `(1, 5)`, which can be useful for certain broadcasting operations.

Of course. Here is the final part of our comprehensive chapter on NumPy, which covers the powerful concept of broadcasting and other essential operations.

-----

## Broadcasting and Other Useful Operations

### Concept Overview

So far, we have mostly performed operations on arrays of the same shape. But what happens when you need to add a single number to every element in an array, or add a 1D array to a 2D array? This is where NumPy's **broadcasting** capability comes in.

Broadcasting is a set of rules that allows NumPy to perform operations on arrays of different—but compatible—shapes. Conceptually, NumPy "stretches" or "duplicates" the smaller array so that its shape matches the larger one, enabling the element-wise operation to proceed. This is done without actually creating extra copies in memory, making it highly efficient.

Beyond broadcasting, we will also cover two other key operations that are fundamental to linear algebra and machine learning:

  * **Transposing:** Flipping a matrix over its diagonal, turning rows into columns and columns into rows.
  * **Matrix Multiplication:** Calculating the dot product of two matrices, a core operation in deep learning and many other algorithms.

-----

### Code Implementation

```python
import numpy as np
```

#### Broadcasting in Action

Let's see how NumPy handles arrays of different shapes.

```python
# Create a 3x4 base array
arr = np.zeros((3, 4))
print("Original 3x4 Array:")
print(arr)

# Example 1: Broadcasting a scalar (a single number)
# The scalar 10 is "stretched" to match the shape of arr
arr_plus_10 = arr + 10
print("\nAfter adding a scalar (10):")
print(arr_plus_10)

# Example 2: Broadcasting a 1D array onto a 2D array
row_vector = np.array([0, 1, 2, 3])
print(f"\nRow vector to add (shape {row_vector.shape}):")
print(row_vector)

# The row_vector is "stretched" down to be added to each row of arr
arr_plus_row = arr + row_vector
print("\nAfter adding the row vector:")
print(arr_plus_row)
```

#### Transposing an Array (`.T`)

The transpose is a fundamental operation that swaps axes.

```python
# Create a 3x4 array
data = np.arange(12).reshape(3, 4)
print("Original 3x4 Data:")
print(data)
print(f"Original shape: {data.shape}")

# Transpose the array
transposed_data = data.T
print("\nTransposed Data:")
print(transposed_data)
print(f"New shape: {transposed_data.shape}")
```

#### Matrix Multiplication (`@`)

This is different from element-wise multiplication (`*`).

```python
# Create two compatible matrices for multiplication
# The inner dimensions must match: (3, 4) and (4, 2) -> 4 == 4
matrix_a = np.arange(12).reshape(3, 4)
matrix_b = np.arange(8).reshape(4, 2)

print(f"\nMatrix A (shape {matrix_a.shape}):")
print(matrix_a)
print(f"\nMatrix B (shape {matrix_b.shape}):")
print(matrix_b)

# Perform matrix multiplication
dot_product = matrix_a @ matrix_b
print(f"\nDot Product (shape {dot_product.shape}):")
print(dot_product)
```

-----

### Output Explanation

  * **Broadcasting:**
      * In Example 1, the scalar `10` is added to every element of the zero matrix, resulting in a 3x4 matrix of all tens.
      * In Example 2, the 1D `row_vector` `[0 1 2 3]` is added to **each** of the three rows of the zero matrix. NumPy conceptually duplicated the row vector three times to match the shape of `arr` before performing the addition.
  * **Transposing:** The output shows that the original 3x4 matrix has been flipped into a 4x3 matrix. The first row of `data` (`[0 1 2 3]`) has become the first column of `transposed_data`.
  * **Matrix Multiplication:** The result of multiplying a `(3, 4)` matrix with a `(4, 2)` matrix is a `(3, 2)` matrix. The calculation is a dot product, a standard linear algebra operation.

-----

### Practical Notes

  * **Remember Feature Normalization?** The feature normalization example from the previous part relied heavily on broadcasting\! When we calculated `(data - feature_means)`, we subtracted a 1D array (`feature_means`) from a 2D array (`data`). NumPy broadcasted the mean vector across all rows for us.
  * **The Transpose Trick:** It's common to receive data where each column is a sample and each row is a feature. However, most machine learning libraries in Python (like scikit-learn) expect the opposite: **samples in rows and features in columns**. A quick `.T` is all you need to fix the orientation of your data matrix.
  * **`*` vs. `@` - A Critical Distinction:** 💡 This is one of the most common bugs in scientific computing.
      * `*` performs **element-wise** multiplication. The arrays must have the same shape or be broadcastable.
      * `@` performs **matrix** multiplication (dot product). The inner dimensions of the two matrices must match.
        Always be sure which one you need for your specific application.