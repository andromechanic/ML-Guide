# Chapter 3: A Crash Course in Python for Data Science

### Concept Overview

Before we can build sophisticated machine learning models, we need to master the basic building blocks of Python. Think of these as the LEGO® bricks of our code. Everything we do, from loading data to training a model, involves manipulating these fundamental pieces.

In this section, we'll cover **variables** and the primary **data types** they hold.

  * **Variables** are simply names or labels that you assign to values. Instead of working with a raw number like `3.14159`, you can assign it to a variable called `pi`, which is much easier to read and use.

  * **Data Types** are the categories that values fall into. Python needs to know if a value is text, a whole number, or a decimal to know how to work with it. The most common types you'll encounter are:

      * **String (`str`):** Plain text, enclosed in single `'...'` or double `"..."` quotes. Used for things like column names, text data, or file paths.
      * **Integer (`int`):** Whole numbers, both positive and negative, with no decimal part (e.g., `10`, `-5`, `2025`).
      * **Float (`float`):** Numbers that have a decimal part (e.g., `3.14`, `-0.5`, `99.99`). Used for most measurements and model outputs.
      * **Boolean (`bool`):** Represents one of two values: **True** or **False**. These are crucial for logic and control flow.

-----

### Code Implementation

Let's see these concepts in a Jupyter Notebook cell. Python is dynamically typed, meaning you don't have to declare the type of a variable; Python figures it out for you.

#### Storing Different Data Types in Variables

```python
# A string variable for the project's focus
project_focus = "Customer Churn Prediction"

# An integer for the number of records in our dataset
record_count = 50000

# A float for the target accuracy of our model
target_accuracy = 0.95

# A boolean to track if data preprocessing is complete
preprocessing_done = True

# Python's type() function can tell you the type of any variable
print(f"'{project_focus}' is of type: {type(project_focus)}")
print(f"{record_count} is of type: {type(record_count)}")
print(f"{target_accuracy} is of type: {type(target_accuracy)}")
print(f"{preprocessing_done} is of type: {type(preprocessing_done)}")
```

#### Working with f-strings

Formatted strings, or "f-strings," are a modern and powerful way to embed variables directly inside a string for printing or logging.

```python
# Using f-strings to create a summary sentence
model_name = "Random Forest"
dataset_version = 3.2

summary = f"We are training a {model_name} model on dataset v{dataset_version}."
print(summary)
```

-----

### Output Explanation

  * **Data Types:** The first code block's output confirms how Python correctly inferred the data type for each variable we created. The `<class 'str'>` means it's a string, `<class 'int'>` is an integer, and so on. Understanding these types is vital because they determine what operations you can perform. For example, you can perform mathematical calculations on integers and floats, but not on strings.
  * **f-strings:** The second block's output shows a clean, readable sentence: `"We are training a Random Forest model on dataset v3.2."`. The f-string automatically substituted the values of the `model_name` and `dataset_version` variables into the text. This is the preferred way to format strings in modern Python.

-----

### Practical Notes

  * **Choose Meaningful Names:** Always use descriptive variable names (e.g., `learning_rate` instead of `lr`). This makes your code vastly more readable for yourself and others.
  * **Quotes Matter:** Python treats single (`'`) and double (`"`) quotes the same for creating strings. The key is to be consistent. If your string contains an apostrophe, it's often easier to enclose the whole string in double quotes, like `"It's a sunny day"`.
  * **Type Errors:** A common bug is a **TypeError**, which happens when you try to perform an operation on the wrong data type, like adding a string to an integer (`'hello' + 5`). Python will stop and tell you this isn't allowed. Be mindful of your data types\! 💡



## Organizing Data with Structures

### Concept Overview

Variables are great for storing single pieces of information, but in machine learning, we almost always work with collections of data. A dataset, for instance, is a collection of thousands of data points. Python's **data structures** are the tools we use to organize and manage these collections efficiently.

We will focus on four essential data structures. Understanding when and why to use each one is a foundational skill.

  * **List:** An ordered, changeable collection of items. This is your go-to structure for holding sequences of data where the order matters and you might need to add, remove, or change items. Think of it as a grocery list. 🛒
  * **Dictionary (`dict`):** An unordered collection of **key-value pairs**. Each item has a unique key (like a word in a dictionary) that maps to a value (its definition). Dictionaries are perfect for storing labeled data, like the properties of a single house.
  * **Tuple:** An ordered, **unchangeable** collection of items. Tuples are like lists, but once you create them, you cannot modify their contents. They are useful for data that should remain constant, such as coordinates or fixed configuration settings.
  * **Set:** An unordered collection of **unique** items. Sets are highly optimized for checking if an item is present in the collection and for removing duplicate values.

-----

### Code Implementation

#### Lists: The Versatile Workhorse

```python
# A list of features for a machine learning model
features = ['sq_feet', 'bedrooms', 'bathrooms', 'year_built']

# Accessing items by index (starts at 0)
print(f"First feature: {features[0]}")

# Slicing to get a sub-list (from index 1 up to, but not including, 3)
print(f"Middle features: {features[1:3]}")

# Modifying an item in the list
features[0] = 'square_feet' 
print(f"List after modification: {features}")

# Adding an item to the end
features.append('garage_size')
print(f"List after adding an item: {features}")

# Getting the number of items in the list
print(f"Total number of features: {len(features)}")
```

#### Dictionaries: Storing Labeled Data

```python
# A dictionary representing a single car for sale
car_profile = {
    'make': 'Toyota',
    'model': 'Camry',
    'year': 2021,
    'mileage': 15000,
    'is_for_sale': True
}

# Accessing a value by its key
print(f"Car make: {car_profile['make']}")

# Adding or updating a key-value pair
car_profile['color'] = 'Blue' # Add a new key
car_profile['mileage'] = 16500 # Update an existing key
print(f"Updated car profile: {car_profile}")

# Get all keys from the dictionary
print(f"All keys: {car_profile.keys()}")
```

#### Tuples: Immutable Sequences

```python
# A tuple to store RGB color values (which shouldn't change)
primary_red = (255, 0, 0)

# Accessing items works just like a list
print(f"The 'green' value is: {primary_red[1]}")

# You CANNOT change a tuple's values. The following line would cause an error:
# primary_red[0] = 200 # This would raise a TypeError
print("Attempting to change a tuple will cause an error.")
```

#### Sets: For Uniqueness and Fast Lookups

```python
# A list with duplicate values
visitor_logs = ['user_A', 'user_B', 'user_C', 'user_A', 'user_B']

# Creating a set from the list automatically removes duplicates
unique_visitors = set(visitor_logs)
print(f"Unique visitors: {unique_visitors}")

# Sets are great for fast membership testing
print(f"Did we see user_C? {'user_C' in unique_visitors}")
print(f"Did we see user_D? {'user_D' in unique_visitors}")
```

-----

### Output Explanation

  * **Lists:** The output demonstrates how you can precisely select, modify, and add items to a list. Notice how the list changes with each operation, showing its mutable nature. The `len()` function correctly reports the number of items.
  * **Dictionaries:** The code shows how to retrieve a specific piece of information using its key (`'make'`). The final print statement shows the dictionary after a new key (`'color'`) was added and an existing value (`'mileage'`) was updated. The `.keys()` method provides a simple way to see all the labels in your dictionary.
  * **Tuples:** The output shows that accessing tuple elements is identical to lists. The key takeaway is what isn't shown: a successful modification. The immutability of tuples makes your code safer when you have data that should not be altered.
  * **Sets:** The first output line shows that the set `unique_visitors` contains only `{'user_A', 'user_B', 'user_C'}`. The duplicate entries were automatically discarded. The next two lines show the results of the membership tests, which are extremely fast operations for sets, returning `True` and `False`.

-----

### Practical Notes

Here’s a simple guide for when to use each data structure:

  * **Use a List when:** You have a collection of items where **order matters** and you might need to **change** the contents (add, remove, reorder). This is the most common data structure you will use.
  * **Use a Dictionary when:** You have a set of labeled data points and want to look them up by that label (**key**). Perfect for representing a single, structured record.
  * **Use a Tuple when:** You have a small, ordered collection of items that should **never change**. They are slightly more memory-efficient than lists.
  * **Use a Set when:** You need to store a collection of items where **uniqueness is mandatory** or when you need to perform very fast **membership checks** (i.e., "is this item in my collection?").



##  Controlling the Flow of Your Code

### Concept Overview

By default, a Python script executes from top to bottom, one line at a time. However, to build useful programs, we need more control. We need to be able to make decisions and repeat actions. This is where **control flow** statements come in. They allow you to dictate the path your program takes.

We will focus on the two most important control flow tools for data science:

1.  **Conditional Statements (`if`, `elif`, `else`):** These are the decision-makers of your code. They allow you to run specific blocks of code only if a certain condition is met. Think of it as a fork in the road: if a condition is `True`, the code goes one way; if not, it goes another. 🛣️
2.  **`for` Loops:** Loops are the workhorses of repetition. A `for` loop allows you to execute the same block of code for every single item in a sequence (like a list or dictionary). This is essential for processing the rows or columns of a dataset.

Mastering these two concepts will allow you to move from simply storing data to actively processing and transforming it.

-----

### Code Implementation

#### Conditional Logic with `if`, `elif`, `else`

Let's use conditionals to categorize a patient's blood pressure reading.

```python
# A variable representing a blood pressure reading
systolic_pressure = 145

# The if/elif/else block checks conditions from top to bottom
if systolic_pressure > 180:
    category = "Hypertensive Crisis"
elif systolic_pressure >= 140:
    category = "High Blood Pressure (Stage 2)"
elif systolic_pressure >= 130:
    category = "High Blood Pressure (Stage 1)"
elif systolic_pressure >= 120:
    category = "Elevated"
else:
    category = "Normal"

print(f"A reading of {systolic_pressure} is categorized as: {category}")
```

#### Looping with `for`

Loops let us iterate over data structures.

```python
# Looping through a list of model names
models_to_train = ['Logistic Regression', 'Random Forest', 'SVM']

print("--- Training Models ---")
for model in models_to_train:
    print(f"Training {model}...")
print("-----------------------")


# Looping through a dictionary's items to get both keys and values
model_performance = {
    'accuracy': 0.92,
    'precision': 0.89,
    'recall': 0.94
}

print("\n--- Model Performance Metrics ---")
for metric_name, score in model_performance.items():
    print(f"{metric_name.capitalize()}: {score}")
print("-------------------------------")
```

#### Combining Loops and Conditionals

This is a very common pattern: looping through data and making a decision for each item.

```python
# Find all test scores above a certain threshold
test_scores = [88, 92, 75, 99, 65, 89]
passing_threshold = 90
high_performers = []

for score in test_scores:
    if score >= passing_threshold:
        print(f"Found a high score: {score}")
        high_performers.append(score)

print(f"\nAll scores above {passing_threshold}: {high_performers}")
```

-----

### Output Explanation

  * **Conditional Logic:** The output is `A reading of 145 is categorized as: High Blood Pressure (Stage 2)`. Python checked `145 > 180` (False), then moved to the first `elif` and checked `145 >= 140` (True). It executed the code for that block and then skipped the rest of the `elif` and `else` blocks entirely.
  * **Looping:** The first loop prints one "Training..." line for each model in the list. The second loop uses the `.items()` method to unpack both the key (e.g., `'accuracy'`) and the value (e.g., `0.92`) from the dictionary in each iteration, printing each metric on a new line.
  * **Combined Example:** The code iterates through every score. The `if` statement inside the loop checks each score against the `passing_threshold`. Only for scores that meet the condition (`92` and `99`) is the "Found a high score" message printed. The final list `high_performers` contains only those scores that passed the test.

-----

### Practical Notes

  * **Indentation is Crucial:** In Python, the code that belongs inside a control flow block **must be indented** (usually with four spaces). This is not just for readability; it is a strict syntax rule. Incorrect indentation will cause an `IndentationError`. 💡

  * **List Comprehensions (A Pro Tip):** Python offers a more concise and efficient way to create lists from loops and conditionals called a **list comprehension**. The combined example above can be rewritten in a single, elegant line.

    ```python
    # The traditional way (from our example)
    high_performers = []
    for score in test_scores:
        if score >= passing_threshold:
            high_performers.append(score)

    # The list comprehension way
    high_performers_comp = [score for score in test_scores if score >= passing_threshold]

    print(f"Using a comprehension: {high_performers_comp}")
    ```

    List comprehensions are heavily used in data science for their readability and performance. It's a great pattern to learn early.



## Writing Reusable Code with Functions

### Concept Overview

As you write more code, you'll often find yourself repeating the same sequence of steps. This is where functions come in. A **function** is a named, reusable block of code that performs a specific, well-defined task.

The core principle behind functions is **DRY**—**D**on't **R**epeat **Y**ourself. Instead of copying and pasting the same lines of code in multiple places, you define it once as a function and then "call" it by name whenever you need it.

A function is like a mini-program within your program. It typically:

1.  Takes some data as input, called **parameters** (or arguments).
2.  Performs a series of operations on that data.
3.  Optionally sends a result back as output, called a **return value**.

Organizing your code into functions is arguably the most important step toward writing clean, readable, and maintainable programs—a critical skill for any machine learning project. ⚙️

-----

### Code Implementation

#### Defining and Calling a Simple Function

This is the most basic form of a function. It takes no inputs and returns no output; it just performs an action.

```python
# def is the keyword to define a function
def print_welcome_message():
    """This is a docstring. It explains what the function does."""
    print("---------------------------------")
    print("Welcome to the ML Model Trainer")
    print("---------------------------------")

# "Calling" the function to execute the code inside it
print_welcome_message()
```

#### A Function with Parameters

Parameters make functions flexible. This function can now greet any user by name.

```python
def greet_user(username):
    """Greets a specific user."""
    print(f"Hello, {username}! Welcome back.")

# Call the function with different arguments
greet_user("Alice")
greet_user("Bob")
```

#### A Function that Returns a Value

This is the most common and powerful pattern. The function performs a calculation and sends the result back.

```python
def calculate_accuracy(correct_predictions, total_predictions):
    """Calculates and returns the accuracy as a percentage."""
    if total_predictions == 0:
        return 0.0 # Avoid division by zero
    
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy

# Call the function and store its return value in a variable
model1_accuracy = calculate_accuracy(892, 1000)
model2_accuracy = calculate_accuracy(1450, 1500)

print(f"Model 1 Accuracy: {model1_accuracy:.2f}%")
print(f"Model 2 Accuracy: {model2_accuracy:.2f}%")
```

-----

### Output Explanation

  * **Simple Function:** When `print_welcome_message()` is called, it executes the three `print` statements defined within its body, displaying a formatted welcome message. The text inside the triple quotes (`"""..."""`) is a **docstring**, which is a special comment that documents the function's purpose.
  * **Function with Parameters:** The first time we call `greet_user("Alice")`, the value `"Alice"` is passed into the `username` parameter, resulting in the output `"Hello, Alice! Welcome back."`. The second call works the same way with `"Bob"`.
  * **Function that Returns a Value:** In the first call, `calculate_accuracy(892, 1000)` performs the calculation and the `return` statement sends the result (`89.2`) back. This value is then assigned to the `model1_accuracy` variable. The final `print` statements use an f-string format specifier (`:.2f`) to display the numbers rounded to two decimal places.

-----

### Practical Notes

  * **Functions are Essential for ML:** In a real project, your code will be structured with functions like `load_dataset()`, `preprocess_data(data)`, `train_model(features, labels)`, and `evaluate_performance(model, test_data)`. This modularity is key.

  * **Parameters vs. Arguments:** It's helpful to know the correct terminology.

      * **Parameters** are the variable names listed in the function's definition (e.g., `username`).
      * **Arguments** are the actual values you pass to the function when you call it (e.g., `"Alice"`).

  * **Default Arguments:** You can make function parameters optional by providing a default value. This makes your functions more flexible.

    ```python
    def make_prediction(model, data, confidence_threshold=0.8):
        # ... prediction logic ...
        print(f"Making prediction with a threshold of {confidence_threshold}")

    # Call using the default threshold
    make_prediction("my_model", "some_data") 

    # Call with a custom threshold, overriding the default
    make_prediction("my_model", "some_data", confidence_threshold=0.95)
    ```