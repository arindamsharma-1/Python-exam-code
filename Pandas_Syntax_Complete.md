# PANDAS SYNTAX GUIDE – Complete with Comments & Output

## 1. IMPORTING PANDAS

```python
# Import the pandas library with alias 'pd'
import pandas as pd

# Output: No output on import, but now pd is available for use
```

---

## 2. CREATING SERIES

### 2.1 Creating Series from a List (with default index 0, 1, 2...)

```python
import pandas as pd

# Create a Series from a list of numbers
# Series: 1D array with automatic index (0, 1, 2, ...)
s = pd.Series([10, 20, 30, 40])
print("Series from list:")
print(s)

# OUTPUT:
# 0    10
# 1    20
# 2    30
# 3    40
# dtype: int64
```

### 2.2 Creating Series with Custom Index

```python
import pandas as pd

# Create a Series with custom index labels instead of default 0, 1, 2
# This is useful when you want meaningful labels for each element
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print("Series with custom index:")
print(s)

# OUTPUT:
# a    10
# b    20
# c    30
# d    40
# dtype: int64
```

### 2.3 Creating Series from a List

```python
import pandas as pd

# Method 1: From a Python list
list1 = [10, 20, 30, 50, 40, 70]
series_from_list = pd.Series(list1)
print("Series from list:")
print(series_from_list)

# OUTPUT:
# 0    10
# 1    20
# 2    30
# 3    50
# 4    40
# 5    70
# dtype: int64
```

### 2.4 Creating Series from a Tuple

```python
import pandas as pd

# Method 2: From a tuple (tuples are immutable)
# Replace square brackets [] with parentheses ()
tup1 = (10, 20, 30, 50, 40, 70)
series_from_tuple = pd.Series(tup1)
print("Series from tuple:")
print(series_from_tuple)

# OUTPUT:
# 0    10
# 1    20
# 2    30
# 3    50
# 4    40
# 5    70
# dtype: int64
```

### 2.5 Creating Series from a Dictionary

```python
import pandas as pd

# Method 3: From a dictionary (keys become the index)
dict1 = {'a': 'hello', 'b': 'bye', 'c': 'hai'}
series_from_dict = pd.Series(dict1)
print("Series from dictionary:")
print(series_from_dict)

# OUTPUT:
# a    hello
# b      bye
# c      hai
# dtype: object
```

### 2.6 Creating Series from NumPy Array

```python
import pandas as pd
import numpy as np

# Method 4: From a NumPy array
arr1 = np.array([10, 20, 40, 30, 50, 60])
series_from_array = pd.Series(arr1)
print("Series from NumPy array:")
print(series_from_array)

# OUTPUT:
# 0    10
# 1    20
# 2    40
# 3    30
# 4    50
# 5    60
# dtype: int64
```

---

## 3. CREATING DATAFRAMES

### 3.1 Creating DataFrame from Dictionary (Most Common)

```python
import pandas as pd

# Create DataFrame from dictionary of lists
# Keys become column names, values become column data
data = {
    'Name': ['Amit', 'Bezek', 'Charan'],
    'Age': [21, 22, 23],
    'City': ['Delhi', 'Mumbai', 'Chennai']
}

df = pd.DataFrame(data)
print("DataFrame from dictionary:")
print(df)

# OUTPUT:
#     Name  Age     City
# 0   Amit   21    Delhi
# 1  Bezek   22   Mumbai
# 2 Charan   23  Chennai
```

### 3.2 Creating DataFrame from List of Lists

```python
import pandas as pd

# Create DataFrame from list of lists
# Each inner list becomes a row
data = [[1, 'Apple'], 
        [2, 'Banana'], 
        [3, 'Orange']]

df = pd.DataFrame(data, columns=['ID', 'Fruit'])
print("DataFrame from list of lists:")
print(df)

# OUTPUT:
#   ID    Fruit
# 0  1   Apple
# 1  2  Banana
# 2  3  Orange
```

### 3.3 Creating DataFrame from NumPy Array

```python
import pandas as pd
import numpy as np

# Create DataFrame from NumPy array
# NumPy arrays must be reshaped into 2D format
arr = np.array([10, 20, 30, 40, 50, 60])
df = pd.DataFrame(arr.reshape(3, 2), columns=['Col1', 'Col2'])
print("DataFrame from NumPy array:")
print(df)

# OUTPUT:
#    Col1  Col2
# 0    10    20
# 1    30    40
# 2    50    60
```

### 3.4 Creating DataFrame from List of Dictionaries

```python
import pandas as pd

# Create DataFrame from list of dictionaries
# Each dictionary becomes a row, keys become column names
data = [
    {'Name': 'Anamika', 'Age': 21, 'Major': 'Computer Science'},
    {'Name': 'Bhagavan', 'Age': 22, 'Major': 'Engineering'},
    {'Name': 'Chaava', 'Age': 23, 'Major': 'Mathematics'}
]

df = pd.DataFrame(data)
print("DataFrame from list of dictionaries:")
print(df)

# OUTPUT:
#       Name  Age             Major
# 0  Anamika   21  Computer Science
# 1 Bhagavan   22      Engineering
# 2   Chaava   23       Mathematics
```

---

## 4. INSPECTING DATAFRAMES

### 4.1 View First N Rows

```python
import pandas as pd

# Create a DataFrame
data = {
    'StudentID': [1, 2, 3, 4, 5],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Marks': [85, 78, 92, 67, 88]
}
df = pd.DataFrame(data)

# View first 5 rows (default) or first n rows
print("First 3 rows:")
print(df.head(3))

# OUTPUT:
#   StudentID     Name  Marks
# 0          1    Alice     85
# 1          2      Bob     78
# 2          3  Charlie     92
```

### 4.2 View Last N Rows

```python
# View last 5 rows (default) or last n rows
print("Last 2 rows:")
print(df.tail(2))

# OUTPUT:
#   StudentID Name  Marks
# 3          4 David     67
# 4          5   Eve     88
```

### 4.3 Get DataFrame Shape

```python
# Shape returns (number of rows, number of columns)
print("Shape of DataFrame (rows, columns):", df.shape)

# OUTPUT:
# Shape of DataFrame (rows, columns): (5, 3)
```

### 4.4 Get Data Types of Columns

```python
# Print data type of each column
# int64 = integer, object = string, float64 = decimal number
print("Data types of columns:")
print(df.dtypes)

# OUTPUT:
# StudentID     int64
# Name         object
# Marks        int64
# dtype: object
```

### 4.5 Get Column Names

```python
# Get all column names as an Index
print("Column names:")
print(df.columns)

# OUTPUT:
# Index(['StudentID', 'Name', 'Marks'], dtype='object')
```

### 4.6 Get Index (Row Labels)

```python
# Get all row index labels
print("Index (row labels):")
print(df.index)

# OUTPUT:
# RangeIndex(start=0, stop=5, step=1)
```

### 4.7 Get Statistical Summary

```python
# Get statistical summary of numerical columns
# Shows count, mean, std, min, 25%, 50%, 75%, max
print("Statistical summary:")
print(df.describe())

# OUTPUT:
#        StudentID   Marks
# count         5.0     5.0
# mean          3.0    82.0
# std           1.414214  9.899495
# min           1.0    67.0
# 25%           2.0    78.0
# 50%           3.0    85.0
# 75%           4.0    88.0
# max           5.0    92.0
```

### 4.8 Get Information About DataFrame

```python
# Get detailed info about DataFrame
print("DataFrame info:")
print(df.info())

# OUTPUT:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 5 entries, 0 to 4
# Data columns (total 3 columns):
#  #   Column      Non-Null Count  Dtype
# ---  ------      --------------  -----
#  0   StudentID   5 non-null      int64
#  1   Name        5 non-null      object
#  2   Marks       5 non-null      int64
# dtypes: int64(2), object(1)
```

### 4.9 Get Memory Usage

```python
# Get memory usage of each column in bytes
print("Memory usage:")
print(df.memory_usage())

# OUTPUT:
# Index          128
# StudentID     40
# Name          40
# Marks         40
# dtype: int64
```

---

## 5. ACCESSING DATA – LOC (Label-based)

### 5.1 Access Single Row by Index Label

```python
import pandas as pd

data = {
    'Name': ['Amit', 'Beena', 'Chirag'],
    'Age': [23, 29, 31],
    'Salary': [50000, 62000, 58000]
}
df = pd.DataFrame(data)

# Access a specific row using .loc[index_label]
print("Row with index 1:")
print(df.loc[1])

# OUTPUT:
# Name        Beena
# Age            29
# Salary      62000
# Name: 1, dtype: object
```

### 5.2 Access Multiple Rows by Index

```python
# Access rows with indices 0, 2
print("Rows 0 and 2:")
print(df.loc[[0, 2]])

# OUTPUT:
#      Name  Age  Salary
# 0    Amit   23   50000
# 2  Chirag   31   58000
```

### 5.3 Access Specific Columns for All Rows

```python
# Access specific columns: 'Name' and 'Salary' for all rows
print("Columns 'Name' and 'Salary':")
print(df.loc[:, ['Name', 'Salary']])

# OUTPUT:
#      Name  Salary
# 0    Amit   50000
# 1   Beena   62000
# 2  Chirag   58000
```

### 5.4 Access Specific Rows and Columns

```python
# Access rows 0 to 2 and specific columns
print("Rows 0-1, columns 'Name' and 'Age':")
print(df.loc[0:1, ['Name', 'Age']])

# OUTPUT:
#    Name  Age
# 0  Amit   23
# 1 Beena   29
```

### 5.5 Conditional Access

```python
# Access rows where Salary > 55000
print("Employees with salary > 55000:")
print(df.loc[df['Salary'] > 55000])

# OUTPUT:
#      Name  Age  Salary
# 1   Beena   29   62000
# 2  Chirag   31   58000
```

### 5.6 Modify Values Using LOC

```python
# Modify the Age of row 1 to 30
df.loc[1, 'Age'] = 30
print("After modifying Age:")
print(df)

# OUTPUT:
#      Name  Age  Salary
# 0    Amit   23   50000
# 1   Beena   30   62000
# 2  Chirag   31   58000
```

---

## 6. ACCESSING DATA – ILOC (Position-based)

### 6.1 Access Row by Position

```python
# Access first row (position 0) using .iloc
print("First row:")
print(df.iloc[0])

# OUTPUT:
# Name        Amit
# Age           23
# Salary     50000
# Name: 0, dtype: object
```

### 6.2 Access Multiple Rows by Position

```python
# Access rows at positions 0 to 2 (excludes 3)
print("First 3 rows:")
print(df.iloc[0:3])

# OUTPUT:
#      Name  Age  Salary
# 0    Amit   23   50000
# 1   Beena   30   62000
# 2  Chirag   31   58000
```

### 6.3 Access Specific Columns by Position

```python
# Access columns at positions 0 and 2 (skip position 1)
print("Columns at positions 0 and 2:")
print(df.iloc[:, [0, 2]])

# OUTPUT:
#      Name  Salary
# 0    Amit   50000
# 1   Beena   62000
# 2  Chirag   58000
```

### 6.4 Access Specific Row and Column by Position

```python
# Access row 1, column 0 (single cell)
print("Row 1, Column 0:")
print(df.iloc[1, 0])

# OUTPUT:
# Beena
```

### 6.5 Access Last Rows/Columns

```python
# Access last 2 rows using negative indexing
print("Last 2 rows:")
print(df.iloc[-2:])

# OUTPUT:
#      Name  Age  Salary
# 1   Beena   30   62000
# 2  Chirag   31   58000
```

---

## 7. FILTERING DATA

### 7.1 Filter Based on Single Condition

```python
# Filter rows where Age > 25
filtered_df = df[df['Age'] > 25]
print("Employees with Age > 25:")
print(filtered_df)

# OUTPUT:
#      Name  Age  Salary
# 1   Beena   30   62000
# 2  Chirag   31   58000
```

### 7.2 Filter Based on Multiple Conditions (AND)

```python
# Filter rows where Salary > 55000 AND Age > 28
filtered_df = df[(df['Salary'] > 55000) & (df['Age'] > 28)]
print("Salary > 55000 AND Age > 28:")
print(filtered_df)

# OUTPUT:
#      Name  Age  Salary
# 1   Beena   30   62000
# 2  Chirag   31   58000
```

### 7.3 Filter Based on Multiple Conditions (OR)

```python
# Filter rows where Salary > 60000 OR Age > 30
filtered_df = df[(df['Salary'] > 60000) | (df['Age'] > 30)]
print("Salary > 60000 OR Age > 30:")
print(filtered_df)

# OUTPUT:
#      Name  Age  Salary
# 1   Beena   30   62000
# 2  Chirag   31   58000
```

### 7.4 Filter Using isin() for Multiple Values

```python
# Filter rows where Name is 'Amit' or 'Chirag'
filtered_df = df[df['Name'].isin(['Amit', 'Chirag'])]
print("Name in ['Amit', 'Chirag']:")
print(filtered_df)

# OUTPUT:
#      Name  Age  Salary
# 0    Amit   23   50000
# 2  Chirag   31   58000
```

---

## 8. SORTING DATA

### 8.1 Sort by Index

```python
# Sort by index in descending order
sorted_df = df.sort_index(ascending=False)
print("Sorted by index (descending):")
print(sorted_df)

# OUTPUT:
#      Name  Age  Salary
# 2  Chirag   31   58000
# 1   Beena   30   62000
# 0    Amit   23   50000
```

### 8.2 Sort by Single Column

```python
# Sort by Salary in ascending order
sorted_df = df.sort_values(by='Salary')
print("Sorted by Salary (ascending):")
print(sorted_df)

# OUTPUT:
#      Name  Age  Salary
# 0    Amit   23   50000
# 2  Chirag   31   58000
# 1   Beena   30   62000
```

### 8.3 Sort by Single Column Descending

```python
# Sort by Salary in descending order
sorted_df = df.sort_values(by='Salary', ascending=False)
print("Sorted by Salary (descending):")
print(sorted_df)

# OUTPUT:
#      Name  Age  Salary
# 1   Beena   30   62000
# 2  Chirag   31   58000
# 0    Amit   23   50000
```

### 8.4 Sort by Multiple Columns

```python
# Sort by Age ascending, then by Salary descending
sorted_df = df.sort_values(by=['Age', 'Salary'], ascending=[True, False])
print("Sorted by Age (asc), then Salary (desc):")
print(sorted_df)

# OUTPUT:
#      Name  Age  Salary
# 0    Amit   23   50000
# 1   Beena   30   62000
# 2  Chirag   31   58000
```

---

## 9. HANDLING MISSING VALUES

### 9.1 Detect Missing Values

```python
import pandas as pd
import numpy as np

# Create DataFrame with missing values (NaN)
data = {
    'Student': ['Tilak', 'Sarojini', 'Bhagath', 'Jhansi', 'Vallabhai'],
    'Department': ['CSE', 'ECE', 'EEE', np.nan, 'CSE'],
    'Marks': [85.0, np.nan, 78.0, 92.0, np.nan],
    'Attendance': [90.0, 85.0, np.nan, 95.0, 80.0]
}
df = pd.DataFrame(data)

# Check which values are NaN
print("Missing values (NaN):")
print(df.isna())

# OUTPUT:
#    Student  Department  Marks  Attendance
# 0    False       False  False       False
# 1    False       False   True       False
# 2    False       False  False        True
# 3    False        True  False       False
# 4    False       False   True       False
```

### 9.2 Count Missing Values

```python
# Count NaN values in each column
print("Count of missing values per column:")
print(df.isna().sum())

# OUTPUT:
# Student        0
# Department     1
# Marks          2
# Attendance     1
# dtype: int64
```

### 9.3 Fill Missing Values with a Constant

```python
# Replace all NaN with 0
df_filled = df.fillna(0)
print("After filling NaN with 0:")
print(df_filled)

# OUTPUT:
#       Student Department  Marks  Attendance
# 0       Tilak        CSE   85.0        90.0
# 1    Sarojini        ECE    0.0        85.0
# 2     Bhagath        EEE   78.0         0.0
# 3      Jhansi          0   92.0        95.0
# 4    Vallabhai        CSE    0.0        80.0
```

### 9.4 Fill Missing Values with Mean

```python
# Fill NaN in Marks column with mean of Marks
df_filled = df.copy()
df_filled['Marks'] = df_filled['Marks'].fillna(df_filled['Marks'].mean())
print("After filling Marks with mean (85.0):")
print(df_filled)

# OUTPUT:
#       Student Department  Marks  Attendance
# 0       Tilak        CSE   85.0        90.0
# 1    Sarojini        ECE   85.0        85.0
# 2     Bhagath        EEE   78.0         NaN
# 3      Jhansi        NaN   92.0        95.0
# 4    Vallabhai        CSE   85.0        80.0
```

### 9.5 Fill Missing Values with Forward Fill (propagate previous value)

```python
# Forward fill: use previous value for NaN
df_ffill = df.fillna(method='ffill')
print("Forward fill:")
print(df_ffill)

# OUTPUT:
#       Student Department  Marks  Attendance
# 0       Tilak        CSE   85.0        90.0
# 1    Sarojini        ECE   85.0        85.0
# 2     Bhagath        EEE   78.0        85.0
# 3      Jhansi        EEE   92.0        95.0
# 4    Vallabhai        CSE   92.0        80.0
```

### 9.6 Drop Rows with Missing Values

```python
# Remove rows with ANY missing value
df_dropped = df.dropna()
print("After dropping rows with NaN:")
print(df_dropped)

# OUTPUT:
#   Student Department  Marks  Attendance
# 0   Tilak        CSE   85.0        90.0
```

---

## 10. GROUPING AND AGGREGATION

### 10.1 Group By and Calculate Mean

```python
# Group by Department and calculate mean Marks
dept_mean = df.groupby('Department')['Marks'].mean()
print("Mean Marks by Department:")
print(dept_mean)

# OUTPUT:
# Department
# CSE    85.0
# ECE    85.0
# EEE    78.0
# Name: Marks, dtype: float64
```

### 10.2 Group By with Multiple Aggregations

```python
# Group by Department and calculate mean, min, max, count
dept_agg = df.groupby('Department')['Marks'].agg(['mean', 'min', 'max', 'count'])
print("Multiple aggregations by Department:")
print(dept_agg)

# OUTPUT:
#         mean  min  max  count
# Department
# CSE     85.0   85.0   85.0    2
# ECE     85.0   85.0   85.0    1
# EEE     78.0   78.0   78.0    1
```

---

## QUICK REFERENCE TABLE

| Function | Purpose | Example |
|----------|---------|---------|
| `pd.Series()` | Create 1D data with index | `pd.Series([1,2,3])` |
| `pd.DataFrame()` | Create 2D table | `pd.DataFrame({'A':[1,2]})` |
| `df.head(n)` | First n rows | `df.head(3)` |
| `df.tail(n)` | Last n rows | `df.tail(2)` |
| `df.shape` | (rows, columns) | `df.shape` → `(5, 3)` |
| `df.dtypes` | Data type of each column | `df.dtypes` |
| `df.loc[]` | Select by label | `df.loc[0, 'Name']` |
| `df.iloc[]` | Select by position | `df.iloc[0, 0]` |
| `df[df['Col'] > val]` | Filter rows | `df[df['Age'] > 25]` |
| `df.sort_values()` | Sort by column | `df.sort_values('Age')` |
| `df.isna()` | Check for NaN | `df.isna()` |
| `df.fillna()` | Fill missing values | `df.fillna(0)` |
| `df.groupby()` | Group and aggregate | `df.groupby('Dept').mean()` |

