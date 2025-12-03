# EXAM REFERENCE SHEET – Python, Pandas, Visualization (All-in-One)

For your **open book exam** – Print this sheet and take to exam.

---

## SECTION A: BASIC PYTHON FUNCTIONS (10 MARKS)

### Template: Function with Two Properties

```python
def function_name(parameter):
    # Comment explaining what this section does
    if parameter > threshold:
        print("Result 1")
    elif parameter < threshold:
        print("Result 2")
    else:
        print("Result 3")
    
    # Check second property
    if parameter % 2 == 0:
        print("Even")
    else:
        print("Odd")

# Call function
value = int(input("Enter value: "))
function_name(value)
```

### Common Operators

```python
# Comparison: ==, !=, >, <, >=, <=
# Logical: and, or, not
# Modulo: n % 2 (even/odd), n % d (divisible)

if n % 2 == 0:           # Even number
    print("Even")
else:
    print("Odd")

if n > 0 and n < 100:    # Both true
    print("Valid")

if x == 5 or x == 10:    # At least one true
    print("Match")
```

---

## SECTION B: PANDAS (FOR DATAFRAMES)

### Import

```python
import pandas as pd
import numpy as np
```

### Create Series (1D)

```python
# From list
s = pd.Series([10, 20, 30])

# From list with custom index
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'])

# From dictionary
s = pd.Series({'a': 10, 'b': 20, 'c': 30})

# From NumPy array
arr = np.array([10, 20, 30])
s = pd.Series(arr)
```

### Create DataFrame (2D Table)

```python
# From dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'Salary': [50000, 60000, 70000]
}
df = pd.DataFrame(data)

# From list of dicts
data = [
    {'Name': 'Alice', 'Age': 25},
    {'Name': 'Bob', 'Age': 30}
]
df = pd.DataFrame(data)

# Load from CSV
df = pd.read_csv('file.csv')

# Load built-in dataset (for EXAM)
df = sns.load_dataset('tips')
```

### Inspect DataFrame

```python
print(df.head())          # First 5 rows
print(df.head(3))         # First 3 rows
print(df.tail())          # Last 5 rows
print(df.shape)           # (rows, columns)
print(df.dtypes)          # Data types of columns
print(df.info())          # Summary info
print(df.describe())      # Statistics
print(df.columns)         # Column names
```

### Access Data

```python
# By column name (returns Series)
df['Age']
df[['Name', 'Age']]       # Multiple columns

# By position (loc = label-based, iloc = position-based)
df.loc[0]                 # First row by label
df.iloc[0]                # First row by position
df.loc[0, 'Age']          # Row 0, column 'Age'
df.iloc[0, 1]             # Row 0, column 1

# Conditional
df[df['Age'] > 25]        # Rows where Age > 25
df[(df['Age'] > 25) & (df['Salary'] > 55000)]  # Multiple conditions
```

### Sort & Group

```python
# Sort by column
df.sort_values(by='Age')
df.sort_values(by='Age', ascending=False)

# Group and aggregate
df.groupby('Category')['Salary'].mean()
df.groupby('Category')['Salary'].sum()
df.groupby('Category').agg(['mean', 'min', 'max'])
```

### Handle Missing Values

```python
df.isna()                      # Check for NaN
df.isna().sum()                # Count NaN per column
df.fillna(0)                   # Fill with 0
df.fillna(df.mean())           # Fill with mean
df.dropna()                    # Remove rows with NaN
df['col'].fillna(df['col'].mean())  # Fill one column
```

---

## SECTION C: VISUALIZATION (MATPLOTLIB & SEABORN)

### Import

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
```

### Basic Plots

```python
# Line plot
plt.plot([1, 2, 3], [10, 20, 15])
plt.show()

# Bar plot
plt.bar([1, 2, 3], [10, 20, 15])
plt.show()

# Histogram
plt.hist(data, bins=20)
plt.show()
```

### Add Labels & Title

```python
plt.title("My Plot Title")
plt.xlabel("X-axis Label")
plt.ylabel("Y-axis Label")
plt.show()
```

### Seaborn Plots (High-Level, Better for Exams)

```python
# Histogram from DataFrame
sns.histplot(data=df, x='column_name')
plt.title("Title")
plt.xlabel("Label")
plt.show()

# Boxplot
sns.boxplot(data=df, x='category', y='value')
plt.show()

# Countplot (category frequency)
sns.countplot(data=df, x='category')
plt.show()

# Scatterplot
sns.scatterplot(data=df, x='col1', y='col2')
plt.show()
```

### Load Built-in Datasets

```python
# Tips dataset (common in exams)
df = sns.load_dataset('tips')
print(df.dtypes)
sns.histplot(data=df, x='total_bill')
plt.title('Distribution of Total Bill')
plt.xlabel('Total Bill Amount')
plt.show()

# Iris dataset
df = sns.load_dataset('iris')

# Titanic dataset
df = sns.load_dataset('titanic')
```

### Styling

```python
# Set style
sns.set_style("darkgrid")      # darkgrid, whitegrid, dark, white
sns.set_palette("Set2")         # Set2, Set1, coolwarm, etc.

# Set figure size
plt.figure(figsize=(10, 6))    # width=10, height=6 inches
```

---

## EXAM QUESTION PATTERNS & SOLUTIONS

### PATTERN 1: Analyze Number (10 Marks)

**Question:** Write function `analyze_number(n)` that prints:
- positive/negative/zero
- even/odd

```python
def analyze_number(n):
    if n > 0:
        print("Positive")
    elif n < 0:
        print("Negative")
    else:
        print("Zero")
    
    if n % 2 == 0:
        print("Even")
    else:
        print("Odd")

n = int(input("Enter number: "))
analyze_number(n)
```

### PATTERN 2: Filter Unique List (10 Marks)

**Question:** Write function `filter_unique(input_list)` that returns unique elements sorted.

```python
def filter_unique(input_list):
    unique_set = set(input_list)      # Remove duplicates
    unique_list = sorted(unique_set)  # Sort ascending
    return unique_list

data = [3, 1, 2, 2, 5, 3, 1]
result = filter_unique(data)
print("Original:", data)
print("Unique sorted:", result)
# Output: Unique sorted: [1, 2, 3, 5]
```

### PATTERN 3: NumPy Array Stats (15 Marks)

**Question:** Create 10 random integers (1-100), print mean/median/std, multiply by 2.

```python
import numpy as np

# Create array
arr = np.random.randint(1, 101, size=10)
print("Array:", arr)

# Calculate statistics
print("Mean:", np.mean(arr))
print("Median:", np.median(arr))
print("Std Dev:", np.std(arr))

# Multiply by 2
arr_times_2 = arr * 2
print("Array × 2:", arr_times_2)
```

### PATTERN 4: Tips Dataset Histogram (15 Marks) ⭐ MOST COMMON

**Question:** Load tips dataset, print dtypes, draw histogram of total_bill.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = sns.load_dataset('tips')

# Print data types
print("Data types:")
print(df.dtypes)

# Draw histogram
sns.histplot(data=df, x='total_bill')
plt.title("Distribution of Total Bill")
plt.xlabel("Total Bill Amount")
plt.show()
```

### PATTERN 5: Any DataFrame + Scatter (15 Marks)

**Question:** Load dataset, explore, create scatter plot.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('iris')

print("Data types:")
print(df.dtypes)
print("Shape:", df.shape)

sns.scatterplot(data=df, x='sepal_length', y='petal_length')
plt.title("Sepal vs Petal Length")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()
```

---

## QUICK COPY-PASTE TEMPLATES

### 10-Mark Template

```python
def solve(param):
    # Property 1
    if param > 0:
        print("positive")
    else:
        print("negative")
    
    # Property 2
    if param % 2 == 0:
        print("even")
    else:
        print("odd")

val = int(input("Enter: "))
solve(val)
```

### 15-Mark NumPy Template

```python
import numpy as np

arr = np.random.randint(1, 101, size=10)
print(arr)
print("Mean:", np.mean(arr))
print("Median:", np.median(arr))
print("Std:", np.std(arr))
print(arr * 2)
```

### 15-Mark Pandas + Plot Template

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('tips')
print(df.dtypes)

sns.histplot(data=df, x='total_bill')
plt.title("Title")
plt.xlabel("Label")
plt.show()
```

---

## COMMON MISTAKES TO AVOID

❌ `if n = 5:` → ✅ `if n == 5:`  
❌ `if n > 5` → ✅ `if n > 5:`  
❌ Missing `plt.show()` → ✅ Always add `plt.show()`  
❌ `df['col']` from non-existent column → ✅ Check column name spelling  
❌ `np.random.randint(1, 100)` → ✅ `np.random.randint(1, 101, size=10)`  
❌ Forgot `import` statement → ✅ Always import at top  

---

## EXAM DAY CHECKLIST

✓ Read question carefully (identify 10-mark vs 15-mark)  
✓ Identify which dataset/data structure to use  
✓ Write imports first  
✓ Write function/code structure  
✓ Test with sample values  
✓ Add titles and labels for plots  
✓ Call `plt.show()` for visualizations  
✓ Print required outputs (dtypes, stats, etc.)  

---

**YOU ARE READY! Good luck! 🚀**

