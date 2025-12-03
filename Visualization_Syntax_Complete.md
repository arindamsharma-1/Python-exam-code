# VISUALIZATION SYNTAX GUIDE – Complete with Comments & Output

## 1. IMPORTING LIBRARIES FOR VISUALIZATION

```python
# Import necessary libraries for plotting
import matplotlib.pyplot as plt   # Core plotting library
import seaborn as sns            # High-level interface (built on matplotlib)
import numpy as np               # For numerical operations
import pandas as pd              # For data handling

# These imports should always be at the top of your script
```

---

## 2. BASIC MATPLOTLIB PLOTS

### 2.1 Line Plot (Trend over Time)

```python
import matplotlib.pyplot as plt

# Create sample data
x = [1, 2, 3, 4, 5]           # x-axis values
y = [10, 15, 8, 20, 12]       # y-axis values

# Create line plot
plt.plot(x, y)
plt.title("Simple Line Plot")   # Add title
plt.xlabel("X-axis")            # Label x-axis
plt.ylabel("Y-axis")            # Label y-axis
plt.show()                       # Display plot

# OUTPUT: A line graph connecting points (1,10), (2,15), (3,8), (4,20), (5,12)
# Title appears at top, axis labels on sides
```

### 2.2 Bar Plot (Compare Categories)

```python
import matplotlib.pyplot as plt

# Create sample data
x = [1, 2, 3, 4, 5]           # Categories
y = [10, 15, 8, 20, 12]       # Values for each category

# Create bar plot
plt.bar(x, y)                  # bar() instead of plot()
plt.title("Simple Bar Plot")
plt.xlabel("Categories")
plt.ylabel("Values")
plt.show()

# OUTPUT: A bar chart with 5 bars at positions 1-5 with heights 10, 15, 8, 20, 12
```

### 2.3 Histogram (Distribution of Data)

```python
import numpy as np
import matplotlib.pyplot as plt

# Create sample data: 100 random numbers
data = np.random.randn(100)    # 100 random numbers from normal distribution

# Create histogram
plt.hist(data)
plt.title("Simple Histogram")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()

# OUTPUT: A histogram showing distribution of 100 random values
# x-axis shows value ranges (bins), y-axis shows count of values in each range
```

---

## 3. CUSTOMIZING MATPLOTLIB PLOTS

### 3.1 Histogram with Multiple Parameters

```python
import numpy as np
import matplotlib.pyplot as plt

# Create sample data
data = np.random.randn(100)

# Create histogram with customization
plt.hist(data, bins=20, edgecolor='black')
# bins=20: divide data into 20 equal ranges
# edgecolor='black': add black borders around bars
# This makes bars more distinct

plt.title("Histogram with Customization")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()

# OUTPUT: Histogram with 20 bins and black borders around each bar
```

### 3.2 Custom X-Axis Labels

```python
import matplotlib.pyplot as plt

# Create data with categories
x = [1, 2, 3]                  # Positions on x-axis
y = [10, 20, 15]               # Values
categories = ['Apple', 'Banana', 'Cherry']  # Category names

# Create plot and set custom labels
plt.plot(x, y, marker='o')
plt.xticks([1, 2, 3], categories)  # Replace default numbers with names
plt.title("Plot with Custom X Labels")
plt.show()

# OUTPUT: Plot with x-axis showing 'Apple', 'Banana', 'Cherry' instead of 1, 2, 3
```

### 3.3 Custom Y-Axis Labels

```python
import matplotlib.pyplot as plt

# Set custom y-axis labels
plt.plot([1, 2, 3], [10, 20, 30])
plt.yticks([10, 20, 30], ['Low', 'Medium', 'High'])
# yticks([positions], [labels])
plt.title("Plot with Custom Y Labels")
plt.show()

# OUTPUT: Plot with y-axis showing 'Low', 'Medium', 'High' instead of 10, 20, 30
```

---

## 4. SEABORN PLOTS (HIGH-LEVEL INTERFACE)

### 4.1 Histogram using Seaborn

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create sample data
data = np.random.randn(100)

# Create seaborn histogram
sns.histplot(data, bins=20, kde=True, edgecolor='black')
# bins=20: divide into 20 ranges
# kde=True: add smooth curve over histogram (Kernel Density Estimate)
# edgecolor='black': black borders around bars

plt.title("Histogram using Seaborn")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.tight_layout()  # Adjust spacing to prevent label cutoff
plt.show()

# OUTPUT: Histogram with 20 bins, black borders, and a smooth curve overlay
```

### 4.2 Boxplot (Distribution & Outliers)

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Create sample data
data = [12, 7, 14, 18, 25, 30, 5, 10, 16]

# Create boxplot
sns.boxplot(data=data)
# Boxplot shows:
# - Box: represents 25%-75% of data (middle 50%)
# - Line inside box: median (50th percentile)
# - Whiskers: min and max of normal data
# - Dots: outliers (unusual extreme values)

plt.title("Boxplot Example")
plt.ylabel("Values")
plt.show()

# OUTPUT: A box showing distribution, with whiskers and potential outliers marked
```

### 4.3 Countplot (Frequency of Categories)

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Create categorical data
departments = ['CSE', 'ECE', 'CSE', 'EEE', 'ECE', 'CSE']

# Create countplot: counts frequency of each category
sns.countplot(x=departments)
# Each bar represents count of that category

plt.title("Countplot Example")
plt.xlabel("Department")
plt.ylabel("Count")
plt.show()

# OUTPUT: Bar chart showing:
# CSE: 3 bars (appears 3 times)
# ECE: 2 bars (appears 2 times)
# EEE: 1 bar (appears 1 time)
```

### 4.4 Scatterplot (Relationship Between Two Variables)

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create sample data
data = {
    'Hours Studied': [2, 3, 5, 7, 8, 6, 4, 9],
    'Test Score': [45, 60, 75, 85, 92, 80, 65, 95]
}
df = pd.DataFrame(data)

# Create scatterplot
sns.scatterplot(data=df, x='Hours Studied', y='Test Score')
# Each point represents one student
# x-position = hours studied, y-position = test score

plt.title("Hours Studied vs Test Score")
plt.xlabel("Hours Studied")
plt.ylabel("Test Score")
plt.show()

# OUTPUT: Scatter plot showing relationship between study hours and scores
# Points trend upward (more study = higher score)
```

---

## 5. SEABORN STYLING & THEMES

### 5.1 Setting Seaborn Style

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Set different styles before plotting
sns.set_style("darkgrid")    # Options: darkgrid, whitegrid, dark, white, ticks
sns.set_palette("Set2")       # Color palette

# Now create any plot
sns.countplot(x=['CSE', 'ECE', 'CSE', 'EEE'])
plt.title("Countplot with Theme")
plt.show()

# OUTPUT: Same plot but with dark grid background and Set2 color palette
```

### 5.2 Different Style Options

```python
import seaborn as sns

# Available styles:
# "darkgrid"  : Dark background with grid lines (default)
# "whitegrid" : White background with grid lines
# "dark"      : Dark background, no grid
# "white"     : White background, no grid (clean)
# "ticks"     : White with tick marks

# Available palettes:
# "Set1", "Set2", "Set3" : Distinct colors
# "husl"    : Rainbow of colors
# "coolwarm": Blue to red gradient
# "viridis" : Perceptually uniform (good for colorblind)
```

### 5.3 KDE Plot (Smooth Distribution)

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create sample data
data = np.random.randn(100)

# Create KDE (Kernel Density Estimate) plot
sns.kdeplot(data=data, fill=True)
# KDE draws a smooth curve instead of bars
# fill=True: fills area under curve

plt.title("KDE Plot")
plt.xlabel("Values")
plt.ylabel("Density")
plt.show()

# OUTPUT: Smooth curve showing distribution of data
```

---

## 6. WORKING WITH DATAFRAME DATA

### 6.1 Plot from DataFrame Columns

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create a DataFrame
df = pd.DataFrame({
    'Month': ['Jan', 'Feb', 'Mar', 'Apr'],
    'Sales': [100, 150, 120, 180]
})

# Plot directly from DataFrame
plt.figure(figsize=(8, 5))      # Set figure size (width, height)
plt.plot(df['Month'], df['Sales'], marker='o')
plt.title("Monthly Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()

# OUTPUT: Line plot of sales data by month with circular markers at each point
```

### 6.2 Boxplot with DataFrame and Grouping

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load sample dataset (Titanic)
titanic = sns.load_dataset('titanic')
# This is a built-in dataset with passenger information

# Create boxplot showing age distribution by passenger class
plt.figure(figsize=(8, 6))
sns.boxplot(x='pclass', y='age', data=titanic, palette='Set2')
# x='pclass': x-axis shows passenger class (1, 2, 3)
# y='age': y-axis shows age
# Separate boxplots for each class

plt.title("Age Distribution by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Age (years)")
plt.show()

# OUTPUT: Three boxplots side by side, one for each passenger class
# Shows age distribution and outliers for each class
```

### 6.3 Countplot with DataFrame

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load titanic dataset
titanic = sns.load_dataset('titanic')

# Create countplot: count of passengers by class
plt.figure(figsize=(6, 5))
sns.countplot(x='pclass', data=titanic, palette='pastel')
# Counts how many passengers in each class

plt.title("Number of Passengers by Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.show()

# OUTPUT: Bar chart showing passenger count for each class (1, 2, 3)
```

### 6.4 Histogram from DataFrame

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load tips dataset
tips = sns.load_dataset('tips')
# This dataset contains restaurant bill and tip information

# Create histogram of total bill
sns.histplot(data=tips, x='total_bill', bins=15, kde=True)
# x='total_bill': plot distribution of total_bill column
# bins=15: divide into 15 ranges
# kde=True: add smooth curve

plt.title("Distribution of Total Bill Amount")
plt.xlabel("Total Bill ($)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# OUTPUT: Histogram showing how bill amounts are distributed
# Most bills clustered around $15-20, with smooth curve overlay
```

---

## 7. MULTIPLE PLOTS & FIGURE MANAGEMENT

### 7.1 Multiple Subplots

```python
import matplotlib.pyplot as plt
import numpy as np

# Create figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
# 2 rows, 2 columns = 4 subplots total
# figsize=(10, 8) = figure size in inches

# Subplot 1 (top-left)
axes[0, 0].plot([1, 2, 3], [1, 2, 3])
axes[0, 0].set_title("Plot 1: Line")

# Subplot 2 (top-right)
axes[0, 1].bar([1, 2, 3], [1, 4, 2])
axes[0, 1].set_title("Plot 2: Bar")

# Subplot 3 (bottom-left)
axes[1, 0].hist([1, 2, 2, 3, 3, 3], bins=3)
axes[1, 0].set_title("Plot 3: Histogram")

# Subplot 4 (bottom-right)
axes[1, 1].scatter([1, 2, 3], [1, 4, 2])
axes[1, 1].set_title("Plot 4: Scatter")

plt.tight_layout()  # Prevent label overlap
plt.show()

# OUTPUT: 2x2 grid of 4 different plots
```

### 7.2 Setting Figure Size

```python
import matplotlib.pyplot as plt

# Set figure size before plotting
plt.figure(figsize=(12, 6))  # width=12 inches, height=6 inches

# Now create plot
plt.plot([1, 2, 3], [1, 4, 2])
plt.title("Large Figure")
plt.show()

# OUTPUT: A wide, horizontal plot (12 inches wide, 6 inches tall)
```

---

## 8. COMPLETE EXAMPLES FOR EXAM

### 8.1 Example 1: Tips Dataset Histogram (LIKE SAMPLE Q4)

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load tips dataset (comes with seaborn)
df = sns.load_dataset('tips')

# Print data types (required in sample Q4)
print("Data types of columns:")
print(df.dtypes)

# Create histogram of total_bill
sns.histplot(data=df, x='total_bill')

# Add labels
plt.title('Distribution of Total Bill')
plt.xlabel('Total Bill Amount')
plt.ylabel('Frequency')

plt.show()

# OUTPUT:
# Data types:
# total_bill    float64
# tip           float64
# sex           object
# smoker        object
# day           object
# time          object
# size          int64
#
# [Histogram showing bill amounts, mostly concentrated 15-20]
```

### 8.2 Example 2: Iris Dataset Scatter Plot

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load iris dataset
df = sns.load_dataset('iris')

# Print data types
print("Data types:")
print(df.dtypes)

# Create scatter plot
sns.scatterplot(data=df, x='sepal_length', y='petal_length')

plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')

plt.show()

# OUTPUT: Scatter plot showing 150 iris measurements
# Points show positive correlation (longer sepal = longer petal)
```

### 8.3 Example 3: Tips Dataset Boxplot by Day

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load tips dataset
df = sns.load_dataset('tips')

# Create boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='day', y='total_bill', palette='Set2')

plt.title('Total Bill Distribution by Day')
plt.xlabel('Day of Week')
plt.ylabel('Total Bill ($)')

plt.show()

# OUTPUT: 4 boxplots (Thu, Fri, Sat, Sun) showing bill distribution for each day
```

---

## 9. QUICK REFERENCE TABLE

| Function | Purpose | Example |
|----------|---------|---------|
| `plt.plot()` | Line plot | `plt.plot([1,2,3], [1,4,2])` |
| `plt.bar()` | Bar chart | `plt.bar([1,2,3], [1,4,2])` |
| `plt.hist()` | Histogram | `plt.hist(data, bins=20)` |
| `sns.histplot()` | Seaborn histogram | `sns.histplot(data=df, x='col')` |
| `sns.boxplot()` | Boxplot | `sns.boxplot(data=df, x='x', y='y')` |
| `sns.countplot()` | Category count | `sns.countplot(x=data)` |
| `sns.scatterplot()` | Scatter plot | `sns.scatterplot(data=df, x='a', y='b')` |
| `sns.kdeplot()` | Smooth curve | `sns.kdeplot(data=data, fill=True)` |
| `plt.title()` | Add title | `plt.title('Title')` |
| `plt.xlabel()` | X-axis label | `plt.xlabel('Label')` |
| `plt.ylabel()` | Y-axis label | `plt.ylabel('Label')` |
| `plt.show()` | Display plot | `plt.show()` |
| `plt.figure()` | Create figure | `plt.figure(figsize=(8,6))` |
| `sns.set_style()` | Set theme | `sns.set_style('darkgrid')` |

---

## 10. COMMON MISTAKES & FIXES

### ❌ Plot not showing
```python
# WRONG: forgot plt.show()
plt.plot([1,2,3], [1,4,2])
plt.title("Test")

# RIGHT: add plt.show()
plt.plot([1,2,3], [1,4,2])
plt.title("Test")
plt.show()
```

### ❌ DataFrame plot error
```python
# WRONG: forgot data= parameter
sns.histplot(x='col')

# RIGHT: specify data and column
sns.histplot(data=df, x='col')
```

### ❌ Figure too small/crowded
```python
# WRONG: no size specification
plt.plot(x, y)

# RIGHT: set figure size
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.show()
```

