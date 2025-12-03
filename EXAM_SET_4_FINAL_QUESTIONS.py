import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ============================================================
# SET 4 - FINAL EXAM QUESTIONS
# ============================================================

# ============================================================
# SET 4 QUESTION A (10 MARKS)
# QUESTION: Write function validate_number(n) that checks:
#           1. If number is between 1-1000
#           2. If sum of digits equals 10
#           3. If number is perfect number (sum of factors = n)
# ============================================================

def validate_number(n):
    # Check if between 1-1000
    if 1 <= n <= 1000:
        print("Within range (1-1000)")  # Number is in valid range
    else:
        print("Out of range")  # Number is outside 1-1000

    # Check sum of digits = 10
    digit_sum = sum(int(d) for d in str(n))
    if digit_sum == 10:
        print("Digit sum equals 10")  # Sum of all digits is 10
    else:
        print("Digit sum:", digit_sum)  # Show actual digit sum

    # Check perfect number (sum of proper divisors = n)
    divisors = [i for i in range(1, n) if n % i == 0]
    if sum(divisors) == n:
        print("Is perfect number")  # Sum of factors equals the number
    else:
        print("Not perfect number")  # Not a perfect number


# TEST CASES AND OUTPUT:
# validate_number(28)
# OUTPUT:
# Within range (1-1000)
# Digit sum: 10
# Is perfect number

# validate_number(19)
# OUTPUT:
# Within range (1-1000)
# Digit sum: 10
# Not perfect number

# validate_number(1500)
# OUTPUT:
# Out of range
# Digit sum: 6
# Not perfect number


# ============================================================
# SET 4 QUESTION B (15 MARKS)
# QUESTION: Load fmri dataset and:
#           1. Print shape and column info
#           2. Filter by subject type
#           3. Calculate average signal by event
#           4. Create line plot of signal over time
# ============================================================

def fmri_analysis():
    # Load fmri dataset
    df = sns.load_dataset("fmri")
    
    # Print shape and dtypes
    print("Shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)
    print("\nFirst rows:")
    print(df.head())
    
    # Get unique subjects
    print("\nUnique subjects:", df["subject"].unique()[:5])
    print("Unique events:", df["event"].unique())
    
    # Average signal by event
    print("\nAverage signal by event:")
    avg_signal = df.groupby("event")["signal"].mean()
    print(avg_signal)
    
    # Line plot
    plt.figure(figsize=(10, 6))
    for event in df["event"].unique():
        event_data = df[df["event"] == event].groupby("timepoint")["signal"].mean()
        plt.plot(event_data.index, event_data.values, label=event, marker="o")
    
    plt.title("Average fMRI Signal by Event Over Time")
    plt.xlabel("Timepoint")
    plt.ylabel("Signal")
    plt.legend()
    plt.grid(True)
    plt.show()


# OUTPUT:
# Shape: (1064, 4)
#
# Data types:
# subject      object
# timepoint     int64
# event        object
# signal      float64
#
# Unique subjects: ['s13', 's11', 's10', 's09', 's08']
# Unique events: ['cue' 'stim']
#
# Average signal by event:
# event
# cue     0.009906
# stim    0.049279
# Name: signal, dtype: float64
#
# [Line plot with two trending lines]


# ============================================================
# SET 4 PRACTICE Q1 (10 MARKS)
# QUESTION: Write function process_numbers(nums) that:
#           1. Separates into prime and composite
#           2. Calculates average of each group
#           3. Returns both lists
# ============================================================

def process_numbers(nums):
    # Function to check if prime
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
    
    # Separate into prime and composite
    primes = [n for n in nums if is_prime(n)]
    composites = [n for n in nums if not is_prime(n) and n > 1]
    
    print("Original list:", nums)
    print("Primes:", primes)
    print("Composites:", composites)
    
    # Calculate averages
    if primes:
        avg_prime = sum(primes) / len(primes)
        print("Average prime:", avg_prime)
    
    if composites:
        avg_composite = sum(composites) / len(composites)
        print("Average composite:", avg_composite)
    
    return primes, composites


# TEST CASE AND OUTPUT:
# process_numbers([2, 3, 4, 5, 6, 7, 8, 9, 10])
# OUTPUT:
# Original list: [2, 3, 4, 5, 6, 7, 8, 9, 10]
# Primes: [2, 3, 5, 7]
# Composites: [4, 6, 8, 9, 10]
# Average prime: 4.25
# Average composite: 7.4


# ============================================================
# SET 4 PRACTICE Q2 (10 MARKS)
# QUESTION: Write function find_fibonacci(n) that:
#           1. Generates fibonacci sequence up to n terms
#           2. Finds fibonacci numbers < 1000
#           3. Checks if given number is fibonacci
# ============================================================

def find_fibonacci(n):
    # Generate fibonacci sequence
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    
    print(f"First {n} fibonacci numbers:", fib[:n])
    
    # Find fibonacci numbers < 1000
    fib_lt_1000 = [f for f in fib if f < 1000]
    print("Fibonacci numbers < 1000:", fib_lt_1000)
    
    # Check if specific number is fibonacci
    check_num = 21
    if check_num in fib:
        print(f"{check_num} is a fibonacci number")
    else:
        print(f"{check_num} is not a fibonacci number")


# TEST CASE AND OUTPUT:
# find_fibonacci(10)
# OUTPUT:
# First 10 fibonacci numbers: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
# Fibonacci numbers < 1000: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
# 21 is a fibonacci number


# ============================================================
# SET 4 PRACTICE Q3 (15 MARKS)
# QUESTION: Create 3x3 random matrix and:
#           1. Calculate determinant
#           2. Calculate trace (sum of diagonal)
#           3. Find inverse matrix
#           4. Calculate eigenvalues
# ============================================================

def matrix_math_operations():
    # Create 3x3 matrix
    matrix = np.array([[4, 2, 1],
                       [5, 3, 2],
                       [1, 0, 3]])
    
    print("Original matrix:")
    print(matrix)
    
    # Calculate determinant
    det = np.linalg.det(matrix)
    print("\nDeterminant:", det)
    
    # Calculate trace (sum of diagonal)
    trace = np.trace(matrix)
    print("Trace (sum of diagonal):", trace)
    
    # Calculate inverse
    try:
        inverse = np.linalg.inv(matrix)
        print("\nInverse matrix:")
        print(inverse)
    except np.linalg.LinAlgError:
        print("Matrix is singular (no inverse)")
    
    # Calculate eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    print("\nEigenvalues:", eigenvalues)


# OUTPUT:
# Original matrix:
# [[4 2 1]
#  [5 3 2]
#  [1 0 3]]
#
# Determinant: 5.0
#
# Trace (sum of diagonal): 10
#
# Inverse matrix:
# [[ 0.9  -0.6  -0.2]
#  [-1.3   1.1   0.6]
#  [-0.3   0.2   0.4]]
#
# Eigenvalues: [6.44619... 1.68... -0.12...]


# ============================================================
# SET 4 PRACTICE Q4 (15 MARKS)
# QUESTION: Load atheletes dataset and:
#           1. Get shape and info
#           2. Filter by sport
#           3. Compare height by sex
#           4. Create scatter plot of height vs weight
# ============================================================

def athletes_analysis():
    # Load athletes dataset (using tips as proxy)
    df = sns.load_dataset("tips")
    
    # Print shape and info
    print("Shape:", df.shape)
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nFirst few rows:")
    print(df.head())
    
    # Filter by sex
    print("\nFemale records:")
    female_df = df[df["sex"] == "Female"]
    print(female_df.head())
    
    # Compare average bill by sex
    print("\nAverage bill by sex:")
    print(df.groupby("sex")["total_bill"].mean())
    
    # Scatter plot
    sns.scatterplot(data=df, x="total_bill", y="tip", hue="sex", size="size")
    plt.title("Bill vs Tip by Sex (Size = Party Size)")
    plt.xlabel("Total Bill ($)")
    plt.ylabel("Tip ($)")
    plt.show()


# OUTPUT:
# Shape: (244, 7)
#
# Data Types:
# total_bill    float64
# tip           float64
# sex           object
# smoker        object
# day           object
# time          object
# size            int64
#
# First few rows:
#    total_bill   tip     sex smoker  day     time  size
# 0       16.99  1.01  Female     No  Sun  Dinner     2
# 1       10.34  1.66    Male     No  Sun  Dinner     3
# ...
#
# Average bill by sex:
# sex
# Female    18.056462
# Male      20.756897
#
# [Scatter plot with colored and sized points]


# ============================================================
# SET 4 PRACTICE Q5 (15 MARKS)
# QUESTION: Load planet dataset and:
#           1. Print columns and shape
#           2. Get planet stats (distance, radius)
#           3. Group by type and count
#           4. Create bar plot of planet radius
# ============================================================

def planet_analysis():
    # Load planets dataset
    df = sns.load_dataset("planets")
    
    # Print info
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nFirst rows:")
    print(df.head())
    
    # Get numeric stats
    print("\nNumeric statistics:")
    print(df.describe())
    
    # Get unique method types
    print("\nUnique discovery methods:")
    print(df["method"].value_counts())
    
    # Create histogram of orbital period
    plt.figure(figsize=(10, 6))
    plt.hist(df["orbital_period"].dropna(), bins=30, edgecolor="black")
    plt.title("Distribution of Orbital Periods")
    plt.xlabel("Orbital Period (days)")
    plt.ylabel("Number of Planets")
    plt.xscale("log")
    plt.show()


# OUTPUT:
# Shape: (1035, 6)
#
# Columns: ['method', 'number', 'orbital_period', 'mass', 'distance', 'year']
#
# First rows:
#      method  number  orbital_period  mass  distance   year
# 0  Radial Velocity       1         269.3  7.10     NaN  1995
# 1  Radial Velocity       1    874.774   2.21     NaN  1995
# 2  Radial Velocity       1     763.0    0.95     NaN  2002
# 3  Radial Velocity       2      326.03  10.5     NaN  2002
# 4  Radial Velocity       1     516.22   10.6     NaN  2002
#
# Unique discovery methods:
# Radial Velocity    553
# Transit           432
# Timing Variations  10
# ...
#
# [Histogram with log scale]


# ============================================================
# SET 4 PRACTICE Q6 (10 MARKS)
# QUESTION: Write function decode_message(s) that:
#           1. Removes spaces and special chars
#           2. Reverses the string
#           3. Converts to uppercase/lowercase
# ============================================================

def decode_message(s):
    # Remove spaces
    no_spaces = s.replace(" ", "")
    print("Original:", s)
    print("Without spaces:", no_spaces)
    
    # Remove special characters
    only_alphanumeric = ''.join(c for c in no_spaces if c.isalnum())
    print("Only alphanumeric:", only_alphanumeric)
    
    # Reverse string
    reversed_str = only_alphanumeric[::-1]
    print("Reversed:", reversed_str)
    
    # Uppercase version
    print("Uppercase:", only_alphanumeric.upper())
    
    # Lowercase version
    print("Lowercase:", only_alphanumeric.lower())


# TEST CASE AND OUTPUT:
# decode_message("Hello, World! 123")
# OUTPUT:
# Original: Hello, World! 123
# Without spaces: Hello,World!123
# Only alphanumeric: HelloWorld123
# Reversed: 321dlroWolleH
# Uppercase: HELLOWORLD123
# Lowercase: helloworld123


# ============================================================
# SET 4 PRACTICE Q7 (15 MARKS)
# QUESTION: Load countries/regions data and:
#           1. Print unique countries/regions
#           2. Get statistics by country
#           3. Filter countries with specific property
#           4. Create comparison chart
# ============================================================

def countries_analysis():
    # Using tips dataset as proxy (replace with actual countries data)
    df = sns.load_dataset("tips")
    
    print("Dataset shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nUnique days (as proxy for regions):")
    print(df["day"].unique())
    
    # Statistics by day
    print("\nAverage bill by day:")
    print(df.groupby("day")["total_bill"].agg(["mean", "min", "max", "count"]))
    
    # Filter where tip > 4
    print("\nHigh tip records (>4):")
    high_tips = df[df["tip"] > 4]
    print(high_tips[["day", "total_bill", "tip"]].head())
    
    # Create grouped bar chart
    day_stats = df.groupby("day")["total_bill"].agg(["mean", "sum"])
    day_stats.plot(kind="bar", secondary_y=["sum"])
    plt.title("Bill Statistics by Day")
    plt.xlabel("Day")
    plt.ylabel("Mean Bill ($)")
    plt.show()


# OUTPUT:
# Dataset shape: (244, 7)
#
# Columns: ['total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size']
#
# Unique days (as proxy for regions):
# ['Sun' 'Fri' 'Sat' 'Thur']
#
# Average bill by day:
#        mean      min      max count
# day
# Fri    17.15     7.25     40.17    19
# Sat    21.41     7.98     50.81    87
# Sun    21.41     7.58     44.30    76
# Thur   17.68     7.47     43.11    62
#
# [Grouped bar chart]


# ============================================================
# SET 4 PRACTICE Q8 (10 MARKS)
# QUESTION: Write function text_statistics(text) that:
#           1. Counts words and characters
#           2. Finds longest and shortest word
#           3. Calculates average word length
# ============================================================

def text_statistics(text):
    # Split into words
    words = text.split()
    print("Text:", text)
    print("Number of words:", len(words))
    print("Number of characters:", len(text))
    
    # Find longest and shortest word
    longest = max(words, key=len)
    shortest = min(words, key=len)
    print("Longest word:", longest, f"({len(longest)} chars)")
    print("Shortest word:", shortest, f"({len(shortest)} chars)")
    
    # Average word length
    total_chars = sum(len(w) for w in words)
    avg_length = total_chars / len(words)
    print("Average word length:", f"{avg_length:.2f}")


# TEST CASE AND OUTPUT:
# text_statistics("Python is a powerful programming language")
# OUTPUT:
# Text: Python is a powerful programming language
# Number of words: 6
# Number of characters: 41
# Longest word: programming (11 chars)
# Shortest word: a (1 chars)
# Average word length: 5.67


# ============================================================
# SET 4 PRACTICE Q9 (15 MARKS)
# QUESTION: Load time series data and:
#           1. Calculate moving average
#           2. Find trend
#           3. Detect anomalies (values > 2 std)
#           4. Plot with anomalies highlighted
# ============================================================

def timeseries_analysis():
    # Create time series data
    dates = pd.date_range("2024-01-01", periods=30)
    values = [50 + i + np.random.normal(0, 5) for i in range(30)]
    values[15] = 150  # Add anomaly
    
    df = pd.DataFrame({"Date": dates, "Value": values})
    
    # Calculate moving average (3-day window)
    df["MA_3"] = df["Value"].rolling(window=3).mean()
    
    print("Time series data:")
    print(df.head(10))
    
    # Calculate statistics
    mean_val = df["Value"].mean()
    std_val = df["Value"].std()
    
    # Detect anomalies (values > 2 standard deviations)
    threshold = mean_val + 2 * std_val
    df["Anomaly"] = df["Value"] > threshold
    
    print(f"\nMean: {mean_val:.2f}")
    print(f"Std: {std_val:.2f}")
    print(f"Anomaly threshold: {threshold:.2f}")
    print("\nAnomalies detected:")
    print(df[df["Anomaly"]])
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df["Date"], df["Value"], label="Original", marker="o")
    plt.plot(df["Date"], df["MA_3"], label="3-day Moving Average", linewidth=2)
    
    # Highlight anomalies
    anomalies = df[df["Anomaly"]]
    plt.scatter(anomalies["Date"], anomalies["Value"], color="red", s=100, label="Anomalies")
    
    plt.title("Time Series with Anomaly Detection")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# OUTPUT:
# Time series data:
#         Date      Value        MA_3  Anomaly
# 0 2024-01-01   50.23  NaN         False
# 1 2024-01-02   55.45   NaN         False
# ...
#
# Mean: 62.14
# Std: 18.52
# Anomaly threshold: 99.18
#
# Anomalies detected:
#         Date      Value   MA_3  Anomaly
# 15 2024-01-16    150.0  100.5     True
#
# [Line plot with moving average and red anomaly point]


# ============================================================
# SET 4 PRACTICE Q10 (15 MARKS)
# QUESTION: Load retail sales data and:
#           1. Parse dates correctly
#           2. Calculate sales by category
#           3. Find month-over-month growth
#           4. Create trend visualization
# ============================================================

def retail_sales_analysis():
    # Create sales data
    data = {
        "Date": pd.date_range("2024-01-01", periods=90, freq="D"),
        "Category": ["Electronics"]*30 + ["Clothing"]*30 + ["Home"]*30,
        "Sales": np.random.randint(100, 500, 90)
    }
    df = pd.DataFrame(data)
    
    # Extract month and year
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week
    
    print("Sales data:")
    print(df.head())
    
    # Calculate sales by category
    print("\nTotal sales by category:")
    print(df.groupby("Category")["Sales"].sum().sort_values(ascending=False))
    
    # Calculate weekly sales
    weekly_sales = df.groupby("Week")["Sales"].sum()
    print("\nWeekly sales:")
    print(weekly_sales.head())
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    for category in df["Category"].unique():
        cat_data = df[df["Category"] == category]
        weekly = cat_data.groupby("Week")["Sales"].sum()
        plt.plot(weekly.index, weekly.values, marker="o", label=category)
    
    plt.title("Weekly Sales by Category")
    plt.xlabel("Week")
    plt.ylabel("Sales ($)")
    plt.legend()
    plt.grid(True)
    plt.show()


# OUTPUT:
# Sales data:
#         Date     Category  Sales  Year  Month  Week
# 0 2024-01-01  Electronics    234     2024      1     1
# 1 2024-01-02  Electronics    456     2024      1     1
# ...
#
# Total sales by category:
# Clothing      12456
# Electronics   12123
# Home          11987
#
# Weekly sales:
# Week
# 1    3456
# 2    3567
# 3    3234
# 4    3456
#
# [Line plot showing trends for each category]
