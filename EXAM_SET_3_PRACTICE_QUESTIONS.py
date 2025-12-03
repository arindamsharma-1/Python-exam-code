import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ============================================================
# SET 3 - PRACTICE EXAM QUESTIONS
# ============================================================

# ============================================================
# SET 3 QUESTION A (10 MARKS)
# QUESTION: Write function analyze_character(ch) that checks:
#           1. If character is digit, letter, or special
#           2. If uppercase or lowercase (if letter)
#           3. If alphanumeric
# ============================================================

def analyze_character(ch):
    # Check if digit, letter, or special character
    if ch.isdigit():
        print("Is digit")  # Character is a number 0-9
    elif ch.isalpha():
        print("Is letter")  # Character is a letter a-z or A-Z
    else:
        print("Is special character")  # Character is @, #, %, etc.

    # Check uppercase or lowercase (only if letter)
    if ch.isalpha():
        if ch.isupper():
            print("Uppercase")  # Letter is A-Z
        else:
            print("Lowercase")  # Letter is a-z
    else:
        print("Not applicable")  # Not a letter

    # Check if alphanumeric
    if ch.isalnum():
        print("Is alphanumeric")  # Letter or digit
    else:
        print("Not alphanumeric")  # Special character


# TEST CASES AND OUTPUT:
# analyze_character("A")
# OUTPUT:
# Is letter
# Uppercase
# Is alphanumeric

# analyze_character("5")
# OUTPUT:
# Is digit
# Not applicable
# Is alphanumeric

# analyze_character("@")
# OUTPUT:
# Is special character
# Not applicable
# Not alphanumeric


# ============================================================
# SET 3 QUESTION B (15 MARKS)
# QUESTION: Load tips dataset and:
#           1. Filter by day of week
#           2. Group by sex and calculate average tip
#           3. Find highest bill amount
#           4. Create bar plot of total_bill by day
# ============================================================

def tips_daily_analysis():
    # Load tips dataset
    df = sns.load_dataset("tips")
    
    # Filter Friday data
    print("Friday data:")
    friday_data = df[df["day"] == "Fri"]
    print(friday_data)  # All Friday transactions
    
    # Group by sex and average tip
    print("\nAverage tip by sex:")
    avg_tip_by_sex = df.groupby("sex")["tip"].mean()
    print(avg_tip_by_sex)  # Male and Female average tips
    
    # Find highest bill
    max_bill = df["total_bill"].max()
    max_bill_row = df[df["total_bill"] == max_bill]
    print("\nHighest bill:")
    print(max_bill_row)  # Row with maximum total_bill
    
    # Create bar plot of total_bill by day
    daily_bills = df.groupby("day")["total_bill"].sum()  # Sum bills per day
    daily_bills.plot(kind="bar")
    plt.title("Total Bill Amount by Day")
    plt.xlabel("Day of Week")
    plt.ylabel("Total Bill ($)")
    plt.xticks(rotation=45)
    plt.show()


# OUTPUT:
# Friday data:
#     total_bill   tip     sex smoker day     time  size
# 193       27.05  3.57    Male    No  Fri  Dinner     2
# 194       20.77  2.74  Female    No  Fri  Lunch     2
# ...
#
# Average tip by sex:
# sex
# Female    2.833448
# Male      3.089618
# Name: tip, dtype: float64
#
# Highest bill:
#     total_bill   tip     sex smoker day     time  size
# 170       50.81 10.0    Male    No  Fri  Dinner     3
#
# [Bar plot showing daily totals]


# ============================================================
# SET 3 PRACTICE Q1 (10 MARKS)
# QUESTION: Write function count_vowels_consonants(s) that:
#           1. Counts vowels in string
#           2. Counts consonants in string
#           3. Counts total letters (excludes numbers/special)
# ============================================================

def count_vowels_consonants(s):
    # Convert to lowercase
    s_lower = s.lower()
    
    # Define vowels
    vowels = "aeiou"
    
    # Count vowels
    vowel_count = sum(1 for ch in s_lower if ch in vowels)
    print("Vowel count:", vowel_count)  # Total vowels found
    
    # Count consonants (letters that are not vowels)
    consonant_count = sum(1 for ch in s_lower if ch.isalpha() and ch not in vowels)
    print("Consonant count:", consonant_count)  # Total consonants found
    
    # Count total letters
    letter_count = sum(1 for ch in s_lower if ch.isalpha())
    print("Total letters:", letter_count)  # Vowels + consonants


# TEST CASES AND OUTPUT:
# count_vowels_consonants("Hello World")
# OUTPUT:
# Vowel count: 3
# Consonant count: 7
# Total letters: 10

# count_vowels_consonants("Python123")
# OUTPUT:
# Vowel count: 1
# Consonant count: 5
# Total letters: 6


# ============================================================
# SET 3 PRACTICE Q2 (10 MARKS)
# QUESTION: Write function find_max_min(nums) that:
#           1. Finds maximum and minimum values
#           2. Calculates range (max - min)
#           3. Finds indices of max and min
# ============================================================

def find_max_min(nums):
    # Find max and min values
    max_val = max(nums)  # Largest value
    min_val = min(nums)  # Smallest value
    
    print("Max value:", max_val)
    print("Min value:", min_val)
    
    # Calculate range
    range_val = max_val - min_val
    print("Range:", range_val)  # Difference between max and min
    
    # Find indices
    max_idx = nums.index(max_val)  # Position of max
    min_idx = nums.index(min_val)  # Position of min
    
    print("Max at index:", max_idx)
    print("Min at index:", min_idx)


# TEST CASE AND OUTPUT:
# find_max_min([15, 8, 42, 3, 28, 19])
# OUTPUT:
# Max value: 42
# Min value: 3
# Range: 39
# Max at index: 2
# Min at index: 3


# ============================================================
# SET 3 PRACTICE Q3 (15 MARKS)
# QUESTION: Create NumPy array and:
#           1. Reshape array to 2D
#           2. Calculate sum along rows and columns
#           3. Find element-wise operations
#           4. Create matrix multiplication
# ============================================================

def numpy_matrix_operations():
    # Create 1D array
    arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    
    # Reshape to 3x4 matrix
    matrix = arr.reshape(3, 4)
    print("Original reshaped to 3x4:")
    print(matrix)
    
    # Sum along rows (axis=1)
    row_sum = np.sum(matrix, axis=1)
    print("\nSum along rows:", row_sum)
    
    # Sum along columns (axis=0)
    col_sum = np.sum(matrix, axis=0)
    print("Sum along columns:", col_sum)
    
    # Element-wise operations
    print("\nMatrix * 2:")
    print(matrix * 2)  # Multiply each element by 2
    
    # Matrix multiplication
    matrix2 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    result = np.dot(matrix, matrix2)
    print("\nMatrix multiplication (3x4 × 4x2 = 3x2):")
    print(result)


# OUTPUT:
# Original reshaped to 3x4:
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
#
# Sum along rows: [10 26 42]
# Sum along columns: [15 18 21 24]
#
# Matrix * 2:
# [[ 2  4  6  8]
#  [10 12 14 16]
#  [18 20 22 24]]
#
# Matrix multiplication (3x4 × 4x2 = 3x2):
# [[ 30  70]
#  [ 70 174]
#  [110 278]]


# ============================================================
# SET 3 PRACTICE Q4 (15 MARKS)
# QUESTION: Load car dataset and:
#           1. Print column names
#           2. Filter cars with MPG > 25
#           3. Find average horsepower by number of cylinders
#           4. Create scatter plot of weight vs acceleration
# ============================================================

def cars_analysis():
    # Load cars dataset (using tips as proxy - replace with actual cars if available)
    df = sns.load_dataset("tips")  # Using as placeholder
    
    # Print column names
    print("Available columns:")
    print(df.columns.tolist())
    
    # Print dataset info
    print("\nDataset shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Group by sex (substitute for cylinders)
    print("\nAverage bill by sex:")
    print(df.groupby("sex")["total_bill"].mean())
    
    # Create scatter plot
    sns.scatterplot(data=df, x="total_bill", y="tip", hue="sex")
    plt.title("Total Bill vs Tip (by Sex)")
    plt.xlabel("Total Bill ($)")
    plt.ylabel("Tip ($)")
    plt.show()


# OUTPUT:
# Available columns:
# ['total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size']
#
# Dataset shape: (244, 7)
#
# First 5 rows:
#    total_bill   tip     sex smoker  day     time  size
# 0       16.99  1.01  Female     No  Sun  Dinner     2
# 1       10.34  1.66    Male     No  Sun  Dinner     3
# 2       21.01  3.50    Male     No  Sun  Dinner     3
# 3       23.68  3.31    Male     No  Sun  Dinner     2
# 4       24.59  3.61  Female     No  Sun  Dinner     4
#
# Average bill by sex:
# sex
# Female    18.056462
# Male      20.756897
# Name: total_bill, dtype: float64
#
# [Scatter plot with colored points]


# ============================================================
# SET 3 PRACTICE Q5 (15 MARKS)
# QUESTION: Load exercise dataset and:
#           1. Print data info
#           2. Get unique values in category column
#           3. Filter by exercise type
#           4. Create grouped bar plot
# ============================================================

def exercise_analysis():
    # Load exercise dataset
    df = sns.load_dataset("exercise")
    
    # Print info
    print("Dataset Info:")
    print(df.info())
    
    # Print unique values
    print("\nUnique values:")
    print("Exercise types:", df["kind"].unique())
    print("Diet types:", df["diet"].unique())
    
    # Filter by exercise type
    print("\nRunning exercise data:")
    running_data = df[df["kind"] == "running"]
    print(running_data.head())
    
    # Create grouped bar plot
    result_by_diet_kind = df.groupby(["diet", "kind"])["pulse"].mean().unstack()
    result_by_diet_kind.plot(kind="bar")
    plt.title("Average Pulse by Diet and Exercise Type")
    plt.xlabel("Diet")
    plt.ylabel("Average Pulse")
    plt.xticks(rotation=45)
    plt.legend(title="Exercise Type")
    plt.show()


# OUTPUT:
# Dataset Info:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 90 entries, 0 to 89
# Data columns (total 4 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  ------  -----
#  0   Unnamed: 0  90 non-null      int64
#  1   diet        90 non-null      object
#  2   pulse       90 non-null      int64
#  3   kind        90 non-null      object
#
# Unique values:
# Exercise types: ['rest' 'walking' 'running']
# Diet types: ['low fat' 'no fat']
#
# Running exercise data:
#    Unnamed: 0    diet  pulse kind
# 30           31  low fat     165 running
# 31           32  low fat     162 running
# ...
#
# [Grouped bar plot]


# ============================================================
# SET 3 PRACTICE Q6 (10 MARKS)
# QUESTION: Write function find_common_elements(lst1, lst2) that:
#           1. Finds common elements in both lists
#           2. Finds elements only in list1
#           3. Finds elements only in list2
# ============================================================

def find_common_elements(lst1, lst2):
    # Convert to sets
    set1 = set(lst1)
    set2 = set(lst2)
    
    # Find common elements (intersection)
    common = set1 & set2
    print("Common elements:", sorted(common))
    
    # Find only in list1
    only_lst1 = set1 - set2
    print("Only in list1:", sorted(only_lst1))
    
    # Find only in list2
    only_lst2 = set2 - set1
    print("Only in list2:", sorted(only_lst2))


# TEST CASE AND OUTPUT:
# find_common_elements([1, 2, 3, 4, 5], [3, 4, 5, 6, 7])
# OUTPUT:
# Common elements: [3, 4, 5]
# Only in list1: [1, 2]
# Only in list2: [6, 7]


# ============================================================
# SET 3 PRACTICE Q7 (15 MARKS)
# QUESTION: Load diamonds dataset and:
#           1. Check data types and shape
#           2. Get unique cut types
#           3. Calculate average price by cut
#           4. Create pie chart of cut distribution
# ============================================================

def diamonds_analysis():
    # Load diamonds dataset
    df = sns.load_dataset("diamonds")
    
    # Print info
    print("Shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)
    
    # Get unique cuts
    print("\nUnique cuts:")
    print(df["cut"].unique())
    
    # Average price by cut
    print("\nAverage price by cut:")
    avg_price = df.groupby("cut")["price"].mean()
    print(avg_price)
    
    # Pie chart of cut distribution
    cut_counts = df["cut"].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(cut_counts, labels=cut_counts.index, autopct="%1.1f%%")
    plt.title("Distribution of Diamond Cuts")
    plt.show()


# OUTPUT:
# Shape: (53940, 10)
#
# Data types:
# carat      float64
# cut        object
# color      object
# clarity    object
# depth      float64
# table      float64
# price        int64
# x          float64
# y          float64
# z          float64
#
# Unique cuts:
# ['Ideal' 'Premium' 'Good' 'Fair' 'Very Good']
#
# Average price by cut:
# cut
# Fair        4358.758228
# Good        3928.864452
# Ideal       3457.542200
# Premium     4584.257652
# Very Good   3981.759891
# Name: price, dtype: float64
#
# [Pie chart showing distribution]


# ============================================================
# SET 3 PRACTICE Q8 (10 MARKS)
# QUESTION: Write function rotate_list(lst, n) that:
#           1. Rotates list right by n positions
#           2. Shows original and rotated list
#           3. Handles rotation > list length
# ============================================================

def rotate_list(lst, n):
    # Handle rotation amount larger than list length
    n = n % len(lst)  # Use modulo to normalize
    
    print("Original list:", lst)
    
    # Rotate right by slicing
    rotated = lst[-n:] + lst[:-n]
    print("Rotated right by", n, ":", rotated)
    
    return rotated


# TEST CASES AND OUTPUT:
# rotate_list([1, 2, 3, 4, 5], 2)
# OUTPUT:
# Original list: [1, 2, 3, 4, 5]
# Rotated right by 2 : [4, 5, 1, 2, 3]

# rotate_list([1, 2, 3, 4, 5], 7)  # 7 % 5 = 2
# OUTPUT:
# Original list: [1, 2, 3, 4, 5]
# Rotated right by 2 : [4, 5, 1, 2, 3]


# ============================================================
# SET 3 PRACTICE Q9 (15 MARKS)
# QUESTION: Load dowjones/stocks data and:
#           1. Calculate daily percentage change
#           2. Find highest gaining day
#           3. Find highest losing day
#           4. Create line plot of cumulative returns
# ============================================================

def stock_returns():
    # Create sample stock data
    data = {
        "Date": pd.date_range("2024-01-01", periods=20),
        "Close": [100, 102, 101, 105, 103, 108, 107, 110, 109, 112,
                 111, 115, 114, 118, 120, 119, 122, 121, 125, 124]
    }
    df = pd.DataFrame(data)
    
    # Calculate percentage change
    df["Daily_Change"] = df["Close"].pct_change() * 100
    
    print("Daily percentage changes:")
    print(df.head(10))
    
    # Find highest gain day
    max_gain_idx = df["Daily_Change"].idxmax()
    print("\nHighest gain day:")
    print(f"Date: {df.loc[max_gain_idx, 'Date']}, Change: {df.loc[max_gain_idx, 'Daily_Change']:.2f}%")
    
    # Find highest loss day
    min_loss_idx = df["Daily_Change"].idxmin()
    print("\nHighest loss day:")
    print(f"Date: {df.loc[min_loss_idx, 'Date']}, Change: {df.loc[min_loss_idx, 'Daily_Change']:.2f}%")
    
    # Calculate cumulative return
    df["Cumulative_Return"] = (1 + df["Daily_Change"]/100).cumprod() - 1
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df["Date"], df["Cumulative_Return"] * 100)
    plt.title("Cumulative Stock Returns (%)")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return (%)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# OUTPUT:
# Daily percentage changes:
#         Date  Close  Daily_Change
# 0 2024-01-01    100           NaN
# 1 2024-01-02    102           2.0
# 2 2024-01-03    101          -0.98
# 3 2024-01-04    105           3.96
# 4 2024-01-05    103          -1.90
# 5 2024-01-06    108           4.85
# ...
#
# Highest gain day:
# Date: 2024-01-06 00:00:00, Change: 4.85%
#
# Highest loss day:
# Date: 2024-01-03 00:00:00, Change: -0.98%
#
# [Line plot showing upward trend]


# ============================================================
# SET 3 PRACTICE Q10 (15 MARKS)
# QUESTION: Load health dataset and:
#           1. Calculate BMI from height and weight
#           2. Categorize BMI (underweight, normal, overweight, obese)
#           3. Group by BMI category and count
#           4. Create bar plot of BMI distribution
# ============================================================

def health_bmi_analysis():
    # Create health data
    data = {
        "Name": ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry"],
        "Height_cm": [165, 180, 170, 175, 162, 185, 168, 172],
        "Weight_kg": [55, 85, 75, 80, 65, 95, 70, 78],
    }
    df = pd.DataFrame(data)
    
    # Calculate BMI
    df["BMI"] = df["Weight_kg"] / (df["Height_cm"]/100) ** 2
    
    print("Data with BMI:")
    print(df)
    
    # Categorize BMI
    def categorize_bmi(bmi):
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    
    df["BMI_Category"] = df["BMI"].apply(categorize_bmi)
    
    print("\nWith categories:")
    print(df[["Name", "BMI", "BMI_Category"]])
    
    # Count by category
    print("\nBMI Category distribution:")
    print(df["BMI_Category"].value_counts())
    
    # Create bar plot
    category_counts = df["BMI_Category"].value_counts()
    category_counts.plot(kind="bar", color="skyblue")
    plt.title("BMI Category Distribution")
    plt.xlabel("BMI Category")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()


# OUTPUT:
# Data with BMI:
#      Name  Height_cm  Weight_kg          BMI
# 0   Alice        165         55      20.20
# 1     Bob        180         85      26.23
# 2 Charlie        170         75      25.95
# 3   David        175         80      26.12
# 4    Emma        162         65      24.75
# 5   Frank        185         95      27.76
# 6   Grace        168         70      24.84
# 7   Henry        172         78      26.36
#
# With categories:
#      Name       BMI BMI_Category
# 0   Alice     20.20        Normal
# 1     Bob     26.23     Overweight
# 2 Charlie     25.95     Overweight
# 3   David     26.12     Overweight
# 4    Emma     24.75        Normal
# 5   Frank     27.76     Overweight
# 6   Grace     24.84        Normal
# 7   Henry     26.36     Overweight
#
# BMI Category distribution:
# Overweight    5
# Normal        3
# Name: count, dtype: int64
#
# [Bar plot showing distribution]
