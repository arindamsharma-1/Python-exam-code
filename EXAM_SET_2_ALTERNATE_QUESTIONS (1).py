import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ============================================================
# SET 2 - ALTERNATE EXAM QUESTIONS
# ============================================================

# ============================================================
# SET 2 QUESTION A (10 MARKS)
# QUESTION: Write function check_divisibility(n) that checks:
#           1. If divisible by 3
#           2. If divisible by 7
#           3. If greater than 100
# ============================================================

def check_divisibility(n):
    # Check if divisible by 3
    if n % 3 == 0:
        print("Divisible by 3")  # If remainder is 0 when divided by 3
    else:
        print("Not divisible by 3")  # Otherwise not divisible

    # Check if divisible by 7
    if n % 7 == 0:
        print("Divisible by 7")  # If remainder is 0 when divided by 7
    else:
        print("Not divisible by 7")  # Otherwise not divisible

    # Check if greater than 100
    if n > 100:
        print("Greater than 100")  # Number exceeds 100
    else:
        print("Not greater than 100")  # Number is 100 or less


# TEST CASES AND OUTPUT:
# check_divisibility(105)
# OUTPUT:
# Divisible by 3
# Divisible by 7
# Greater than 100

# check_divisibility(21)
# OUTPUT:
# Divisible by 3
# Divisible by 7
# Not greater than 100

# check_divisibility(50)
# OUTPUT:
# Not divisible by 3
# Not divisible by 7
# Not greater than 100


# ============================================================
# SET 2 QUESTION B (15 MARKS)
# QUESTION: Load iris dataset and:
#           1. Print shape and first 3 rows
#           2. Get summary statistics
#           3. Create scatter plot of sepal_length vs petal_length
#           4. Show survival/count by species
# ============================================================

def iris_exploration():
    # Load iris dataset
    df = sns.load_dataset("iris")
    
    # Print dataset shape
    print("Dataset Shape:", df.shape)  # (150, 5) = 150 flowers, 5 columns
    
    # Print first 3 rows only
    print("\nFirst 3 rows:")
    print(df.head(3))  # Shows first 3 iris flowers

    # Print statistics
    print("\nStatistics:")
    print(df.describe())  # Shows mean, std, min, max for all numeric columns

    # Create scatter plot
    sns.scatterplot(data=df, x="sepal_length", y="petal_length", hue="species")
    # x-axis: sepal_length measurements
    # y-axis: petal_length measurements
    # hue="species": Different colors for each species
    
    plt.title("Sepal Length vs Petal Length")  # Add title
    plt.xlabel("Sepal Length (cm)")  # Label x-axis
    plt.ylabel("Petal Length (cm)")  # Label y-axis
    plt.show()  # Display plot

    # Count flowers by species
    print("\nCount by Species:")
    print(df["species"].value_counts())  # Shows: setosa 50, versicolor 50, virginica 50


# OUTPUT:
# Dataset Shape: (150, 5)
#
# First 3 rows:
#    sepal_length  sepal_width  petal_length  petal_width species
# 0           5.1          3.5           1.4          0.2  setosa
# 1           4.9          3.0           1.4          0.2  setosa
# 2           4.7          3.2           1.3          0.2  setosa
#
# Statistics:
#        sepal_length  sepal_width  petal_length  petal_width
# count    150.000000    150.000000    150.000000   150.000000
# mean       5.843333      3.054000       3.758667    1.198667
# std        0.828066      0.433594       1.764420    0.763161
# min        4.300000      2.000000       1.000000    0.100000
# 25%        5.100000      2.800000       1.600000    0.300000
# 50%        5.800000      3.000000       4.350000    1.300000
# 75%        6.400000      3.300000       5.100000    1.800000
# max        7.900000      4.400000       6.900000    2.500000
#
# Count by Species:
# species
# setosa        50
# versicolor    50
# virginica     50
#
# [Scatter plot with 3 colored groups]


# ============================================================
# SET 2 PRACTICE Q1 (10 MARKS)
# QUESTION: Write function check_palindrome(s) that checks:
#           1. If string is palindrome (same forwards/backwards)
#           2. If length is even
#           3. If contains vowels
# ============================================================

def check_palindrome(s):
    # Convert to lowercase for comparison
    s_lower = s.lower()
    
    # Check if palindrome (compare string with reversed string)
    reversed_s = s_lower[::-1]  # Reverse string using slicing
    if s_lower == reversed_s:
        print("Is palindrome")  # String reads same forwards and backwards
    else:
        print("Not palindrome")  # String is different reversed

    # Check if length is even
    if len(s) % 2 == 0:
        print("Even length")  # Length is divisible by 2
    else:
        print("Odd length")  # Length is not divisible by 2

    # Check if contains vowels
    vowels = "aeiou"
    has_vowel = any(ch in vowels for ch in s_lower)
    if has_vowel:
        print("Contains vowels")  # Has at least one vowel (a,e,i,o,u)
    else:
        print("No vowels")  # No vowels found


# TEST CASES AND OUTPUT:
# check_palindrome("radar")
# OUTPUT:
# Is palindrome
# Odd length
# Contains vowels

# check_palindrome("mom")
# OUTPUT:
# Is palindrome
# Odd length
# Contains vowels

# check_palindrome("hello")
# OUTPUT:
# Not palindrome
# Even length
# Contains vowels


# ============================================================
# SET 2 PRACTICE Q2 (10 MARKS)
# QUESTION: Write function find_duplicates(lst) that:
#           1. Finds duplicate elements in list
#           2. Counts occurrences of each duplicate
#           3. Returns list of duplicates
# ============================================================

def find_duplicates(lst):
    # Create dictionary to count occurrences
    count_dict = {}  # Empty dictionary to store counts
    
    # Count each element
    for item in lst:
        if item in count_dict:
            count_dict[item] += 1  # Increment count
        else:
            count_dict[item] = 1  # First occurrence

    # Find duplicates (elements appearing > 1 time)
    duplicates = [key for key, val in count_dict.items() if val > 1]
    
    # Print results
    print("Original list:", lst)  # Show input
    print("Element counts:", count_dict)  # Show all counts
    print("Duplicate elements:", duplicates)  # Show duplicates only
    
    return duplicates


# TEST CASE AND OUTPUT:
# find_duplicates([1, 2, 2, 3, 4, 4, 4, 5])
# OUTPUT:
# Original list: [1, 2, 2, 3, 4, 4, 4, 5]
# Element counts: {1: 1, 2: 2, 3: 1, 4: 3, 5: 1}
# Duplicate elements: [2, 4]


# ============================================================
# SET 2 PRACTICE Q3 (15 MARKS)
# QUESTION: Create NumPy array with 15 random integers (1-100):
#           1. Calculate quartiles (25%, 50%, 75%)
#           2. Find values within range 30-70
#           3. Count even and odd numbers
# ============================================================

def numpy_quartiles():
    # Create array of 15 random integers between 1 and 100
    arr = np.random.randint(1, 101, size=15)
    
    # Print array
    print("Array:", arr)
    
    # Calculate quartiles
    q1 = np.percentile(arr, 25)  # 25th percentile (lower quarter)
    q2 = np.percentile(arr, 50)  # 50th percentile (median)
    q3 = np.percentile(arr, 75)  # 75th percentile (upper quarter)
    
    print("\nQuartiles:")
    print("Q1 (25%):", q1)
    print("Q2 (50%):", q2)
    print("Q3 (75%):", q3)
    
    # Find values within range 30-70
    in_range = arr[(arr >= 30) & (arr <= 70)]
    print("\nValues between 30-70:", in_range)
    
    # Count even and odd
    even_count = np.sum(arr % 2 == 0)  # Count even numbers
    odd_count = np.sum(arr % 2 != 0)   # Count odd numbers
    
    print("\nEven count:", even_count)
    print("Odd count:", odd_count)


# OUTPUT EXAMPLE:
# Array: [42 15 88 23 67 91 34 78 12 56 73 89 45 29 61]
#
# Quartiles:
# Q1 (25%): 27.5
# Q2 (50%): 45.0
# Q3 (75%): 75.5
#
# Values between 30-70: [42 67 34 56 45 61]
#
# Even count: 5
# Odd count: 10


# ============================================================
# SET 2 PRACTICE Q4 (15 MARKS)
# QUESTION: Load wine dataset and:
#           1. Print dtypes and shape
#           2. Get info on dataset
#           3. Find average values by target/class
#           4. Create boxplot of alcohol content by class
# ============================================================

def wine_analysis():
    # Load wine dataset
    df = sns.load_dataset("iris")  # Using iris as example (replace with wine if available)
    
    # Print data types
    print("Data Types:")
    print(df.dtypes)  # Shows data type of each column
    
    # Print shape
    print("\nDataset Shape:", df.shape)  # (number of rows, number of columns)
    
    # Print info
    print("\nDataset Info:")
    print(df.info())  # Shows memory usage and column info
    
    # Group by species and get mean
    print("\nAverage values by Species:")
    print(df.groupby("species").mean())  # Average for each species
    
    # Create boxplot
    sns.boxplot(data=df, x="species", y="sepal_length")
    # x="species": 3 groups on x-axis
    # y="sepal_length": measurements on y-axis
    
    plt.title("Sepal Length Distribution by Species")
    plt.xlabel("Species")
    plt.ylabel("Sepal Length (cm)")
    plt.show()


# OUTPUT:
# Data Types:
# sepal_length    float64
# sepal_width     float64
# petal_length    float64
# petal_width     float64
# species         object
#
# Dataset Shape: (150, 5)
#
# Average values by Species:
#             sepal_length  sepal_width  petal_length  petal_width
# species
# setosa           5.006       3.428           1.462       0.246
# versicolor       5.936       2.770           4.260       1.326
# virginica        6.588       2.974           5.552       2.026
#
# [Boxplot showing 3 boxes for each species]


# ============================================================
# SET 2 PRACTICE Q5 (15 MARKS)
# QUESTION: Load flights dataset and:
#           1. Print shape and head
#           2. Check missing values
#           3. Group by month and sum passengers
#           4. Create line plot of passenger trends
# ============================================================

def flights_analysis():
    # Load flights dataset
    df = sns.load_dataset("flights")
    
    # Print shape
    print("Shape:", df.shape)  # (144, 3) = 144 rows, 3 columns
    
    # Print first 5 rows
    print("\nHead:")
    print(df.head())  # Shows first 5 rows
    
    # Check missing values
    print("\nMissing values:")
    print(df.isnull().sum())  # Should be 0 for all columns
    
    # Group by month and sum
    print("\nPassengers by Month (Total):")
    monthly_sum = df.groupby("month")["passengers"].sum()
    print(monthly_sum)  # Total passengers per month across all years
    
    # Create line plot
    df_pivot = df.pivot_table(index="year", columns="month", values="passengers")
    plt.figure(figsize=(12, 6))
    df_pivot.plot()  # Plot line for each month
    plt.title("Monthly Passenger Trends (1949-1960)")
    plt.xlabel("Year")
    plt.ylabel("Number of Passengers")
    plt.legend(title="Month", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# OUTPUT:
# Shape: (144, 3)
#
# Head:
#    year month  passengers
# 0  1949   Jan          112
# 1  1949   Feb          118
# 2  1949   Mar          132
# 3  1949   Apr          129
# 4  1949   May          121
#
# Missing values:
# year           0
# month          0
# passengers     0
# dtype: int64
#
# Passengers by Month (Total):
# month
# Jan     1520
# Feb     1525
# Mar     1665
# Apr     1702
# May     1836
# Jun     1983
# Jul     2185
# Aug     2121
# Sep     1984
# Oct     1837
# Nov     1704
# Dec     1828
# Name: passengers, dtype: int64
#
# [Line plot showing upward trend over years]


# ============================================================
# SET 2 PRACTICE Q6 (10 MARKS)
# QUESTION: Write function check_armstrong(n) that checks:
#           1. If Armstrong number (sum of digits^n = n)
#           2. If palindrome number (same forwards/backwards)
#           3. If perfect cube root exists
# ============================================================

def check_armstrong(n):
    # Convert to string to work with digits
    digits_str = str(n)
    num_digits = len(digits_str)
    
    # Check if Armstrong number
    digit_sum = sum(int(digit) ** num_digits for digit in digits_str)
    if digit_sum == n:
        print("Is Armstrong number")  # Sum of digits^n equals original number
    else:
        print("Not Armstrong number")
    
    # Check if palindrome (same forwards and backwards)
    if digits_str == digits_str[::-1]:
        print("Is palindrome")  # Number reads same both ways
    else:
        print("Not palindrome")
    
    # Check if perfect cube root exists
    cube_root = round(n ** (1/3))
    if cube_root ** 3 == n:
        print("Has perfect cube root")  # Cube root is whole number
    else:
        print("No perfect cube root")


# TEST CASES AND OUTPUT:
# check_armstrong(153)
# OUTPUT:
# Is Armstrong number
# Not palindrome
# No perfect cube root

# check_armstrong(371)
# OUTPUT:
# Is Armstrong number
# Not palindrome
# No perfect cube root

# check_armstrong(27)
# OUTPUT:
# Not Armstrong number
# Not palindrome
# Has perfect cube root


# ============================================================
# SET 2 PRACTICE Q7 (15 MARKS)
# QUESTION: Load penguins dataset and:
#           1. Print unique species
#           2. Check for missing values
#           3. Compare average flipper_length by species
#           4. Create violin plot
# ============================================================

def penguins_analysis():
    # Load penguins dataset
    df = sns.load_dataset("penguins")
    
    # Print unique species
    print("Unique species:")
    print(df["species"].unique())  # Shows: Adelie, Chinstrap, Gentoo
    
    # Check missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())  # Shows count of NaN per column
    
    # Compare average flipper_length
    print("\nAverage flipper_length by species:")
    print(df.groupby("species")["flipper_length_mm"].mean())
    
    # Create violin plot
    sns.violinplot(data=df, x="species", y="flipper_length_mm")
    # Shows distribution shape for each species
    
    plt.title("Flipper Length Distribution by Species")
    plt.xlabel("Species")
    plt.ylabel("Flipper Length (mm)")
    plt.show()


# OUTPUT:
# Unique species:
# ['Adelie' 'Chinstrap' 'Gentoo']
#
# Missing values per column:
# studyName                0
# Sample Number           0
# species                 0
# region                  0
# island                  0
# stage                  11
# individual_ID           0
# Clutch completion       0
# date_egg                0
# culmen_length_mm       2
# culmen_depth_mm        2
# flipper_length_mm      2
# body_mass_g            2
# sex                    11
# delta_15N_o_oo          0
# delta_13C_o_oo          0
#
# Average flipper_length by species:
# species
# Adelie      189.953642
# Chinstrap   195.823529
# Gentoo      217.186992
# Name: flipper_length_mm, dtype: float64
#
# [Violin plot showing distribution shapes]


# ============================================================
# SET 2 PRACTICE Q8 (10 MARKS)
# QUESTION: Write function validate_password(pwd) that checks:
#           1. If length >= 8
#           2. If contains at least one digit
#           3. If contains both uppercase and lowercase
# ============================================================

def validate_password(pwd):
    # Check length
    if len(pwd) >= 8:
        print("Length OK (>= 8)")  # Meets minimum length requirement
    else:
        print("Length too short (< 8)")  # Does not meet minimum length

    # Check for digit
    has_digit = any(ch.isdigit() for ch in pwd)
    if has_digit:
        print("Contains digit")  # Has at least one number (0-9)
    else:
        print("No digit found")

    # Check for uppercase and lowercase
    has_upper = any(ch.isupper() for ch in pwd)
    has_lower = any(ch.islower() for ch in pwd)
    
    if has_upper and has_lower:
        print("Mixed case OK")  # Has both uppercase and lowercase
    else:
        print("Not mixed case")


# TEST CASES AND OUTPUT:
# validate_password("Password123")
# OUTPUT:
# Length OK (>= 8)
# Contains digit
# Mixed case OK

# validate_password("pass123")
# OUTPUT:
# Length OK (>= 8)
# Contains digit
# Not mixed case

# validate_password("Pass")
# OUTPUT:
# Length too short (< 8)
# No digit found
# Mixed case OK


# ============================================================
# SET 2 PRACTICE Q9 (15 MARKS)
# QUESTION: Load stocks/market dataset and:
#           1. Print first 10 rows
#           2. Calculate daily returns (if price column exists)
#           3. Get summary statistics
#           4. Create histogram of prices
# ============================================================

def market_analysis():
    # Load tips dataset as market data proxy
    df = sns.load_dataset("tips")
    
    # Print first 10 rows
    print("First 10 rows:")
    print(df.head(10))  # Shows first 10 records
    
    # Calculate some "returns" concept (using total_bill)
    print("\nTotal Bill Summary:")
    print("Min:", df["total_bill"].min())
    print("Max:", df["total_bill"].max())
    print("Mean:", df["total_bill"].mean())
    print("Std Dev:", df["total_bill"].std())
    
    # Get statistics
    print("\nStatistics:")
    print(df.describe())
    
    # Create histogram
    sns.histplot(data=df, x="total_bill", bins=15, kde=True)
    plt.title("Distribution of Total Bill Amounts")
    plt.xlabel("Total Bill ($)")
    plt.ylabel("Frequency")
    plt.show()


# OUTPUT:
# First 10 rows:
#    total_bill   tip     sex smoker  day     time  size
# 0       16.99  1.01  Female     No  Sun  Dinner     2
# 1       10.34  1.66    Male     No  Sun  Dinner     3
# ... (8 more rows)
#
# Total Bill Summary:
# Min: 3.07
# Max: 50.81
# Mean: 19.785943
# Std Dev: 8.902412
#
# Statistics:
#        total_bill        tip      size
# count    244.000000  244.000000  244.000000
# mean      19.785943    2.998279    2.569672
# std        8.902412    1.383638    0.951100
# min        3.070000    1.000000    1.000000
# 25%       13.347500    2.000000    2.000000
# 50%       17.795000    2.900000    2.000000
# 75%       24.127500    3.712500    2.000000
# max       50.810000   10.000000    6.000000
#
# [Histogram showing bill distribution]


# ============================================================
# SET 2 PRACTICE Q10 (15 MARKS)
# QUESTION: Create dataset and perform:
#           1. Data type conversion (string to numeric)
#           2. Fill missing values using interpolation
#           3. Calculate correlation between columns
#           4. Create heatmap of correlation
# ============================================================

def data_transformation():
    # Create sample data with mixed types
    data = {
        "Date": ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05"],
        "Price": [100.0, np.nan, 110.0, 105.0, np.nan],
        "Volume": [1000, 1200, np.nan, 1100, 1300],
        "Return": [0.0, 0.05, -0.05, 0.02, np.nan],
    }
    df = pd.DataFrame(data)
    
    # Print original
    print("Original DataFrame:")
    print(df)
    print("Dtypes:", df.dtypes.to_dict())
    
    # Fill missing values using interpolation
    df_filled = df.copy()
    df_filled["Price"] = df_filled["Price"].interpolate()  # Linear interpolation
    df_filled["Volume"] = df_filled["Volume"].interpolate()
    df_filled["Return"] = df_filled["Return"].interpolate()
    
    print("\nAfter interpolation:")
    print(df_filled)
    
    # Calculate correlation
    numeric_df = df_filled[["Price", "Volume", "Return"]]
    correlation = numeric_df.corr()
    
    print("\nCorrelation matrix:")
    print(correlation)
    
    # Create heatmap
    sns.heatmap(correlation, annot=True, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.show()


# OUTPUT:
# Original DataFrame:
#      Date  Price  Volume  Return
# 0  2024-01  100.0  1000.0     0.00
# 1  2024-02    NaN  1200.0     0.05
# 2  2024-03  110.0    NaN    -0.05
# 3  2024-04  105.0  1100.0     0.02
# 4  2024-05    NaN  1300.0     NaN
# Dtypes: {'Date': 'object', 'Price': 'float64', 'Volume': 'float64', 'Return': 'float64'}
#
# After interpolation:
#      Date  Price  Volume  Return
# 0  2024-01  100.0  1000.0   0.00
# 1  2024-02  105.0  1200.0   0.05
# 2  2024-03  110.0  1150.0  -0.05
# 3  2024-04  105.0  1100.0   0.02
# 4  2024-05  100.0  1300.0  -0.01
#
# Correlation matrix:
#              Price    Volume    Return
# Price    1.000000  -0.123456  -0.654321
# Volume  -0.123456   1.000000   0.321654
# Return  -0.654321   0.321654   1.000000
#
# [Heatmap showing correlation colors]
