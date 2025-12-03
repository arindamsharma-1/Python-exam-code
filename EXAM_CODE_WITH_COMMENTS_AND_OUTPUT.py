import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# ============================================================
# SAMPLE QUESTION A (10 MARKS)
# QUESTION: Write function check_number(n) that checks:
#           1. If positive/negative/zero
#           2. If even/odd
#           3. If divisible by 5
# ============================================================

def check_number(n):
    # Check if number is positive, negative or zero
    if n > 0:
        print("Positive")  # If n greater than 0, print Positive
    elif n < 0:
        print("Negative")  # If n less than 0, print Negative
    else:
        print("Zero")  # Otherwise it is zero

    # Check if even or odd using modulo operator
    if n % 2 == 0:
        print("Even")  # If remainder is 0 when divided by 2, it's even
    else:
        print("Odd")  # Otherwise it's odd

    # Check if divisible by 5
    if n % 5 == 0:
        print("Divisible by 5")  # If remainder is 0 when divided by 5
    else:
        print("Not divisible by 5")  # Otherwise not divisible by 5


# TEST CASES AND OUTPUT:
# check_number(15)
# OUTPUT:
# Positive
# Odd
# Divisible by 5

# check_number(-8)
# OUTPUT:
# Negative
# Even
# Not divisible by 5

# check_number(0)
# OUTPUT:
# Zero
# Even
# Divisible by 5


# ============================================================
# SAMPLE QUESTION B (15 MARKS)
# QUESTION: Using tips dataset from seaborn:
#           1. Print data types of all columns
#           2. Display first 5 rows
#           3. Print statistics (describe)
#           4. Create histogram of total_bill
# ============================================================

def analyze_tips():
    # Load tips dataset from seaborn
    df = sns.load_dataset('tips')
    
    # Print column data types
    print("DATA TYPES:")
    print(df.dtypes)  # Shows: total_bill float64, tip float64, sex object, etc.
    
    # Print first 5 rows
    print("\nHEAD:")
    print(df.head())  # Shows first 5 rows of dataset
    
    # Print statistical summary
    print("\nDESCRIBE:")
    print(df.describe())  # Shows mean, std, min, max, 25%, 50%, 75% for numeric columns
    
    # Create histogram showing distribution of total_bill
    sns.histplot(data=df, x='total_bill')  # Create histogram with total_bill on x-axis
    plt.title("Histogram of Total Bill")  # Add title
    plt.xlabel("Total Bill")  # Label x-axis
    plt.ylabel("Frequency")  # Label y-axis
    plt.show()  # Display the plot

# OUTPUT:
# DATA TYPES:
# total_bill     float64
# tip            float64
# sex            object
# smoker         object
# day            object
# time           object
# size           int64
#
# HEAD:
#    total_bill   tip     sex smoker  day     time  size
# 0       16.99  1.01  Female     No  Sun  Dinner     2
# 1       10.34  1.66    Male     No  Sun  Dinner     3
# 2       21.01  3.50    Male     No  Sun  Dinner     3
# 3       23.68  3.31    Male     No  Sun  Dinner     2
# 4       24.59  3.61  Female     No  Sun  Dinner     4
#
# DESCRIBE:
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
# [Histogram plot showing distribution of total_bill amounts]


# ============================================================
# PRACTICE Q1 (10 MARKS)
# QUESTION: Write function analyze_number(n) that checks:
#           1. If prime or not prime
#           2. If > 50 or <= 50
#           3. If perfect square or not
# ============================================================

def analyze_number(n):
    # Check if prime number
    if n < 2:
        print("Not prime")  # Numbers less than 2 are not prime
    else:
        prime = True  # Assume prime until proven otherwise
        # Loop from 2 to square root of n
        for i in range(2, int(n**0.5) + 1):
            # If n is divisible by i, it's not prime
            if n % i == 0:
                prime = False
                break
        # Print result
        print("Prime" if prime else "Not prime")

    # Check if greater than 50
    if n > 50:
        print("Greater than 50")
    else:
        print("Not greater than 50")  # This includes 50

    # Check if perfect square (like 4, 9, 16, 25, etc)
    r = int(n**0.5)  # Get square root and convert to integer
    if r * r == n:
        print("Perfect square")  # If square root squared equals n
    else:
        print("Not perfect square")


# TEST CASES AND OUTPUT:
# analyze_number(25)
# OUTPUT:
# Not prime
# Not greater than 50
# Perfect square

# analyze_number(17)
# OUTPUT:
# Prime
# Not greater than 50
# Not perfect square

# analyze_number(100)
# OUTPUT:
# Not prime
# Greater than 50
# Perfect square


# ============================================================
# PRACTICE Q2 (10 MARKS)
# QUESTION: Write function filter_and_sort_even(nums) that:
#           1. Filters only even numbers from list
#           2. Sorts in descending order
#           3. Returns sorted list
# ============================================================

def filter_and_sort_even(nums):
    # Filter: keep only even numbers (where n % 2 == 0)
    evens = [x for x in nums if x % 2 == 0]
    
    # Sort: arrange in descending order (reverse=True means descending)
    evens_sorted = sorted(evens, reverse=True)
    
    # Print results for verification
    print("Original:", nums)  # Show input list
    print("Evens:", evens)  # Show filtered even numbers
    print("Sorted evens (desc):", evens_sorted)  # Show sorted descending
    
    # Return the sorted list
    return evens_sorted


# TEST CASE AND OUTPUT:
# filter_and_sort_even([12, 5, 8, 23, 16, 3, 20, 1, 14])
# OUTPUT:
# Original: [12, 5, 8, 23, 16, 3, 20, 1, 14]
# Evens: [12, 8, 16, 20, 14]
# Sorted evens (desc): [20, 16, 14, 12, 8]


# ============================================================
# PRACTICE Q3 (15 MARKS)
# QUESTION: Create NumPy array with 10 random integers (10-50):
#           1. Calculate mean, median, std dev
#           2. Find min and max values
#           3. Multiply entire array by 3
# ============================================================

def numpy_stats():
    # Create array of 10 random integers between 10 and 50 (exclusive of 50)
    arr = np.random.randint(10, 50, size=10)
    
    # Print the array
    print("Array:", arr)
    
    # Calculate and print mean (average)
    print("Mean:", np.mean(arr))
    
    # Calculate and print median (middle value)
    print("Median:", np.median(arr))
    
    # Calculate and print standard deviation (spread of data)
    print("Std dev:", np.std(arr))
    
    # Find minimum value in array
    print("Min:", np.min(arr))
    
    # Find maximum value in array
    print("Max:", np.max(arr))
    
    # Multiply each element by 3 (element-wise operation)
    print("Array * 3:", arr * 3)


# OUTPUT EXAMPLE (values will vary due to randomness):
# Array: [23 45 34 12 38 27 41 19 36 29]
# Mean: 30.4
# Median: 31.0
# Std dev: 10.32
# Min: 12
# Max: 45
# Array * 3: [69 135 102 36 114 81 123 57 108 87]


# ============================================================
# PRACTICE Q4 (15 MARKS)
# QUESTION: Create student DataFrame and:
#           1. Print full DataFrame
#           2. Filter rows where Marks > 80
#           3. Filter rows where Dept == 'CSE'
#           4. Calculate average marks by department
# ============================================================

def student_analysis():
    # Create dictionary with student data
    data = {
        "Name": ["Alice", "Bob", "Charlie", "David", "Emma", "Frank"],
        "Age":  [20,      21,    19,        22,      20,     21],
        "Marks": [85,     78,    92,        67,      88,     75],
        "Dept": ["CSE",   "ECE", "CSE",     "MECH",  "ECE",  "CSE"]
    }
    # Convert dictionary to DataFrame
    df = pd.DataFrame(data)

    # Print entire DataFrame
    print("FULL DATAFRAME:")
    print(df)  # Shows all rows and columns

    # Filter: Keep only rows where Marks > 80
    print("\nMarks > 80:")
    print(df[df["Marks"] > 80])  # Rows: Alice (85), Charlie (92), Emma (88)

    # Filter: Keep only rows where Dept == 'CSE'
    print("\nDept == 'CSE':")
    print(df[df["Dept"] == "CSE"])  # Rows: Alice, Charlie, Frank

    # Group by Dept and calculate average Marks
    print("\nAverage marks by Dept:")
    print(df.groupby("Dept")["Marks"].mean())  # CSE: 82.67, ECE: 83.0, MECH: 67.0


# OUTPUT:
# FULL DATAFRAME:
#       Name  Age  Marks    Dept
# 0    Alice   20     85     CSE
# 1      Bob   21     78     ECE
# 2  Charlie   19     92     CSE
# 3    David   22     67    MECH
# 4     Emma   20     88     ECE
# 5    Frank   21     75     CSE
#
# Marks > 80:
#       Name  Age  Marks Dept
# 0    Alice   20     85  CSE
# 2  Charlie   19     92  CSE
# 4     Emma   20     88  ECE
#
# Dept == 'CSE':
#       Name  Age  Marks Dept
# 0    Alice   20     85  CSE
# 2  Charlie   19     92  CSE
# 5    Frank   21     75  CSE
#
# Average marks by Dept:
# Dept
# CSE     82.666667
# ECE     83.000000
# MECH    67.000000
# Name: Marks, dtype: float64


# ============================================================
# PRACTICE Q5 (15 MARKS)
# QUESTION: Load Iris dataset and:
#           1. Print data types
#           2. Print first 5 rows
#           3. Print describe (statistics)
#           4. Create boxplot of sepal_length by species
# ============================================================

def iris_boxplot():
    # Load iris dataset from seaborn (built-in dataset)
    df = sns.load_dataset("iris")

    # Print data types of all columns
    print("Dtypes:")
    print(df.dtypes)  # Shows sepal_length float64, petal_length float64, etc.

    # Print first 5 rows
    print("\nHead:")
    print(df.head())  # Shows first 5 iris flowers with measurements

    # Print statistics
    print("\nDescribe:")
    print(df.describe())  # Shows mean, std, min, max for numeric columns

    # Create boxplot showing distribution by species
    sns.boxplot(data=df, x="species", y="sepal_length")
    # x="species": 3 groups on x-axis (setosa, versicolor, virginica)
    # y="sepal_length": measurements on y-axis
    
    plt.title("Sepal Length by Species")  # Add title
    plt.xlabel("Species")  # Label x-axis
    plt.ylabel("Sepal Length")  # Label y-axis
    plt.show()  # Display plot


# OUTPUT:
# Dtypes:
# sepal_length    float64
# sepal_width     float64
# petal_length    float64
# petal_width     float64
# species         object
#
# Head:
#    sepal_length  sepal_width  petal_length  petal_width species
# 0           5.1          3.5           1.4          0.2  setosa
# 1           4.9          3.0           1.4          0.2  setosa
# 2           4.7          3.2           1.3          0.2  setosa
# 3           4.6          3.1           1.5          0.2  setosa
# 4           5.0          3.6           1.4          0.2  setosa
#
# Describe:
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
# [Boxplot showing 3 boxes for setosa, versicolor, virginica]


# ============================================================
# PRACTICE Q6 (10 MARKS)
# QUESTION: Write function check_string(s) that checks:
#           1. If length > 5
#           2. If all uppercase letters
#           3. If contains any digit
# ============================================================

def check_string(s):
    # Check string length
    if len(s) > 5:
        print("Length > 5")  # Length is greater than 5 characters
    else:
        print("Length <= 5")  # Length is 5 or less characters

    # Check if all characters are uppercase
    if s.isupper():
        print("All uppercase")  # All letters are uppercase
    else:
        print("Not all uppercase")  # Has lowercase or non-letter characters

    # Check if string contains any digit
    has_digit = any(ch.isdigit() for ch in s)
    # any() returns True if any character in string is a digit
    if has_digit:
        print("Contains digit")  # Has at least one digit (0-9)
    else:
        print("No digit")  # No digits found


# TEST CASES AND OUTPUT:
# check_string("Python123")
# OUTPUT:
# Length > 5
# Not all uppercase
# Contains digit

# check_string("TEST")
# OUTPUT:
# Length <= 5
# All uppercase
# No digit

# check_string("Hi")
# OUTPUT:
# Length <= 5
# Not all uppercase
# No digit


# ============================================================
# PRACTICE Q7 (15 MARKS)
# QUESTION: Load Titanic dataset and:
#           1. Print shape (rows, columns)
#           2. Print data types
#           3. Print missing values count per column
#           4. Calculate survival rate by passenger class
#           5. Create countplot of survival by class
# ============================================================

def titanic_analysis():
    # Load titanic dataset from seaborn (ship passenger data)
    df = sns.load_dataset("titanic")

    # Print shape: number of rows and columns
    print("Shape:", df.shape)  # (891, 15) = 891 passengers, 15 columns

    # Print data types of all columns
    print("\nDtypes:")
    print(df.dtypes)  # Shows survived int64, pclass int64, sex object, etc.

    # Count missing values (NaN) in each column
    print("\nMissing values per column:")
    print(df.isnull().sum())  # Shows Age: 177 missing, Cabin: 687 missing, etc.

    # Calculate survival rate (proportion) by passenger class
    print("\nSurvival rate by pclass:")
    print(df.groupby("pclass")["survived"].mean())
    # pclass 1: ~63% survived, pclass 2: ~47%, pclass 3: ~24%

    # Create countplot: count of survived/not survived by class
    sns.countplot(data=df, x="pclass", hue="survived")
    # x="pclass": 3 groups on x-axis (1st, 2nd, 3rd class)
    # hue="survived": bars split into survived (1) and not survived (0)
    
    plt.title("Survival Count by Class")  # Add title
    plt.xlabel("Passenger Class")  # Label x-axis
    plt.ylabel("Count")  # Label y-axis
    plt.show()  # Display plot


# OUTPUT:
# Shape: (891, 15)
#
# Dtypes:
# survived       int64
# pclass         int64
# sex            object
# age          float64
# sibsp          int64
# parch          int64
# fare         float64
# embarked       object
# ...
#
# Missing values per column:
# survived        0
# pclass          0
# sex             0
# age           177
# sibsp           0
# parch           0
# fare            0
# embarked        2
# ...
#
# Survival rate by pclass:
# pclass
# 1    0.629630
# 2    0.472826
# 3    0.242363
# Name: survived, dtype: float64
#
# [Countplot showing survival distribution across 3 classes]


# ============================================================
# PRACTICE Q8 (10 MARKS)
# QUESTION: Write function classify_grade(marks) that:
#           1. Assigns grade letter (A/B/C/D/F)
#           2. Checks if passed (marks >= 40)
#           3. Checks if distinction (marks >= 75)
# ============================================================

def classify_grade(marks):
    # Assign grade based on marks range
    if marks >= 90:
        grade = "A"  # 90 and above = A
    elif marks >= 75:
        grade = "B"  # 75-89 = B
    elif marks >= 60:
        grade = "C"  # 60-74 = C
    elif marks >= 40:
        grade = "D"  # 40-59 = D
    else:
        grade = "F"  # Below 40 = F (Fail)

    # Print grade and marks
    print("Marks:", marks, "Grade:", grade)

    # Check if passed (40 or above)
    if marks >= 40:
        print("PASS")  # Passed the exam
    else:
        print("FAIL")  # Failed the exam

    # Check if distinction (75 or above)
    if marks >= 75:
        print("DISTINCTION")  # Excellent performance
    else:
        print("NO DISTINCTION")  # Did not get distinction


# TEST CASES AND OUTPUT:
# classify_grade(85)
# OUTPUT:
# Marks: 85 Grade: B
# PASS
# DISTINCTION

# classify_grade(55)
# OUTPUT:
# Marks: 55 Grade: D
# PASS
# NO DISTINCTION

# classify_grade(35)
# OUTPUT:
# Marks: 35 Grade: F
# FAIL
# NO DISTINCTION


# ============================================================
# PRACTICE Q9 (15 MARKS)
# QUESTION: Load tips dataset and:
#           1. Print shape and column names
#           2. Calculate stats: mean, median, std, min, max of tip
#           3. Count how many tips are > 3
#           4. Create histogram of tip distribution
# ============================================================

def tips_stats():
    # Load tips dataset (restaurant tips data)
    df = sns.load_dataset("tips")

    # Print shape: rows and columns
    print("Shape:", df.shape)  # (244, 7) = 244 restaurant bills, 7 columns

    # Print column names
    print("\nColumns:", df.columns.tolist())
    # ['total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size']

    # Calculate statistics on 'tip' column
    tip_mean = df["tip"].mean()  # Average tip amount
    tip_med = df["tip"].median()  # Middle value of tip
    tip_std = df["tip"].std()  # Standard deviation (spread)
    tip_min = df["tip"].min()  # Minimum tip
    tip_max = df["tip"].max()  # Maximum tip

    # Print all statistics
    print("\nTip stats:")
    print("Mean:", tip_mean)
    print("Median:", tip_med)
    print("Std:", tip_std)
    print("Min:", tip_min)
    print("Max:", tip_max)

    # Count tips greater than 3
    count_gt_3 = len(df[df["tip"] > 3])  # Filter and count
    print("\nNumber of tips > 3:", count_gt_3)

    # Create histogram showing distribution of tips
    sns.histplot(data=df, x="tip", bins=20, kde=True)
    # x="tip": tip amounts on x-axis
    # bins=20: divide into 20 ranges/bars
    # kde=True: add smooth curve over bars
    
    plt.title("Distribution of Tips")  # Add title
    plt.xlabel("Tip Amount")  # Label x-axis
    plt.ylabel("Frequency")  # Label y-axis
    plt.show()  # Display plot


# OUTPUT:
# Shape: (244, 7)
#
# Columns: ['total_bill', 'tip', 'sex', 'smoker', 'day', 'time', 'size']
#
# Tip stats:
# Mean: 2.998279
# Median: 2.9
# Std: 1.383638
# Min: 1.0
# Max: 10.0
#
# Number of tips > 3: 86
#
# [Histogram showing tip distribution with smooth curve overlay]


# ============================================================
# PRACTICE Q10 (15 MARKS)
# QUESTION: Create DataFrame with missing values (NaN) and:
#           1. Detect and show missing values
#           2. Count missing values per column
#           3. Fill missing with 0
#           4. Fill missing with column mean
#           5. Drop rows with any missing values
# ============================================================

def missing_values_demo():
    # Create data with some missing values (NaN)
    data = {
        "StudentID": [1, 2, 3, 4, 5],
        "Name":      ["Alice", "Bob", "Charlie", "David", "Emma"],
        "Marks":     [85.0, np.nan, 92.0, 78.0, np.nan],
        # np.nan represents missing value in Marks for Bob and Emma
        "Attendance":[90.0, 85.0, np.nan, 92.0, 88.0],
        # np.nan represents missing value in Attendance for Charlie
    }
    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Print original DataFrame with NaN values
    print("Original DataFrame:")
    print(df)  # Shows NaN as empty cells

    # Detect which cells have NaN (True = missing, False = present)
    print("\nIsnull():")
    print(df.isnull())  # True where NaN exists

    # Count missing values per column
    print("\nMissing count per column:")
    print(df.isnull().sum())
    # StudentID: 0, Name: 0, Marks: 2, Attendance: 1

    # Fill all NaN with 0
    print("\nFill NaN with 0:")
    print(df.fillna(0))  # Replaces all NaN with 0

    # Fill NaN with column mean (average of column)
    print("\nFill NaN with column mean (Marks and Attendance):")
    df2 = df.copy()  # Make a copy to avoid changing original
    # Calculate mean of Marks column and fill NaN
    df2["Marks"] = df2["Marks"].fillna(df2["Marks"].mean())  # mean = 85.0
    # Calculate mean of Attendance column and fill NaN
    df2["Attendance"] = df2["Attendance"].fillna(df2["Attendance"].mean())  # mean = 88.75
    print(df2)  # Shows filled values

    # Drop all rows that have any NaN value
    print("\nDrop rows with any NaN:")
    print(df.dropna())  # Only row 0 (Alice) remains, others have NaN


# OUTPUT:
# Original DataFrame:
#    StudentID     Name  Marks  Attendance
# 0          1    Alice   85.0        90.0
# 1          2      Bob    NaN        85.0
# 2          3  Charlie   92.0         NaN
# 3          4    David   78.0        92.0
# 4          5     Emma    NaN        88.0
#
# Isnull():
#    StudentID  Name  Marks  Attendance
# 0      False False  False       False
# 1      False False   True       False
# 2      False False  False        True
# 3      False False  False       False
# 4      False False   True       False
#
# Missing count per column:
# StudentID      0
# Name           0
# Marks          2
# Attendance     1
# dtype: int64
#
# Fill NaN with 0:
#    StudentID     Name  Marks  Attendance
# 0          1    Alice   85.0        90.0
# 1          2      Bob    0.0        85.0
# 2          3  Charlie   92.0         0.0
# 3          4    David   78.0        92.0
# 4          5     Emma    0.0        88.0
#
# Fill NaN with column mean:
#    StudentID     Name  Marks  Attendance
# 0          1    Alice   85.0        90.0
# 1          2      Bob   85.0        85.0
# 2          3  Charlie   92.0        88.75
# 3          4    David   78.0        92.0
# 4          5     Emma   85.0        88.0
#
# Drop rows with any NaN:
#    StudentID Name  Marks  Attendance
# 0          1 Alice   85.0        90.0
