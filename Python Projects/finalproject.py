# ---3---
#    ---Titanic Survival Analysis---

# Description:Perform exploratory data analysis (EDA) on the Titanic dataset to understand the factors that influenced passenger 
# survival. 

# Create visualizations to represent survival rates by class, gender, age, etc.

# Dataset: Titanic Dataset
#  is available on Kaggle.

# Tools/Libraries: Pandas for data manipulation, Matplotlib and Seaborn for visualizations.


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_titanic_data():
    while True:
        file_input = input("Please enter the CSV file name (e.g., 'titanic.csv') or full path (or press Enter to exit): ").strip()
        
        if not file_input:
            print("No file name or path provided. Exiting.")
            exit()
        
        if os.path.isabs(file_input):
            file_path = file_input
        else:
            file_path = os.path.join(os.getcwd(), file_input)
        
        if os.path.exists(file_path):
            print(f"Loading dataset from '{file_path}'.")
            return pd.read_csv(file_path)
        else:
            print(f"Error: File '{file_path}' not found. Please try again.")


try:
    titanic = load_titanic_data()
    print("Dataset loaded successfully!\n")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()


print("--- Basic Information ---")
print(titanic.info())

print("\n--- First 5 Rows ---")
print(titanic.head())

print("\n--- Missing Values ---")
print(titanic.isnull().sum())


numeric_cols = titanic.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    titanic[col] = titanic[col].fillna(titanic[col].mean())


categorical_cols = titanic.select_dtypes(include=['object']).columns
for col in categorical_cols:
    titanic[col] = titanic[col].fillna(titanic[col].mode()[0] if not titanic[col].mode().empty else 'Unknown')

irrelevant_cols = ['Cabin', 'Ticket']
for col in irrelevant_cols:
    if col in titanic.columns:
        titanic = titanic.drop(columns=col)

print("\n--- Cleaned Data Overview ---")
print(titanic.info())

# Exploratory Data Analysis ---

# Survival Distribution
if 'Survived' in titanic.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=titanic, x='Survived', palette='Set2')
    plt.title('Survival Distribution')
    plt.xlabel('Survived (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    plt.show()
else:
    print("Column 'Survived' not found. Skipping survival distribution plot.")

# Passenger Class Distribution
if 'Pclass' in titanic.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=titanic, x='Pclass', palette='Set2')
    plt.title('Passenger Class Distribution')
    plt.xlabel('Passenger Class')
    plt.ylabel('Count')
    plt.show()
else:
    print("Column 'Pclass' not found. Skipping passenger class distribution plot.")

# Gender Distribution
if 'Sex' in titanic.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=titanic, x='Sex', palette='Set1')
    plt.title('Gender Distribution')
    plt.xlabel('Gender')
    plt.ylabel('Count')
    plt.show()
else:
    print("Column 'Sex' not found. Skipping gender distribution plot.")

# Age Distribution
if 'Age' in titanic.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=titanic, x='Age', kde=True, color='blue')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.show()
else:
    print("Column 'Age' not found. Skipping age distribution plot.")

# Fare Distribution
if 'Fare' in titanic.columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(data=titanic, x='Fare', kde=True, color='green')
    plt.title('Fare Distribution')
    plt.xlabel('Fare')
    plt.ylabel('Frequency')
    plt.show()
else:
    print("Column 'Fare' not found. Skipping fare distribution plot.")

# Embarked Port Distribution
if 'Embarked' in titanic.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=titanic, x='Embarked', palette='pastel')
    plt.title('Embarked Port Distribution')
    plt.xlabel('Port of Embarkation')
    plt.ylabel('Count')
    plt.show()
else:
    print("Column 'Embarked' not found. Skipping embarked port distribution plot.")

# Correlation Heatmap
if len(titanic.select_dtypes(include=['float64', 'int64']).columns) > 1:
    plt.figure(figsize=(8, 5))
    sns.heatmap(titanic.corr(numeric_only=True), annot=True, cmap='magma', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()
else:
    print("Not enough numeric columns for correlation heatmap.")

print("\n--- Titanic EDA Completed Successfully ---")