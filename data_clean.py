import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.impute import KNNImputer
import re

# Configure plotting and warnings
plt.rcParams["figure.figsize"] = (12, 6)
sns.set_style("whitegrid")
warnings.filterwarnings("ignore")
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def duplicate_values(df):
    """Check and remove duplicate rows from dataframe"""
    print("Duplicate check...")
    num_duplicates = df.duplicated(subset=None, keep='first').sum()
    if num_duplicates > 0:
        print(f"There are {num_duplicates} duplicated observations in the dataset.")
        df.drop_duplicates(keep='first', inplace=True)
        print(f"{num_duplicates} duplicates were dropped!")
    else:
        print("There are no duplicated observations in the dataset.")
    return df

def non_numeric_values(df, column_name):
    """Find non-numeric values in a column"""
    pattern = r'\D+'
    non_numeric_values = df[column_name].astype(str).str.findall(pattern)
    non_numeric_values = [item for sublist in non_numeric_values for item in sublist]
    return set(non_numeric_values)

def knn_impute_column(df, column_name, n_neighbors=5):
    """Impute missing values using KNN"""
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df[[column_name]] = imputer.fit_transform(df[[column_name]])
    return df

def clean_data(filepath):
    # Read data
    df = pd.read_csv(filepath)
    
    # Drop unnecessary columns
    df.drop(['ID','Customer_ID','Month','Name','SSN', 'Type_of_Loan', 
             'Changed_Credit_Limit', 'Monthly_Inhand_Salary'], axis=1, inplace=True)

    # Replace empty values with NaN
    df.replace("_", "", regex=True, inplace=True)
    df.replace('', np.nan, inplace=True)

    # Clean Age
    df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
    df = df[(df['Age'] <= 100) & (df['Age'] >= 0)]

    # Clean Occupation
    df['Occupation'].fillna('Other', inplace=True)

    # Clean numeric columns
    numeric_cols = ['Num_Bank_Accounts', 'Num_of_Loan', 'Delay_from_due_date',
                    'Num_of_Delayed_Payment', 'Num_Credit_Inquiries', 
                    'Amount_invested_monthly', 'Monthly_Balance']
                   
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df[df[col] >= 0]
        df = knn_impute_column(df, col)

    # Clean Credit Mix
    df['Credit_Mix'].fillna('Unknown', inplace=True)

    # Clean Credit History Age
    mode_by_credit_mix = df.groupby('Credit_Mix')['Credit_History_Age'].transform(lambda x: x.mode()[0])
    df['Credit_History_Age'].fillna(mode_by_credit_mix, inplace=True)
    
    # Convert Credit History Age to months
    df['Credit_History_Years'] = df['Credit_History_Age'].str.extract(r'(\d+) Years').astype(float)
    df['Credit_History_Months'] = df['Credit_History_Age'].str.extract(r'(\d+) Months').astype(float)
    df['Credit_History_Age_Months'] = df['Credit_History_Years'] * 12 + df['Credit_History_Months']
    df.drop(columns=['Credit_History_Age', 'Credit_History_Years', 'Credit_History_Months'], inplace=True)

    # Clean Payment Behavior
    df['Payment_Behaviour'].replace('!@9#%8', pd.NA, inplace=True)
    mode_value = df['Payment_Behaviour'].mode()[0]
    df['Payment_Behaviour'].fillna(mode_value, inplace=True)
    
    # Format text columns
    df['Payment_Behaviour'] = df['Payment_Behaviour'].str.replace('Lowspent', 'Low_spent_')
    df['Payment_Behaviour'] = df['Payment_Behaviour'].str.replace('Highspent', 'High_spent_')
    df['Payment_Behaviour'] = df['Payment_Behaviour'].str.replace('Smallvalue', 'Small_value_')
    df['Payment_Behaviour'] = df['Payment_Behaviour'].str.replace('Largevalue', 'Large_value_')
    df['Payment_Behaviour'] = df['Payment_Behaviour'].str.replace('Mediumvalue', 'Medium_value_')

    # Convert dtypes
    df['Annual_Income'] = df['Annual_Income'].astype(float)
    df['Outstanding_Debt'] = df['Outstanding_Debt'].astype(float)

    # Add monthly expense column
    df['Monthly_expense'] = df['Annual_Income'] / 12 - df['Monthly_Balance']
    df['Monthly_expense'] = df['Monthly_expense'].astype(float)

    return df

if __name__ == "__main__":
    input_path = r"C:\Users\HP\Downloads\Industry-Baby\datasets\test.csv"
    output_path = "Creditscore_test_cleaned.csv"
    
    df = clean_data(input_path)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")