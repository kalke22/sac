import numpy as np
import pandas as pd


# 1. Reading from CSV
def load_csv(file_path):
    df = pd.read_csv(file_path)
    return df


# 2. Reading from Excel
def load_excel(file_path, sheet_name=0):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df


# 3. Mean and Median Fill (For Numerical Columns)
def impute_numerical(df):
    # Filling with Mean
    df_mean = df.copy()
    # Select only numeric columns for mean/median to avoid errors
    numeric_cols = df_mean.select_dtypes(include=[np.number]).columns
    df_mean[numeric_cols] = df_mean[numeric_cols].fillna(df_mean[numeric_cols].mean())

    # Filling with Median
    df_median = df.copy()
    df_median[numeric_cols] = df_median[numeric_cols].fillna(
        df_median[numeric_cols].median()
    )

    return df_mean, df_median


# 4. General Fill NA (For Categorical/Transaction Data)
def fill_general_na(df, value="Unknown"):
    # Often in transaction data, we fill NaNs with a placeholder or empty string
    return df.fillna(value)


# Example Usage:
# df = load_csv('transactions.csv')
# df_filled = fill_general_na(df, value="Missing_Item")
