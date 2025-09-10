"""
Constants and configuration for the churn pipeline.

Author: Tayo Akindolie
Date: 2025-07-09
"""

import os
from pathlib import Path

# Base folders for artifacts written by pipeline
MODELS_PATH: str = os.path.join(".", "models")
EDA_IMAGE_PATH: str = os.path.join(".", "images", "eda")
RESULTS_IMAGE_PATH: str = os.path.join(".", "images", "results")
DATASET_PATH: str = os.path.join(".", "data", "bank_data.csv")

LOG_FILE_PATH: str = os.path.join(".", "logs", "churn_library.log")

# Ensure paths exist at import-time
for _p in (MODELS_PATH, EDA_IMAGE_PATH, RESULTS_IMAGE_PATH):
    Path(_p).mkdir(parents=True, exist_ok=True)

# Categorical columns that will be encoded
CATEGORICAL_COLS: list[str] = [
    "Gender",
    "Education_Level",
    "Marital_Status",
    "Income_Category",
    "Card_Category"]

# Numeric + feature-engineered columns for modeling
FEATURE_COLS: list[str] = [
    "Customer_Age",
    "Dependent_count",
    "Months_on_book",
    "Total_Relationship_Count",
    "Months_Inactive_12_mon",
    "Contacts_Count_12_mon",
    "Credit_Limit",
    "Total_Revolving_Bal",
    "Avg_Open_To_Buy",
    "Total_Amt_Chng_Q4_Q1",
    "Total_Trans_Amt",
    "Total_Trans_Ct",
    "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio",

    # feature engineered columns
    "Gender_Churn",
    "Education_Level_Churn",
    "Marital_Status_Churn",
    "Income_Category_Churn",
    "Card_Category_Churn"]
