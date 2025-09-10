"""
Module defining functions and configuration for the churn pipeline.

Author: Tayo Akindolie
Date: 2025-07-09
"""


# import libraries
import os
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay, classification_report

from constants import (
    DATASET_PATH,
    EDA_IMAGE_PATH,
    CATEGORICAL_COLS,
    FEATURE_COLS,
    RESULTS_IMAGE_PATH,
    MODELS_PATH,
)

os.environ["QT_QPA_PLATFORM"] = "offscreen"

sns.set_theme()


def import_data(pth: str) -> Optional[pd.DataFrame]:
    """
    Load a CSV file and create a binary `Churn` target column.

    This converts `Attrition_Flag` to a numeric `Churn` column
    (1 = Attrited Customer, 0 = Existing Customer).

    Args:
        pth (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: DataFrame with the original columns and a new `Churn` column.
    """

    try:
        df = pd.read_csv(pth)
        df["Churn"] = df["Attrition_Flag"].apply(
            lambda value: 0 if value == "Existing Customer" else 1)

        return df
    except FileNotFoundError:
        print("File not found")
        return None


def perform_eda(df: pd.DataFrame) -> None:
    """
    Create EDA plots and save them under `EDA_IMAGE_PATH`.

    Generates:
        - Churn distribution
        - Customer age distribution
        - Marital status distribution
        - Total transaction count distribution
        - Correlation heatmap (numeric columns)

    Args:
        df (pandas.DataFrame): Input data containing `Churn` and other fields.
    """
    figsize = (20, 10)

    # Figure 1: Churn distribution
    fig = plt.figure(figsize=figsize)
    df["Churn"].hist()
    fig.savefig(os.path.join(EDA_IMAGE_PATH, "churn_dist.png"))
    plt.close(fig)

    # Figure 2: Age distribution
    fig = plt.figure(figsize=figsize)
    df["Customer_Age"].hist()
    fig.savefig(os.path.join(EDA_IMAGE_PATH, "age_dist.png"))
    plt.close(fig)

    # Figure 3: Marital status distribution
    fig = plt.figure(figsize=figsize)
    df["Marital_Status"].value_counts(normalize=True).plot(kind="bar")
    fig.savefig(os.path.join(EDA_IMAGE_PATH, "marital_status_dist.png"))
    plt.close(fig)

    # Figure 4: Total transaction count distribution
    fig = plt.figure(figsize=figsize)
    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True)
    fig.savefig(os.path.join(EDA_IMAGE_PATH, "total_trans_ct_dist.png"))
    plt.close(fig)

    # Figure 5: Correlations
    fig = plt.figure(figsize=figsize)
    sns.heatmap(
        df.select_dtypes(
            exclude=["object"]).corr(),
        annot=False,
        cmap="Dark2_r",
        linewidths=2)
    fig.savefig(os.path.join(EDA_IMAGE_PATH, "corr_heatmap.png"))
    plt.close(fig)


def encoder_helper(
        df: pd.DataFrame,
        category_lst: list[str] = CATEGORICAL_COLS,
        response: str = "Churn") -> pd.DataFrame:
    """
    Target-rate encode categorical columns as `<col>_Churn`.

    For each categorical column, this maps each category to the mean
    churn rate and appends a numeric `<col>_Churn` column.

    Args:
        df (pandas.DataFrame): Input data containing the target column.
        category_lst (list[str]): Categorical columns to encode.
        response (str): Target column name.

    Returns:
        pandas.DataFrame: The input DataFrame with new encoded columns added.
    """

    for col in category_lst:
        col_encoded_name = f"{col}_{response}"
        col_groups = df.groupby(col)[response].mean()

        df[col_encoded_name] = df[col].map(col_groups)

    return df


def perform_feature_engineering(df: pd.DataFrame,
                                category_lst: list[str] = CATEGORICAL_COLS,
                                feature_lst: list[str] = FEATURE_COLS,
                                response: str = "Churn") -> Tuple[pd.DataFrame,
                                                                  pd.DataFrame,
                                                                  pd.Series,
                                                                  pd.Series]:
    """
    Create the model matrix and split into train and test sets.

    This applies `encoder_helper`, selects `FEATURE_COLS`, and returns
    a 70/30 train/test split.

    Args:
        df (pandas.DataFrame): Input data.
        category_lst (list[str]): Categorical columns to encode.
        feature_lst (list[str]): Feature columns.
        response (str): Target column name.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    df = encoder_helper(df, category_lst, response=response)

    X = df[feature_lst].copy()
    y = df[response].copy()

    return train_test_split(X, y, test_size=0.3, random_state=42)


def _save_classification_report_image(
    title: str,
    y_train: pd.Series,
    y_train_pred: pd.Series,
    y_test: pd.Series,
    y_test_pred: pd.Series,
    out_path: str,
) -> None:
    """
    Render train/test classification reports for a model and save as PNG.

    Args:
        title (str): Model name for the figure title.
        y_train (pandas.Series): True labels for the training set.
        y_train_pred (pandas.Series): Predicted labels for the training set.
        y_test (pandas.Series): True labels for the test set.
        y_test_pred (pandas.Series): Predicted labels for the test set.
        out_path (str): Output path for the image file.
    """
    fig = plt.figure(figsize=(6, 6))
    plt.text(0.01, 0.95, f"{title} — Train", {
             "fontsize": 10, "fontweight": "bold"})
    plt.text(
        0.01, 0.55, classification_report(
            y_train, y_train_pred), {
            "fontsize": 9})
    plt.text(0.01, 0.45, f"{title} — Test", {
             "fontsize": 10, "fontweight": "bold"})
    plt.text(
        0.01, 0.05, classification_report(
            y_test, y_test_pred), {
            "fontsize": 9})
    plt.axis("off")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def classification_report_image(y_train: pd.Series,
                                y_test: pd.Series,
                                y_train_preds_lr: pd.Series,
                                y_train_preds_rf: pd.Series,
                                y_test_preds_lr: pd.Series,
                                y_test_preds_rf: pd.Series) -> None:
    """
    Save classification report images for Logistic Regression and Random Forest.

    Args:
        y_train (pandas.Series): True labels for training data.
        y_test (pandas.Series): True labels for test data.
        y_train_preds_lr (pandas.Series): Logistic Regression train predictions.
        y_train_preds_rf (pandas.Series): Random Forest train predictions.
        y_test_preds_lr (pandas.Series): Logistic Regression test predictions.
        y_test_preds_rf (pandas.Series): Random Forest test predictions.
    """

    _save_classification_report_image(
        title="Logistic Regression",
        y_train=y_train,
        y_train_pred=y_train_preds_lr,
        y_test=y_test,
        y_test_pred=y_test_preds_lr,
        out_path=os.path.join(
            RESULTS_IMAGE_PATH,
            "classification_report_logistic.png"),
    )

    _save_classification_report_image(
        title="Random Forest",
        y_train=y_train,
        y_train_pred=y_train_preds_rf,
        y_test=y_test,
        y_test_pred=y_test_preds_rf,
        out_path=os.path.join(
            RESULTS_IMAGE_PATH,
            "classification_report_random_forest.png"),
    )


def feature_importance_plot(
        model: GridSearchCV,
        X_data: pd.DataFrame,
        output_pth: str) -> None:
    """
    Plot feature importances for the best estimator in the fitted GridSearchCV.

    Args:
        model (sklearn.model_selection.GridSearchCV): Fitted grid search object.
        X_data (pandas.DataFrame): Feature matrix used for training.
        output_pth (str): Output path for the PNG image.
    """
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    fig = plt.figure(figsize=(32, 12))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel("Importance")

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=45)

    # Save the image to a file
    fig.savefig(output_pth)
    plt.close(fig)


def train_models(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series) -> None:
    """Train Logistic Regression and Random Forest, evaluate, and save artifacts.

    Saves:
        - models/logistic_model.pkl
        - models/rfc_model.pkl
        - images/results/roc_curve.png
        - images/results/classification_report_*.png
        - images/results/feature_importances.png

    Args:
        X_train (pandas.DataFrame): Training features.
        X_test (pandas.DataFrame): Test features.
        y_train (pandas.Series): Training labels.
        y_test (pandas.Series): Test labels.
    """
    #  Initialize Random Forest and Logistic Regression models
    rfc = RandomForestClassifier(random_state=42, n_jobs=-1)
    lrc = Pipeline(
        steps=[
            ("scaler",
             StandardScaler()),
            ("clf",
             LogisticRegression(
                 solver="lbfgs",
                 max_iter=5000,
                 tol=1e-3,
                 C=0.5))])

    # Define hyperparameters for Grid Search
    param_grid = {
        "n_estimators": [300, 600],
        "max_depth": [6, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "criterion": ["gini", "entropy"],
    }

    # Grid Search for Random Forest model
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # Train Logistic Regression model
    lrc.fit(X_train, y_train)

    # Generate predictions for both train and test data
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Save classification reports
    classification_report_image(
        y_train=y_train,
        y_test=y_test,
        y_train_preds_lr=y_train_preds_lr,
        y_train_preds_rf=y_train_preds_rf,
        y_test_preds_lr=y_test_preds_lr,
        y_test_preds_rf=y_test_preds_rf,
    )

    # Plot and save ROC curves
    fig = plt.figure(figsize=(20, 10))
    ax = plt.gca()
    RocCurveDisplay.from_estimator(
        lrc,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8,
        name="LogisticRegressionClassifier")
    RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8, name="RandomForestClassifier")
    plt.title("ROC Curves — Logistic Regression vs Random Forest")
    fig.savefig(os.path.join(RESULTS_IMAGE_PATH, "roc_curve.png"))
    plt.close(fig)

    # Save feature importance plot
    feature_importance_plot(
        model=cv_rfc,
        X_data=X_train,
        output_pth=os.path.join(RESULTS_IMAGE_PATH, "feature_importances.png"),
    )

    # Save best models to disk
    joblib.dump(
        cv_rfc.best_estimator_,
        os.path.join(
            MODELS_PATH,
            "rfc_model.pkl"))
    joblib.dump(lrc, os.path.join(MODELS_PATH, "logistic_model.pkl"))


def _ensure_paths() -> None:
    """Create artifact folders if they do not exist.

    This ensures `images/eda`, `images/results`, and `models` exist.
    """
    for p in (EDA_IMAGE_PATH, RESULTS_IMAGE_PATH, MODELS_PATH):
        Path(p).mkdir(parents=True, exist_ok=True)


def _run_pipeline() -> None:
    """
    Run the pipeline on `DATASET_PATH`
    """

    _ensure_paths()
    df = import_data(DATASET_PATH)

    if df is None:
        print("Could not load dataframe")
        return

    perform_eda(df)

    X_train, X_test, y_train, y_test = perform_feature_engineering(df)
    train_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    _run_pipeline()
