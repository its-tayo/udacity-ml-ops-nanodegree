import os
import sys
import json
import pickle
import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


# Function for training the model
def train_model(output_folder_path, output_model_path):
    # Load data
    csv_file = os.path.join(output_folder_path, "finaldata.csv")

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"finaldata.csv not found at: {csv_file}")

    df = pd.read_csv(csv_file)
    logger.info("Loaded dataset with shape %s", df.shape)

    # Split features/target
    y = df["exited"]
    X = df.drop(columns=["exited", "corporation"])

    # Logistic regression for training
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class="ovr",
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )

    # Fit the logistic regression to your data
    model.fit(X, y)
    logger.info("Model trained")

    # Save trained model
    os.makedirs(output_model_path, exist_ok=True)
    model_file = os.path.join(output_model_path, "trainedmodel.pkl")
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    logger.info("Saved trained model to %s", model_file)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stdout,
        force=True
    )

    # Load config.json and get path variables
    with open("config.json", "r") as f:
        config = json.load(f)

    output_folder_path = os.path.join(config["output_folder_path"])
    output_model_path = os.path.join(config["output_model_path"])

    train_model(output_folder_path, output_model_path)
