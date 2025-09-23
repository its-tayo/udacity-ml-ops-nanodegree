import os
import sys
import json
import pickle
import timeit
import logging
import subprocess

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# Function to get model predictions
def model_predictions(df, prod_deployment_path):
    # read the deployed model and a test dataset, calculate predictions
    model_file = os.path.join(prod_deployment_path, "trainedmodel.pkl")
    model = pickle.load(open(model_file, "rb"))

    X = df.drop(columns=["exited", "corporation"])
    preds = model.predict(X)
    return preds


# Function to get summary statistics
def dataframe_summary(output_folder_path):
    # calculate summary statistics here
    csv_file = os.path.join(output_folder_path, "finaldata.csv")
    df = pd.read_csv(csv_file)

    num_df = df.select_dtypes(include=[np.number])

    means = num_df.mean().tolist()
    medians = num_df.median().tolist()
    stds = num_df.std().tolist()

    return [means, medians, stds]


# Function to check missing data
def missing_data(output_folder_path):
    csv_file = os.path.join(output_folder_path, "finaldata.csv")
    df = pd.read_csv(csv_file)

    na_pct = (df.isna().mean() * 100.0).tolist()
    return na_pct


# Function to get timings
def execution_time():
    def _run(script: str) -> float:
        subprocess.run([sys.executable, script], check=True, capture_output=True)

    ingestion_sec = timeit.timeit(lambda: _run("ingestion.py"), number=1)
    training_sec = timeit.timeit(lambda: _run("training.py"), number=1)

    return [ingestion_sec, training_sec]


# Function to check dependencies
def outdated_packages_list():
    out = subprocess.check_output(
        [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"], text=True
    )

    data = json.loads(out)

    return [
        {
            "name": d["name"],
            "installed_version": d["version"],
            "latest_version": d["latest_version"],
        }
        for d in data
    ]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stdout,
        force=True
    )

    # Load config.json and get environment variables
    with open("config.json", "r") as f:
        config = json.load(f)

    test_data_path = os.path.join(config["test_data_path"])
    output_folder_path = os.path.join(config["output_folder_path"])
    prod_deployment_path = os.path.join(config["prod_deployment_path"])

    TEST_DATA = os.path.join(config["test_data_path"], "testdata.csv")
    df = pd.read_csv(TEST_DATA)

    preds = model_predictions(df, prod_deployment_path)
    logger.info("Sample predictions (first 5): %s", preds[:5])

    stats = dataframe_summary(output_folder_path)
    logger.info("Summary stats (means, medians, stds) lengths: %s", [len(x) for x in stats])

    na_percents = missing_data(output_folder_path)
    logger.info("Missing data %% per column: %s", na_percents)

    times = execution_time()
    logger.info("Execution times [ingestion, training] (s): %s", times)

    deps = outdated_packages_list()
    logger.info("Outdated packages check (first 10): %s", deps[:10])