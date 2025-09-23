import os
import sys
import json
import pickle
import logging

import pandas as pd
from sklearn import metrics


logger = logging.getLogger(__name__)


# Function for model scoring
def score_model(TEST_DATA, output_model_path):
    # Load data
    df = pd.read_csv(TEST_DATA)
    logger.info("Loaded dataset with shape %s", df.shape)

    # Split features/target
    y = df["exited"]
    X = df.drop(columns=["exited", "corporation"])

    # Load model
    model_file = os.path.join(output_model_path, "trainedmodel.pkl")
    model = pickle.load(open(model_file, "rb"))

    # Predict
    y_pred = model.predict(X)

    # Score
    f1 = metrics.f1_score(y, y_pred)

    # Write score
    score_file = os.path.join(output_model_path, "latestscore.txt")
    with open(score_file, "w") as f:
        f.write(f"{f1}\n")

    logger.info("F1 score: %.6f", f1)
    return f1


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

    TEST_DATA = os.path.join(config["test_data_path"], "testdata.csv")
    output_model_path = os.path.join(config["output_model_path"])

    score_model(TEST_DATA, output_model_path)
