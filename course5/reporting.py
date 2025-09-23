import os
import sys
import json
import logging

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


from diagnostics import model_predictions

logger = logging.getLogger(__name__)

# Function for reporting
def generate_reports(df, output_model_path, prod_deployment_path):
    y_pred = model_predictions(df, prod_deployment_path)
    y_true = df["exited"].to_numpy()

    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()

    out_path = os.path.join(output_model_path, "confusionmatrix.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    logger.info("Confusion matrix saved to %s", out_path)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stdout,
        force=True,
    )

    # Load config.json and get path variables
    with open("config.json", "r") as f:
        config = json.load(f)

    TEST_DATA = os.path.join(config["test_data_path"], "testdata.csv")
    df = pd.read_csv(TEST_DATA)

    output_model_path = os.path.join(config["output_model_path"])
    prod_deployment_path = os.path.join(config["prod_deployment_path"])

    generate_reports(df, output_model_path, prod_deployment_path)
