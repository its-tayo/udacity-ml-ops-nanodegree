import os
import json

import pandas as pd
from flask import Flask, request

import diagnostics


# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = "1652d576-484a-49fd-913a-6879acfa6ba4"

with open("config.json", "r") as f:
    config = json.load(f)

output_model_path = os.path.join(config["output_model_path"])
output_folder_path = os.path.join(config["output_folder_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])


# Prediction Endpoint
@app.route("/prediction", methods=["POST", "OPTIONS"])
def predict():
    dataset_path = request.args.get("dataset_path")
    df = pd.read_csv(dataset_path)

    preds = diagnostics.model_predictions(df, prod_deployment_path)
    return str(list(preds))


# Scoring Endpoint
@app.route("/scoring", methods=["GET", "OPTIONS"])
def score():
    score_file = os.path.join(output_model_path, "latestscore.txt")
    with open(score_file, "r") as f:
        latestscore = f.read()

    return latestscore


# Summary Statistics Endpoint
@app.route("/summarystats", methods=["GET", "OPTIONS"])
def stats():
    sumamry_dict = diagnostics.dataframe_summary(output_folder_path)
    return sumamry_dict


# Diagnostics Endpoint
@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def diagnose():
    timings = diagnostics.execution_time()
    missing = diagnostics.missing_data(output_folder_path)
    outdated = diagnostics.outdated_packages_list()

    return {"timings": timings, "missing_data": missing, "outdated_pckgs": outdated}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
