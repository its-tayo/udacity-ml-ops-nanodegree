import os
import sys
import json
import glob
import logging
import subprocess

import pandas as pd

import ingestion
import diagnostics
import training
import scoring
import deployment
import reporting

logger = logging.getLogger(__name__)


def _read_ingested_list(path: str):
    if not os.path.exists(path):
        return []
    
    with open(path, "r") as f:
        items = f.read().splitlines()

    return [os.path.basename(x) for x in items if x.strip()]


def _list_csv_basenames(folder: str):
    return sorted([os.path.basename(p) for p in glob.glob(os.path.join(folder, "*.csv"))])



def main():
    # Load config.json and get environment variables
    with open("config.json", "r") as f:
        config = json.load(f)

    input_folder_path = config["input_folder_path"]
    output_folder_path = config["output_folder_path"]
    prod_deployment_path = config["prod_deployment_path"]
    output_model_path = config["output_model_path"]

    
    #Check and read new data
    ingested_path = os.path.join(prod_deployment_path, "ingestedfiles.txt")
    previously_ingested = set(_read_ingested_list(ingested_path))
    current_sources = set(_list_csv_basenames(output_folder_path))

    new_files = sorted(list(current_sources - previously_ingested))

    # Deciding whether to proceed, part 1
    # if you found new data, you should proceed. otherwise, do end the process here
    if new_files:
        logger.info("New data detected: %s", new_files)
        # ingest (merge + write ingestedfiles.txt in output_folder)
        ingestion.merge_multiple_dataframe(input_folder_path, output_folder_path)
    else:
        logger.info("No new data. Exiting.")
        return
    

    # Checking for model drift
    # read deployed model's last score
    last_score_file = os.path.join(prod_deployment_path, "latestscore.txt")

    with open(last_score_file, "r") as f:
        prev_score = float(f.read().strip())

    FINAL_DATA = os.path.join(output_folder_path, "finaldata.csv")
    df = pd.read_csv(FINAL_DATA)

    new_score = scoring.score_model(FINAL_DATA, os.path.join(prod_deployment_path))

    logger.info("Previous deployed F1: %.6f; New F1 on latest data: %.6f", prev_score, new_score)
    drift = new_score < prev_score

    # Deciding whether to proceed, part 2
    # if you found model drift, you should proceed. otherwise, do end the process here
    if not drift:
        logger.info("No model drift detected. Exiting.")
        return

    training.train_model(os.path.join(output_folder_path), os.path.join(output_model_path))
    scoring.score_model(FINAL_DATA, os.path.join(output_model_path))
    deployment.deploy_model(os.path.join(output_folder_path), os.path.join(output_model_path), os.path.join(prod_deployment_path))
    reporting.generate_reports(df, os.path.join(output_model_path), os.path.join(prod_deployment_path))  

    subprocess.run([sys.executable, "apicalls.py"], check=False)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stdout,
        force=True
    )

    main()