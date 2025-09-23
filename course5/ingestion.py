import os
import sys
import json
import glob
import logging

import pandas as pd


logger = logging.getLogger(__name__)


# Function for data ingestion
def merge_multiple_dataframe(input_folder_path, output_folder_path):
    # Collect all CSV file paths
    csv_files = glob.glob(os.path.join(input_folder_path, "*.csv"))

    try:
        assert len(csv_files) > 0
    except AssertionError as e:
        logger.error("Input folder %s does not contain .csv files", input_folder_path)
        raise e

    # Read and concatenate them into one DataFrame
    df_list = [pd.read_csv(file) for file in csv_files]
    merged_df = pd.concat(df_list, ignore_index=True)

    # Drop duplicate rows
    merged_df = merged_df.drop_duplicates()

    # Save the merged, deduped file
    os.makedirs(output_folder_path, exist_ok=True)  
    output_file = os.path.join(output_folder_path, "finaldata.csv")
    merged_df.to_csv(output_file, index=False)

    logger.info("Merged and deduped CSV saved to %s", output_file)

    # Save ingested file names to text file
    ingested_files = [os.path.basename(file) for file in csv_files]
    ingested_file_path = os.path.join(output_folder_path, "ingestedfiles.txt")

    with open(ingested_file_path, "w") as f:
        f.write("\n".join(ingested_files))

    logger.info("Record of ingested files saved to %s", ingested_file_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s:%(message)s",
        stream=sys.stdout,
        force=True
    )
    
    # Load config.json and get input and output paths
    with open("config.json", "r") as f:
        config = json.load(f)

    input_folder_path = config["input_folder_path"]
    output_folder_path = config["output_folder_path"]

    merge_multiple_dataframe(input_folder_path, output_folder_path)
