#!/usr/bin/env python
"""
Performs basic cleaning on the data and saves the results in W&B
"""
import logging
import argparse

import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info(f"Fetching artifact {args.input_artifact}")

    local_artifact = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(local_artifact)

    logger.info(f"Setting min and max price to {args.min_price} and {args.max_price} to remove outliers")
    min_price = args.min_price
    max_price = args.max_price

    idx = df["price"].between(min_price, max_price)
    df = df[idx].copy()

    logger.info("convert last_review to datetime")
    df["last_review"] = pd.to_datetime(df["last_review"])

    filename = "clean_sample.csv"
    df.to_csv(filename, index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)
    run.log_artifact(artifact)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This step cleans the data")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of artifact downloaded from weights & biases",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of artifact created and uploaded to weights & biases",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of artifact to be uploaded to weights & biases",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of artifact to be uploaded to weights & biases",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="The minimum price to consider when removing outliers",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="The maximum price to consider when removing outliers",
        required=True
    )


    args = parser.parse_args()

    go(args)
