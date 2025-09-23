import os
import sys
import json
import shutil
import logging


logger = logging.getLogger(__name__)


# Copy file to dst_dir
def _copy_file(src, dst_dir):
    if not os.path.exists(src):
        raise FileNotFoundError(f"Expected file not found: {src}")

    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))

    shutil.copy2(src, dst)
    logger.info("Copied %s -> %s", src, dst)
    return dst


# Function for deployment
def deploy_model(
    output_folder_path, output_model_path, prod_deployment_path
):
    sources = {
        "trainedmodel.pkl": os.path.join(output_model_path, "trainedmodel.pkl"),
        "latestscore.txt": os.path.join(output_model_path, "latestscore.txt"),
        "ingestedfiles.txt": os.path.join(output_folder_path, "ingestedfiles.txt"),
    }

    for _, src in sources.items():
        _copy_file(src, prod_deployment_path)

    logger.info("Deployment complete. Files available in %s", prod_deployment_path)

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
    prod_deployment_path = os.path.join(config["prod_deployment_path"])

    deploy_model(output_folder_path, output_model_path, prod_deployment_path)
