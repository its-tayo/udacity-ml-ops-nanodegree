import os
import json
import requests


def fetch_api_returns(output_model_path, url):
    # Call each API endpoint and store the responses
    r1 = requests.post(f"{url}prediction", params={"dataset_path": "testdata/testdata.csv"})
    r2 = requests.get(f"{url}scoring")
    r3 = requests.get(f"{url}summarystats")
    r4 = requests.get(f"{url}diagnostics")

    return "\n".join(
        [
            f"prediction: {r1.text}",
            f"scoring: {r2.text}",
            f"summarystats: {r3.text}",
            f"diagnostics: {r4.text}",
        ]
    )


if __name__ == "__main__":
    # Load config.json and get path variables
    with open("config.json", "r") as f:
        config = json.load(f)

    output_model_path = os.path.join(config["output_model_path"])
    out_path = os.path.join(output_model_path, "apireturns.txt")

    combined = fetch_api_returns(output_model_path, url="http://127.0.0.1:8000/")

    with open(out_path, "w") as f:
        f.write(combined)
