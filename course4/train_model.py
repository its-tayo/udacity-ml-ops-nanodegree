"""
Script to train machine learning model.

Author: Tayo Akindolie
"""
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.data import process_data
from utils.model import (
    train_model,
    inference,
    compute_model_metrics,
    compare_slice_performance,
)

if __name__ == "__main__":
    # Add code to load in the data.
    data = pd.read_csv("data/census.csv")

    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Proces the test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    # Train and save a model.
    model = train_model(X_train, y_train)

    # Save the model and label binarizer to disk.
    model_filename = "model/model.pkl"
    encoder_filename = "model/encoder.pkl"
    lb_filename = "model/label_binarizer.pkl"

    with open(model_filename, "wb") as model_file:
        pickle.dump(model, model_file)

    with open(encoder_filename, "wb") as encoder_file:
        pickle.dump(encoder, encoder_file)

    with open(lb_filename, "wb") as lb_file:
        pickle.dump(lb, lb_file)

    x_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    y_preds = inference(model, x_test)

    precision, recall, fbeta = compute_model_metrics(y_test, y_preds)

    print(
        f"Performance for test set - Precision: {precision}, "
        f"Recall: {recall}, "
        f"Fbeta: {fbeta}"
    )

    with open("model/slice_output.txt", "w") as f:
        for (
            feature,
            value,
            precision,
            recall,
            fbeta,
        ) in compare_slice_performance(
            test, cat_features, "salary", encoder, lb, model
        ):
            line_header = f"Feature: {feature}, Value: {value}"
            line_p = f"Precision: {precision:.2f}"
            line_r = f"Recall: {recall:.2f}"
            line_f = f"F1: {fbeta:.2f}"
            sep = "-" * 20

            f.write(line_header + "\n")
            f.write(line_p + "\n")
            f.write(line_r + "\n")
            f.write(line_f + "\n")
            f.write(sep + "\n")
