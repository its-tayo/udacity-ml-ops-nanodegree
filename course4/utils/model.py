from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from .data import process_data

# Optional: implement hyperparameter tuning.


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    rfc = RandomForestClassifier(n_estimators=300, random_state=42)
    rfc.fit(X_train, y_train)
    return rfc


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision,
    recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def compare_slice_performance(
    test_df, categorical_features, label, encoder, lb, model
):
    for feature in categorical_features:
        for value in sorted(test_df[feature].dropna().unique()):
            mask = (test_df[feature] == value).to_numpy()

            if not mask.any():
                continue

            X_slice = test_df.loc[mask].copy()

            X_proc, y_proc, _, _ = process_data(
                X_slice,
                categorical_features=categorical_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb,
            )

            preds = model.predict(X_proc)
            precision, recall, fbeta = compute_model_metrics(
                y_proc, preds
            )

            yield feature, value, float(precision), float(
                recall
            ), float(fbeta)
