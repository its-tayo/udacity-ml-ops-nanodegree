"""
Pytest suite for the churn pipeline in churn_library.py.

Purpose:
    Validate the full workflow:
        - data import
        - EDA artifact generation
        - categorical target-rate encoding
        - feature engineering & splitting
        - model training and artifact persistence

Author: Tayo Akindolie
Date: 2025-07-09
"""


# import libraries
import logging
import os
from pathlib import Path
from typing import Iterable, List, Tuple, TypeAlias

import joblib
import numpy as np
import pandas as pd
import pytest

import churn_library as cl
from constants import (
    DATASET_PATH,
    EDA_IMAGE_PATH,
    CATEGORICAL_COLS,
    FEATURE_COLS,
    RESULTS_IMAGE_PATH,
    MODELS_PATH,
    LOG_FILE_PATH,
)

# Ensure the logs directory exists before configuring logging.
Path(LOG_FILE_PATH).parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(asctime)s - %(levelname)s - %(message)s",
)

TrainTestSplits: TypeAlias = tuple[pd.DataFrame,
                                   pd.DataFrame, pd.Series, pd.Series]


def _ensure_dirs(paths: Iterable[str]) -> None:
    """
    Ensure each directory in `paths` exists.

    Args:
        paths: One or more directory paths to create if missing.

    Returns:
        None
    """
    for p in paths:
        try:
            Path(p).mkdir(parents=True, exist_ok=True)
        except Exception:
            logging.error("Failed to create directory: %s", p, exc_info=True)
            raise


def _list_files(dir_path: str) -> List[str]:
    """
    List files in a directory non-recursively.

    Args:
        dir_path: Directory to scan.

    Returns:
        List[str]: Filenames (not full paths) contained in `dir_path`.

    Raises:
        FileNotFoundError: If `dir_path` does not exist.
        NotADirectoryError: If `dir_path` is not a directory.
    """
    p = Path(dir_path)
    if not p.exists():
        logging.error("Directory not found: %s", dir_path)
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    if not p.is_dir():
        logging.error("Not a directory: %s", dir_path)
        raise NotADirectoryError(f"Not a directory: {dir_path}")
    return [f.name for f in p.iterdir() if f.is_file()]


def _assert_has_all(
        actual: Iterable[str],
        expected: Iterable[str],
        context: str) -> None:
    """
    Assert that every item in `expected` appears in `actual`.

    Args:
        actual: Observed items (e.g., filenames).
        expected: Required items.
        context: Label used in error messages.

    Returns:
        None

    Raises:
        AssertionError: If any expected item is missing.
    """
    actual_set = set(actual)
    missing = [x for x in expected if x not in actual_set]
    if missing:
        logging.error("%s: missing items: %s", context, missing)
    assert not missing, f"{context}: missing items: {missing}"


@pytest.fixture
def dataset_path() -> str:
    """
    Dataset path on disk.

    Args:
        None

    Returns:
        str: File path to the CSV dataset.
    """
    return DATASET_PATH


@pytest.fixture
def eda_expected_images() -> List[str]:
    """
    Expected EDA image filenames produced by `perform_eda`.

    Args:
        None

    Returns:
        List[str]: Required EDA artifact filenames.
    """
    return [
        "churn_dist.png",
        "age_dist.png",
        "marital_status_dist.png",
        "total_trans_ct_dist.png",
        "corr_heatmap.png",
    ]


@pytest.fixture
def results_expected_images() -> List[str]:
    """
    Expected evaluation image filenames produced by `train_models`.

    Args:
        None

    Returns:
        List[str]: Required results artifact filenames.
    """
    return [
        "roc_curve.png",
        "classification_report_logistic.png",
        "classification_report_random_forest.png",
        "feature_importances.png",
    ]


@pytest.fixture
def expected_models() -> List[str]:
    """
    Expected model filenames saved by `train_models`.

    Args:
        None

    Returns:
        List[str]: Model artifact filenames.
    """
    return ["rfc_model.pkl", "logistic_model.pkl"]


@pytest.fixture(scope="session")
def io_paths() -> Tuple[str, str, str]:
    """
    Artifact directories for EDA, results, and models.
    Ensures directories exist before running tests.

    Args:
        None

    Returns:
        Tuple[str, str, str]: (eda_dir, results_dir, models_dir)
    """
    try:
        _ensure_dirs([EDA_IMAGE_PATH, RESULTS_IMAGE_PATH, MODELS_PATH])
    except Exception:
        logging.error(
            "Failed to ensure artifact directories exist: %s, %s, %s",
            EDA_IMAGE_PATH, RESULTS_IMAGE_PATH, MODELS_PATH, exc_info=True
        )
        raise

    return (EDA_IMAGE_PATH, RESULTS_IMAGE_PATH, MODELS_PATH)


@pytest.fixture
def import_data_fn():
    """
    Reference to `import_data` to simplify injection/mocking.

    Args:
        None

    Returns:
        Callable: The `import_data` function from `churn_library`.
    """
    return cl.import_data


@pytest.fixture
def df_raw(import_data_fn, dataset_path: str) -> pd.DataFrame:
    """
    Load the raw dataset via the library function.

    Args:
        import_data_fn: Data import function.
        dataset_path: Path to the CSV dataset.

    Returns:
        pandas.DataFrame: DataFrame with the raw columns plus `Churn`.

    Raises:
        AssertionError: If the returned object is not a non-empty DataFrame.
    """
    try:
        df = import_data_fn(dataset_path)
    except Exception:
        logging.error(
            "import_data failed for dataset_path=%s",
            dataset_path,
            exc_info=True)
        raise
    try:
        assert isinstance(
            df, pd.DataFrame), "import_data must return a pandas DataFrame"
        assert df.shape[0] > 0 and df.shape[1] > 0, "Imported DataFrame has no rows/columns"
    except AssertionError:
        logging.error(
            "import_data returned invalid DataFrame: shape=%s",
            getattr(df, "shape", None),
            exc_info=True
        )
        raise
    return df


@pytest.fixture
def perform_eda_fn():
    """
    Reference to `perform_eda`.

    Args:
        None

    Returns:
        Callable: The `perform_eda` function.
    """
    return cl.perform_eda


@pytest.fixture
def encoder_helper_fn():
    """
    Reference to `encoder_helper`.

    Args:
        None

    Returns:
        Callable: The `encoder_helper` function.
    """
    return cl.encoder_helper


@pytest.fixture
def perform_feature_engineering_fn():
    """
    Reference to `perform_feature_engineering`.

    Args:
        None

    Returns:
        Callable: The `perform_feature_engineering` function.
    """
    return cl.perform_feature_engineering


@pytest.fixture
def train_models_fn():
    """
    Reference to `train_models`.

    Args:
        None

    Returns:
        Callable: The `train_models` function.
    """
    return cl.train_models


@pytest.fixture
def train_test_splits(
        perform_feature_engineering_fn,
        df_raw) -> TrainTestSplits:
    """
    Args:
        perform_feature_engineering_fn: Feature engineering function.
        df_raw: Fresh input DataFrame.

    Returns:
        train_test_splits: (X_train, X_test, y_train, y_test)
    """
    try:
        X_tr, X_te, y_tr, y_te = perform_feature_engineering_fn(
            df_raw.copy(),
            category_lst=CATEGORICAL_COLS,
            feature_lst=FEATURE_COLS,
            response="Churn",
        )
    except Exception:
        logging.error(
            "perform_feature_engineering failed with feature list and categories provided.",
            exc_info=True)
        raise
    return X_tr, X_te, y_tr, y_te


def test_import_data_shape(df_raw: pd.DataFrame) -> None:
    """
    Verify that the imported DataFrame is non-empty.

    Args:
        df_raw: Dataset returned by `import_data`.

    Returns:
        None
    """
    logging.info("test_import_data_shape: Data shape = %s", df_raw.shape)
    try:
        assert df_raw.shape[0] > 0
        assert df_raw.shape[1] > 0
    except AssertionError:
        logging.error(
            "test_import_data_shape failed: shape=%s",
            df_raw.shape,
            exc_info=True)
        raise


def test_perform_eda_creates_images(
    perform_eda_fn,
    df_raw: pd.DataFrame,
    eda_expected_images: List[str],
    io_paths: Tuple[str, str, str],
) -> None:
    """
    Run EDA and assert required image artifacts exist.

    Args:
        perform_eda_fn: EDA function from the library.
        df_raw: Input data.
        eda_expected_images: Expected EDA figure filenames.
        io_paths: Tuple of (eda_dir, results_dir, models_dir).

    Returns:
        None
    """
    eda_dir, _, _ = io_paths
    try:
        perform_eda_fn(df_raw)
    except Exception:
        logging.error("perform_eda failed.", exc_info=True)
        raise
    try:
        produced = _list_files(eda_dir)
        _assert_has_all(produced, eda_expected_images, "EDA images")
    except Exception:
        logging.error(
            "EDA artifacts check failed. eda_dir=%s", eda_dir, exc_info=True
        )
        raise
    logging.info("test_perform_eda_creates_images: produced=%s", produced)


def test_encoder_helper_adds_target_rate_columns(
    encoder_helper_fn,
    df_raw: pd.DataFrame,
) -> None:
    """
    Verify target-rate encoding creates `<col>_Churn` columns without NA values.

    Args:
        encoder_helper_fn: Encoding function from the library.
        df_raw: Input data containing `Churn`.

    Returns:
        None
    """
    try:
        df = encoder_helper_fn(
            df_raw.copy(),
            CATEGORICAL_COLS,
            response="Churn")
        for col in CATEGORICAL_COLS:
            col_name = f"{col}_Churn"
            assert col_name in df.columns, f"Missing encoded column: {col_name}"
            assert int(np.sum(df[col_name].isna())
                       ) == 0, f"Nulls found in {col_name}"
    except Exception:
        logging.error(
            "encoder_helper validation failed for columns: %s",
            CATEGORICAL_COLS,
            exc_info=True
        )
        raise
    logging.info("test_encoder_helper_adds_target_rate_columns: OK")


def test_perform_feature_engineering_shapes(
    train_test_splits
) -> None:
    """
    Check that feature engineering yields non-empty, aligned train/test splits.

    Args:
        train_test_splits: Tuple of train/test splits.

    Returns:
        None
    """
    X_train, X_test, y_train, y_test = train_test_splits
    try:
        assert X_train.shape[0] > 0 and X_test.shape[0] > 0
        assert y_train.shape[0] > 0 and y_test.shape[0] > 0
        assert X_train.shape[1] == len(FEATURE_COLS)
        assert X_test.shape[1] == X_train.shape[1]
    except AssertionError:
        logging.error(
            "Feature engineering shapes invalid. "
            "X_train=%s, X_test=%s, y_train=%s, y_test=%s, expected_features=%d",
            X_train.shape,
            X_test.shape,
            y_train.shape,
            y_test.shape,
            len(FEATURE_COLS),
            exc_info=True)
        raise

    logging.info(
        "test_perform_feature_engineering_shapes: X_train=%s, X_test=%s",
        X_train.shape,
        X_test.shape,
    )


def test_train_models_and_artifacts(
    train_models_fn,
    results_expected_images: List[str],
    expected_models: List[str],
    io_paths: Tuple[str, str, str],
    train_test_splits: TrainTestSplits,
) -> None:
    """
    Train models, then assert model binaries and evaluation images exist and can be loaded.

    Args:
        train_models_fn: Training function from the library.
        results_expected_images: Expected filenames for result images.
        expected_models: Expected filenames for model binaries.
        io_paths: Tuple of (eda_dir, results_dir, models_dir).
        train_test_splits: Tuple of train/test splits.

    Returns:
        None

    Raises:
        AssertionError: If expected artifacts are missing or a model cannot be loaded.
    """
    _, results_dir, models_dir = io_paths
    X_train, X_test, y_train, y_test = train_test_splits

    logging.info(
        "Starting training with shapes -> X_train=%s, X_test=%s, y_train=%s, y_test=%s",
        X_train.shape,
        X_test.shape,
        y_train.shape,
        y_test.shape,
    )
    logging.info(
        "Artifact directories -> results_dir=%s, models_dir=%s",
        results_dir,
        models_dir)

    try:
        train_models_fn(X_train, X_test, y_train, y_test)
    except Exception:
        logging.error("train_models failed.", exc_info=True)
        raise

    try:
        produced_results = _list_files(results_dir)
        _assert_has_all(
            produced_results,
            results_expected_images,
            "Result images")
    except Exception:
        logging.error(
            "Result images validation failed. results_dir=%s",
            results_dir,
            exc_info=True)
        raise

    try:
        produced_models = _list_files(models_dir)
        _assert_has_all(produced_models, expected_models, "Model files")
        for name in expected_models:
            path = os.path.join(models_dir, name)
            try:
                _ = joblib.load(path)
            except Exception:
                logging.error(
                    "Failed to load model artifact: %s",
                    path,
                    exc_info=True)
                raise
    except Exception:
        logging.error(
            "Model artifacts validation failed. models_dir=%s",
            models_dir,
            exc_info=True)
        raise

    logging.info(
        "test_train_models_and_artifacts: models=%s, images=%s",
        produced_models,
        produced_results,
    )


if __name__ == "__main__":
    import pytest as _pytest

    _ensure_dirs([Path(LOG_FILE_PATH).parent.as_posix()])
    try:
        raise SystemExit(_pytest.main(['-s', os.path.abspath(__file__)]))
    except Exception:
        logging.error("Pytest invocation failed.", exc_info=True)
        raise
