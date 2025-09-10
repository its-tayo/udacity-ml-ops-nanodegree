# Predict Customer Churn

This project builds a small, reproducible pipeline to predict credit-card customer churn.  
It loads a tabular dataset (`./data/bank_data.csv`), performs EDA, applies simple target-rate encoding for categorical columns, engineers a feature matrix, and trains two models:

- **Logistic Regression** (with `StandardScaler` in a pipeline)
- **Random Forest** (with `GridSearchCV` for hyperparameters)

The pipeline saves EDA plots, model-evaluation figures (classification reports and ROC curves), and trained models to disk.

**Objectives**

- Provide a clean, procedural Python library (`churn_library.py`) with type hints.
- Produce deterministic artifacts under `images/` and `models/`.
- Offer an automated test suite with `pytest` to verify functionality.

**Key artifacts produced**

- images/eda/
  - churn_dist.png
  - age_dist.png
  - marital_status_dist.png
  - total_trans_ct_dist.png
  - corr_heatmap.png
- images/results/
  - classification_report_logistic.png
  - classification_report_random_forest.png
  - roc_curve.png
  - feature_importances.png
- models/
  - logistic_model.pkl
  - rfc_model.pkl

**Notes**

- Dataset must be placed at `./data/bank_data.csv` (Credit Card Customers dataset).
- Paths and feature lists are defined in `constants.py`.

---

## Running Files

#### 1. Set up environment

```bash
python -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

#### 2. Run the pipeline

This executes EDA, feature engineering, model training, and artifact saving.

```bash
ipython churn_library.py
```

**Expected results**

- EDA figures saved under images/eda/.
- Evaluation figures (classification reports, ROC curves, feature importances) saved under images/results/.
- Trained models saved under models/.
- Logs written to logs/churn_library.log.

#### 3. Run the tests

```bash
pytest -q
# or:
ipython churn_script_logging_and_tests.py
```

**What tests cover**

- Data import shape checks.
- Presence of EDA images.
- Target-rate encoded columns (<col>\_Churn) with no missing values.
- Train/test split shapes and feature dimensions.
- Model training, saved model files, and evaluation images; model files are also load-tested.
