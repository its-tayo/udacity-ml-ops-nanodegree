# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This project uses a Random Forest Classifier from the scikit-learn library. The model predicts whether an individual’s annual income is above or below $50,000 using census data. It was built as part of an end-to-end machine learning pipeline to practice training, evaluating, and deploying models in a reproducible way.

## Intended Use

The model is intended for demonstration and learning purposes. It can be used to explore the process of training, testing, and deploying a machine learning system. While it does produce predictions from census features such as age, education, occupation, and hours worked per week, it is not designed for making real financial or employment decisions.

## Training Data

The model was trained on the UCI Census Income dataset, which records demographic and employment information about individuals from the mid-1990s. Eighty percent of the data was used for training.

## Evaluation Data

The remaining twenty percent of the same dataset was held back to measure model performance and verify generalisation.

## Metrics

Performance was assessed with three standard classification metrics:

- **Precision**: 0.7417943107221007
- **Recall**: 0.6340399002493765
- **F1 Score**: 0.6836974789915966

## Ethical Considerations

The census data reflects the social and economic conditions of its time and may contain biases linked to race, gender, or occupation. Even though the model is only trained on observed features, those underlying patterns can reproduce inequalities in the predictions. For that reason, care should be taken if extending this model beyond its original educational setting.

## Caveats and Recommendations

Because the dataset is dated, the predictions may not represent present-day trends. The preprocessing steps are relatively simple, so there is room for improvement with more careful feature engineering. Anyone wishing to adapt the model for serious use should retrain it on up-to-date data and run careful fairness checks across different demographic groups.
