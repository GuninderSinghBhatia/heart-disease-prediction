# Heart Disease Prediction

A beginner machine learning project that predicts whether a patient has heart disease using **Logistic Regression**, built in a Kaggle notebook.

## Dataset

- **File:** `heart_disease_data.csv` (UCI Heart Disease dataset)
- **Size:** 303 patients, 13 features + 1 target
- **Target:** `target` — `1` = heart disease, `0` = no heart disease
- **Missing values:** none

| Feature | Description |
|---|---|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = true) |
| restecg | Resting ECG results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (1 = yes) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Number of major vessels (0–3) |
| thal | Thalassemia type |

## Workflow

1. Load the data with pandas
2. Explore it (`info()`, missing-value check)
3. Split into features `x` (all columns except `target`) and label `y` (`target`)
4. Train/validation split — 75% / 25%, `random_state=42`
5. Train `LogisticRegression(max_iter=1000)`
6. Evaluate with accuracy on both sets

## Results

| Set | Accuracy |
|---|---|
| Training | ~85.5% |
| Validation | ~89.5% |

Similar train and validation scores suggest the model is not overfitting.

## Tech Stack

- Python
- pandas, NumPy
- scikit-learn

## How to Run

1. Open `heart_disease_prediction.ipynb` on Kaggle or in Jupyter.
2. Update the CSV path in the "Loading Data" cell to where your dataset is stored.
3. Run all cells.

## Next Steps

- Scale features with `StandardScaler`
- Add precision, recall and a confusion matrix
- Compare with other models (Random Forest, KNN)
