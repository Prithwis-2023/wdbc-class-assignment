# WDBC Classification

## Project Purpose
This project aims to classify breast cancer tumors as either **malignant** or **benign** using various machine learning models. It evaluates the performance of different classifiers on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

## Dataset Used
The project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which is available through `sklearn.datasets.load_breast_cancer()`. 
- **Features**: 30 numeric attributes (mean, standard error, and "worst" or largest of various cell nuclei characteristics).
- **Target**: Malignant (0) or Benign (1).
- **Samples**: 569 instances.

## How to Install Requirements
To set up the environment, ensure you have Python installed and then run:
```bash
pip install -r requirements.txt
```

## How to Run the Code
You can run the classification script using:
```bash
python wdbc_classification.py
```
*Note: If you are using a virtual environment, ensure it is activated or use the direct path to the python binary.*

## Output Files
- **Console Output**: Prints dataset description, performance metrics (Accuracy, Recall, Precision, F1-Score) for each model, and identifies the best model.
- **Plots**:
  - A Confusion Matrix for the best model (displayed via `plt.show()`).
  - A Scatter Plot of "Mean Radius" vs "Mean Texture" (displayed via `plt.show()`).

## Best Classifier Result
The best performing model identified during evaluation is:
- **Model**: Random Forest
- **Accuracy**: 96.49%

### Comparison:
| Model | Accuracy |
| :--- | :--- |
| **Random Forest** | **96.49%** |
| K-Nearest Neighbors | 95.61% |
| SVM | 94.74% |
| Decision Tree | 94.74% |
