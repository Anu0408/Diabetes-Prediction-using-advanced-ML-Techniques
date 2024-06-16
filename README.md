# Diabetes Prediction using advanced Machine Learning Techniques

This project aims to predict diabetes using the PIMA diabetes dataset by applying machine learning techniques to develop a predictive model.

## Requirements
libraries:
- `pandas`
- `matplotlib`
- `numpy`
- `scikit-learn`

Install these libraries using:
```
pip install pandas matplotlib numpy scikit-learn
```
## Dataset
The dataset (`pima-data.csv`) includes:
- `num_preg`: Number of pregnancies
- `glucose_conc`: Glucose concentration
- `diastolic_bp`: Diastolic blood pressure
- `thickness`: Skin thickness
- `insulin`: Insulin level
- `bmi`: Body mass index
- `diab_pred`: Diabetes pedigree function
- `age`: Age
- `skin`: Skin measurement
- `diabetes`:
 Diabetes status (True/False)

## Steps to Run the Project
1. **Import Libraries**:
    ```
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    %matplotlib inline
    ```
2. **Load Data**:
    ```
    data = pd.read_csv("./data/pima-data.csv")
    ```
3. **Explore Data**:
    ```
    print(data.shape)
    print(data.head(5))
    ```
4. **Check for Null Values**:
    ```python
    print(data.isnull().values.any())
    ```

## Data Visualization
Visualize the dataset to understand feature distributions and relationships.
``` import seaborn as sns ```
# Pairplot
``` sns.pairplot(data, hue='diabetes')```
# Correlation matrix
```
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()
```
## Model Building
Build and evaluate machine learning models such as Logistic Regression.
```
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

X = data.drop('diabetes', axis=1)
y = data['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

## Evaluation Metrics
Evaluate the model with accuracy, precision, recall, F1-score, and ROC-AUC score.
```
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))
```

After running the evaluation code, the results were:
- **Accuracy**: 0.79
- **Precision**: 0.74
- **Recall**: 0.68
- **F1 Score**: 0.71
- **ROC AUC Score**: 0.85

## Conclusion
- **Findings**: Logistic Regression provided balanced performance with an accuracy of 79% and an ROC AUC score of 0.85.
- **Future Work**: Explore hyperparameter tuning, feature engineering, and advanced techniques like ensemble learning for improvement.

## Acknowledgements
The dataset is from the National Institute of Diabetes and Digestive and Kidney Diseases.

## License
This project is licensed under the GNU General Public License v3.0. See the LICENSE file for more details.
