# Diabetes Prediction Using Machine Learning 🩺🔍

## Project Overview
This project focuses on predicting diabetes using machine learning models, applied to the Pima Indians Diabetes dataset. The goal is to create a predictive model based on health features such as glucose levels, BMI, blood pressure, and more, to identify individuals at risk of developing diabetes.

**Key Objectives:**
- Perform **Exploratory Data Analysis (EDA)** to uncover patterns in the data.
- Implement various machine learning models, such as **Logistic Regression** and **Random Forest**, to predict diabetes.
- Handle class imbalance using **SMOTE** (Synthetic Minority Over-sampling Technique) to improve the model's ability to detect diabetes-positive cases.
- Evaluate model performance using metrics like **accuracy**, **precision**, **recall**, **F1-score**, and the **ROC curve**.

## Dataset
The dataset used for this project is the **Pima Indians Diabetes dataset** from the UCI Machine Learning Repository. It contains medical data with the following features:
- **Pregnancies**: Number of times the patient was pregnant.
- **Glucose**: Plasma glucose concentration (mg/dL).
- **BloodPressure**: Diastolic blood pressure (mm Hg).
- **SkinThickness**: Triceps skinfold thickness (mm).
- **Insulin**: 2-Hour serum insulin (mu U/ml).
- **BMI**: Body mass index (weight in kg/(height in m)^2).
- **DiabetesPedigreeFunction**: A function which scores the likelihood of diabetes based on family history.
- **Age**: Age of the patient.
- **Outcome**: Binary variable (0 = no diabetes, 1 = diabetes).

## Main Dependencies:
- pandas: Data manipulation and analysis.
- numpy: Numerical computing.
- matplotlib and seaborn: Data visualization.
- scikit-learn: Machine learning algorithms and evaluation metrics.
- imblearn: SMOTE for handling class imbalance.

## Project Workflow
1. **Data Preprocessing**
Missing Value Handling: Features with missing values (e.g., insulin) were imputed using median values.
Outlier Detection: Box plots were used to identify and handle outliers in features like SkinThickness and Insulin.
Scaling: Numerical features were standardized using StandardScaler.
2. **Exploratory Data Analysis (EDA)**
Visualizations: Relationships between variables such as Glucose, BMI, and Age were explored using scatter plots, histograms, and box plots.
Key Patterns: Higher glucose levels and BMI were found to be strongly associated with diabetes-positive cases.
3. **Feature Engineering**
Created BMI categories and Age groups to enhance interpretability.
Applied one-hot encoding for categorical features if necessary.
4. **Model Building**
Logistic Regression: Used as a baseline model for binary classification.
Random Forest Classifier: A more complex model used to capture non-linear relationships between features.
SMOTE: Applied to handle class imbalance, ensuring better sensitivity (recall) for diabetes-positive cases.
5. **Model Evaluation**
Confusion Matrix: Used to evaluate the performance of the model by visualizing true positives, false positives, false negatives, and true negatives.

ROC Curve and AUC: Showed the trade-off between true positive rate and false positive rate. 
## The AUC score of the final model was 0.82.
Model Metrics:
- Accuracy: 77%
- Recall (Class 1): 74%
- F1-Score (Class 1): 0.70

**Results and Insights**
Glucose levels emerged as the most important feature in predicting diabetes, followed by BMI and Age.
The final Random Forest model with SMOTE demonstrated a balanced performance with:
- Accuracy: 77%
- Precision: 66%
- Recall: 74%
- F1-Score: 70%

## Conclusion
This project highlights the potential of machine learning in healthcare for predicting diabetes risk based on health indicators. By handling class imbalance with SMOTE, the model's ability to detect true diabetes cases was significantly improved. The insights gained from this analysis align with medical knowledge, reaffirming the importance of glucose levels, BMI, and age in diabetes prediction.
