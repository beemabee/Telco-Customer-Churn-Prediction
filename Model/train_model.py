# -------------------------------- Library ----------------------------------------------------
import numpy as np 
import pandas as pd 
import os 
import joblib 
from catboost import CatBoostClassifier, Pool 

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, 
    roc_curve, auc, roc_auc_score,
    classification_report, confusion_matrix
)
# --------------------------------- Data Loading and Edit -------------------------------------
data_path = r'D:\DS_ML_Project\Telco-Customer-Churn-Prediction\Telco-Customer-Churn.csv' 
df = pd.read_csv(data_path)

# convert column 'TotalCharges' to numeric, filling NaN Values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['tenure'] * df['MonthlyCharges'], inplace=True)

# convert column 'SeniorCitizen' to object
df['SeniorCitizen'] = df['SeniorCitizen'].astype(object)

# replace 'No phone service' and 'No internet service' with 'No'
df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')
columns_to_replace = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                      'TechSupport', 'StreamingTV', 'StreamingMovies']
for column in columns_to_replace:
    df[column] = df[column].replace('No internet service', 'No')

# convert 'Churn' categorical variable to numeric
df['Churn'] = df['Churn'].replace({'No': 0, 'Yes': 1})

# --------------------------------- StratifiedShuffleSplit -------------------------------------

# create the split object
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=23)

train_index, test_index = next(strat_split.split(df, df['Churn']))

# create train-test sets
strat_train_set = df.loc[train_index]
strat_test_set = df.loc[test_index]

X_train = strat_train_set.drop('Churn', axis=1)
y_train = strat_train_set['Churn'].copy()

X_test = strat_test_set.drop('Churn', axis=1)
y_test = strat_test_set['Churn'].copy()

# Save the preprocessed DataFrame as a parquet file
df.to_parquet(r'D:\DS_ML_Project\Telco-Customer-Churn-Prediction\Model\data\churn_data_regulated.parquet')

# Save the datasets
joblib.dump(X_train, r'D:\DS_ML_Project\Telco-Customer-Churn-Prediction\Model\data\X_train.pkl')
joblib.dump(y_train, r'D:\DS_ML_Project\Telco-Customer-Churn-Prediction\Model\data\y_train.pkl')
joblib.dump(X_test, r'D:\DS_ML_Project\Telco-Customer-Churn-Prediction\Model\data\X_test.pkl')
joblib.dump(y_test, r'D:\DS_ML_Project\Telco-Customer-Churn-Prediction\Model\data\y_test.pkl')

#  ---------------------------------------------- CATBOOST ---------------------------------------

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Initialize and fit CatBoostClassifier
cat_model = CatBoostClassifier(verbose=False, random_state=0, scale_pos_weight=3)
cat_model.fit(X_train, y_train, cat_features=categorical_columns, eval_set=(X_test, y_test))

# Predict on test set
y_pred = cat_model.predict(X_test)

# Calculate evaluation metrics
accuracy, recall, roc_auc, precision = [round(metric(y_test, y_pred), 4) for metric in [accuracy_score, recall_score, roc_auc_score, precision_score]]

# Create a DataFrame to store results
model_names = ['CatBoost_Model']
result = pd.DataFrame({'Accuracy': accuracy, 'Recall': recall, 'Roc_Auc': roc_auc, 'Precision': precision}, index=model_names)

# Print results
print(result)

# Save the model in the 'model' directory
model_dir = r"D:\DS_ML_Project\Telco-Customer-Churn-Prediction\Model"
model_path = os.path.join(model_dir, "catboost_model.cbm")
cat_model.save_model(model_path)