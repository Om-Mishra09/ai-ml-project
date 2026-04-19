# -*- coding: utf-8 -*-
"""AIML-Project"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE # 1. ADDED SMOTE IMPORT
import joblib

# Load the dataset
df = pd.read_csv('KaggleV2-May-2016.csv')
print(df.head())

# 1. Rename columns
df.rename(columns={'Hipertension': 'Hypertension', 'Handcap': 'Handicap', 'No-show':'NoShow'}, inplace=True)

# 2. Convert to datetime AND normalize to midnight
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']).dt.normalize()
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']).dt.normalize()

# 3. FEATURE ENGINEERING
df['WaitDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
df = df[df['WaitDays'] >= 0] # Remove anomalies

# 4. Convert Target Variable
df['NoShow'] = df['NoShow'].map({'Yes': 1, 'No': 0})

# 5. Convert Gender
df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})

# 6. Drop columns
cols_to_drop = ['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay', 'Neighbourhood']
df = df.drop(columns=cols_to_drop)

print("Data shape after preprocessing:", df.shape)

# Separate Features (X) and Target (y)
X = df.drop('NoShow', axis=1)
y = df['NoShow']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- CLEAN NANS ---
nan_mask_train = np.isnan(X_train_scaled).any(axis=1) | y_train.isnull()
X_train_scaled_cleaned = X_train_scaled[~nan_mask_train]
y_train_cleaned = y_train[~nan_mask_train]

nan_mask_test = np.isnan(X_test_scaled).any(axis=1) | y_test.isnull()
X_test_scaled_cleaned = X_test_scaled[~nan_mask_test]
y_test_cleaned = y_test[~nan_mask_test]

# --- 2. APPLY SMOTE HERE ---
print("\nApplying SMOTE to balance the training data...")
smote = SMOTE(random_state=42)
# We only resample the CLEANED training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled_cleaned, y_train_cleaned)

print(f"Original training target distribution:\n{y_train_cleaned.value_counts()}")
print(f"New SMOTE training target distribution:\n{y_train_resampled.value_counts()}")


# --- MODEL TRAINING (Using Resampled Data) ---
print("\n--- Logistic Regression ---")
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_resampled, y_train_resampled) # 3. UPDATED TO USE RESAMPLED DATA

lr_preds = lr_model.predict(X_test_scaled_cleaned)
print("Accuracy:", accuracy_score(y_test_cleaned, lr_preds))
print(classification_report(y_test_cleaned, lr_preds))

print("\n--- Decision Tree ---")
# Increased max_depth slightly to 10 to handle the larger, balanced dataset better
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42) 
dt_model.fit(X_train_resampled, y_train_resampled) # 4. UPDATED TO USE RESAMPLED DATA

dt_preds = dt_model.predict(X_test_scaled_cleaned)
print("Accuracy:", accuracy_score(y_test_cleaned, dt_preds))
print(classification_report(y_test_cleaned, dt_preds))

# Get Feature Importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:\n", feature_importances)

# Export Models
best_model = dt_model
joblib.dump(best_model, 'noshow_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("\n✅ Model and Scaler exported successfully! SMOTE Upgrade Complete.")