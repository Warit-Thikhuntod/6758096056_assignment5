import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

os.makedirs('data/output', exist_ok=True)

print("Loading model...")
model_path = 'examples/data5_classification_model.h5'
if os.path.exists(model_path):
    model = keras.models.load_model(model_path, compile=False)
    ensemble_models = [model]
else:
    print(f"Error: Model {model_path} not found!")
    exit()

with open('examples/data5_columns.txt', 'r') as f:
    EXPECTED_COLUMNS = f.read().split(',')

scaler = joblib.load('examples/data5_scaler.joblib')

threshold = 0.5
if os.path.exists('examples/data5_threshold.txt'):
    with open('examples/data5_threshold.txt', 'r') as f:
        threshold = float(f.read().strip())
    print(f"Loaded optimized threshold: {threshold:.3f}")

print("Loading data for inference...")
df = pd.read_csv('data/Data_5_inference.csv')
df = df.dropna().reset_index(drop=True)

X = df.drop('ProdTaken', axis=1)
y = df['ProdTaken']

def engineer_features(df_local):
    X_local = df_local.copy()
    X_local['Income_Pitch_Score'] = X_local['MonthlyIncome'] * X_local['PitchSatisfactionScore']
    X_local['DurationOfPitch_log'] = np.log1p(X_local['DurationOfPitch'])
    X_local['MonthlyIncome_log'] = np.log1p(X_local['MonthlyIncome'])
    
    X_local['Age_Income'] = X_local['Age'] * X_local['MonthlyIncome']
    X_local['Visits_Income'] = X_local['NumberOfPersonVisiting'] * X_local['MonthlyIncome']
    X_local['Pitch_Duration'] = X_local['PitchSatisfactionScore'] * X_local['DurationOfPitch']
    X_local['Income_per_Person'] = X_local['MonthlyIncome'] / (X_local['NumberOfPersonVisiting'] + 1)
    
    X_local['Age_2'] = X_local['Age'] ** 2
    X_local['MonthlyIncome_2'] = X_local['MonthlyIncome'] ** 2
    X_local['Duration_2'] = X_local['DurationOfPitch'] ** 2
    
    X_local['Age_Group'] = pd.cut(X_local['Age'], bins=[0, 30, 45, 60, 100], labels=['Young', 'Middle', 'Senior', 'Old'])
    
    return X_local

X = engineer_features(X)

categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

for col in EXPECTED_COLUMNS:
    if col not in X_encoded.columns:
        X_encoded[col] = 0

X_final = X_encoded[EXPECTED_COLUMNS]

X_scaled = scaler.transform(X_final)

print(f"\nMaking predictions with ensemble of {len(ensemble_models)} models...")
all_preds = []
for model in ensemble_models:
    all_preds.append(model.predict(X_scaled, verbose=0).flatten())

ensemble_probs = np.mean(all_preds, axis=0)
y_pred = (ensemble_probs > threshold).astype(int).flatten()

accuracy = np.mean(y_pred == y.values)
print(f"\nInference Dataset Accuracy (Threshold {threshold:.3f}):", accuracy)
print("\nClassification Report:")
print(classification_report(y.values, y_pred))

cm = confusion_matrix(y.values, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No', 'Yes'],
            yticklabels=['No', 'Yes'],
            cbar_kws={'label': 'Count'})

plt.title('Confusion Matrix - Data 5 Classification Model', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

tn, fp, fn, tp = cm.ravel()
stats_text = f'True Negatives: {tn}\nFalse Positives: {fp}\nFalse Negatives: {fn}\nTrue Positives: {tp}'
stats_text += f'\n\nAccuracy: {accuracy:.4f}'
plt.text(2.5, 0.5, stats_text, fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         verticalalignment='center')

plt.tight_layout()
plt.savefig('data/output/confusion_matrix_data5.jpg', dpi=300, bbox_inches='tight')
print("\nConfusion matrix saved to data/output/confusion_matrix_data5.jpg")

plt.figure(figsize=(10, 8))
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=['No', 'Yes'],
            yticklabels=['No', 'Yes'],
            cbar_kws={'label': 'Percentage'})
plt.title('Normalized Confusion Matrix - Data 5 Classification Model', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
plt.savefig('data/output/confusion_matrix_normalized_data5.jpg', dpi=300, bbox_inches='tight')
print(f"Normalized confusion matrix plot saved to: data/output/confusion_matrix_normalized_data5.jpg")