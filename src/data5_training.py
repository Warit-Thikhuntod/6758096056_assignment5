import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import joblib

print("Loading data...")
df = pd.read_csv('data/Data_5_train.csv')

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

columns_for_inference = X_encoded.columns.tolist()
os.makedirs('examples', exist_ok=True)
with open('examples/data5_columns.txt', 'w') as f:
    f.write(','.join(columns_for_inference))

X_train_orig, X_val_orig, y_train_orig, y_val_orig = train_test_split(
    X_encoded, y, test_size=0.1, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_orig)
X_val_scaled = scaler.transform(X_val_orig)
X_test_scaled = X_val_scaled
y_test = y_val_orig

joblib.dump(scaler, 'examples/data5_scaler.joblib')
print("Scaler saved to examples/data5_scaler.joblib")

is_positive = (y_train_orig.values == 1)
X_pos = X_train_scaled[is_positive]
y_pos = y_train_orig.values[is_positive]

is_negative = (y_train_orig.values == 0)
X_neg = X_train_scaled[is_negative]
y_neg = y_train_orig.values[is_negative]

n_neg = len(X_neg)
idx_pos = np.random.choice(len(X_pos), n_neg, replace=True)
X_train_final = np.vstack([X_neg, X_pos[idx_pos]])
y_train_final = np.concatenate([y_neg, y_pos[idx_pos]])

print(f"Oversampled training set: {len(X_train_final)} samples (Balanced)")

def build_model(input_dim):
    from tensorflow.keras.regularizers import l2
    
    input_layer = keras.Input(shape=(input_dim,))
    
    deep = keras.layers.Dense(256, activation='swish', kernel_regularizer=l2(0.001))(input_layer)
    deep = keras.layers.BatchNormalization()(deep)
    deep = keras.layers.Dropout(0.4)(deep)
    
    deep = keras.layers.Dense(128, activation='swish', kernel_regularizer=l2(0.001))(deep)
    deep = keras.layers.BatchNormalization()(deep)
    deep = keras.layers.Dropout(0.3)(deep)
    
    deep = keras.layers.Dense(64, activation='swish', kernel_regularizer=l2(0.001))(deep)
    deep = keras.layers.BatchNormalization()(deep)
    
    wide = keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(0.001))(input_layer)
    
    concat = keras.layers.concatenate([deep, wide])
    output = keras.layers.Dense(1, activation='sigmoid')(concat)
    
    model = keras.Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

ENSEMBLE_SIZE = 1
models = []
print(f"\nTraining Wide & Deep model...")

for i in range(ENSEMBLE_SIZE):
    print(f"\nTraining Model...")
    model = build_model(X_train_final.shape[1])
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    ]
    
    history = model.fit(
        X_train_final, y_train_final,
        epochs=150,
        batch_size=64,
        validation_data=(X_val_scaled, y_val_orig),
        verbose=0,
        callbacks=callbacks
    )
    
    model.save('examples/data5_classification_model.h5')
    models.append(model)
    print(f"Model saved. Final Train Loss: {history.history['loss'][-1]:.4f}, Final Train Acc: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Val Loss: {history.history['val_loss'][-1]:.4f}, Final Val Acc: {history.history['val_accuracy'][-1]:.4f}")

print("\nOptimizing threshold...")
all_preds = []
for model in models:
    all_preds.append(model.predict(X_test_scaled, verbose=0).flatten())

probs = np.mean(all_preds, axis=0)

best_t = 0.5
best_acc = np.mean((probs > 0.5) == y_test.values)

for t in np.arange(0.1, 0.9, 0.001):
    y_tmp = (probs > t).astype(int)
    acc = np.mean(y_tmp == y_test.values)
    if acc > best_acc:
        best_acc = acc
        best_t = t

print(f"Final Optimized Accuracy: {best_acc:.4f} at threshold {best_t:.3f}")

with open('examples/data5_threshold.txt', 'w') as f:
    f.write(str(best_t))

y_pred = (probs > best_t).astype(int)
print("\nClassification Report:")
print(classification_report(y_test.values, y_pred))

os.makedirs('data/output', exist_ok=True)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.savefig('data/output/learning_curves.png')
print("Sample learning curves saved.")