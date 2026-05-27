# ============================================================
# AIR POLLUTION PREDICTION - COMPLETE END-TO-END PIPELINE
# ============================================================
# Author: Air Quality Research Team
# Description: 1D CNN model to classify AQI into Safe / Polluted
#              with GIS outputs and Power BI-ready exports.
# ============================================================

import os
import sys
import io
import warnings
warnings.filterwarnings('ignore')

# Fix Windows console encoding (cp1252 can't print Unicode)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── Core ────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import pickle

# ── Visualisation ───────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (no pop-up needed)
import matplotlib.pyplot as plt
import seaborn as sns

# ── ML / DL ─────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ── Folder setup ─────────────────────────────────────────────
for folder in ['data', 'model', 'outputs', 'dashboard', 'gis']:
    os.makedirs(folder, exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# 1. LOAD DATASET
# ============================================================
print("\n" + "="*55)
print("  STEP 1 — LOADING DATASET")
print("="*55)

DATA_PATH = 'data/air_pollution.csv'

if not os.path.exists(DATA_PATH):
    sys.exit(f"[ERROR] Dataset not found at '{DATA_PATH}'.\n"
             "  Run  python generate_data.py  first.")

data = pd.read_csv(DATA_PATH)
print(f"Dataset loaded -- {data.shape[0]} rows, {data.shape[1]} columns")
print("\nFirst 5 rows:")
print(data.head())
print("\nColumn types:")
print(data.dtypes)


# ============================================================
# 2. DATA CLEANING — HANDLE MISSING VALUES
# ============================================================
print("\n" + "="*55)
print("  STEP 2 — DATA CLEANING")
print("="*55)

missing_before = data.isnull().sum().sum()
print(f"Missing values BEFORE cleaning: {missing_before}")

# Fill numeric columns with their median (robust to outliers)
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

missing_after = data.isnull().sum().sum()
print(f"Missing values AFTER  cleaning: {missing_after}")
print("Data cleaning complete [OK]")


# ============================================================
# 3. FEATURE SELECTION
# ============================================================
print("\n" + "="*55)
print("  STEP 3 — FEATURE SELECTION")
print("="*55)

FEATURES = ['Temperature', 'Humidity', 'PM2.5', 'PM10']
TARGET   = 'AQI'

X = data[FEATURES].values
y_raw = data[TARGET].values

print(f"Input features : {FEATURES}")
print(f"Target column  : {TARGET}")


# ============================================================
# 4. AQI CLASSIFICATION  (AQI < 100 → Safe=0, else Polluted=1)
# ============================================================
print("\n" + "="*55)
print("  STEP 4 — AQI CLASSIFICATION")
print("="*55)

y = (y_raw >= 100).astype(int)        # 0 = Safe, 1 = Polluted
safe_count     = int((y == 0).sum())
polluted_count = int((y == 1).sum())

print(f"AQI threshold  : 100")
print(f"  Safe      (0): {safe_count:,}  ({safe_count/len(y)*100:.1f} %)")
print(f"  Polluted  (1): {polluted_count:,}  ({polluted_count/len(y)*100:.1f} %)")


# ── AQI Distribution chart ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(y_raw, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
ax.axvline(100, color='red', linewidth=2, linestyle='--', label='Threshold (AQI=100)')
ax.set_xlabel('AQI Value', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('AQI Distribution in Dataset', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/aqi_distribution.png', dpi=150)
plt.close()
print("Chart saved -> outputs/aqi_distribution.png")


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================
print("\n" + "="*55)
print("  STEP 5 — TRAIN / TEST SPLIT  (70/30)")
print("="*55)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, data.index,
    test_size=0.30, random_state=42, stratify=y
)
print(f"Training set : {X_train.shape[0]:,} samples")
print(f"Testing  set : {X_test.shape[0]:,} samples")


# ============================================================
# 6. STANDARDISE DATA
# ============================================================
print("\n" + "="*55)
print("  STEP 6 — STANDARDISATION")
print("="*55)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Save scaler for later use (dashboard / API)
with open('model/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved  -> model/scaler.pkl")


# ============================================================
# 7. RESHAPE FOR 1D CNN  →  (samples, features, 1)
# ============================================================
print("\n" + "="*55)
print("  STEP 7 — RESHAPING DATA FOR 1D CNN")
print("="*55)

X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn  = X_test.reshape( X_test.shape[0],  X_test.shape[1],  1)
print(f"X_train shape : {X_train_cnn.shape}")
print(f"X_test  shape : {X_test_cnn.shape}")


# ============================================================
# 8. BUILD 1D CNN MODEL
# ============================================================
print("\n" + "="*55)
print("  STEP 8 — BUILDING MODEL")
print("="*55)

n_features = X_train_cnn.shape[1]

model = Sequential([
    # ── Block 1 ─────────────────────────────────────────────
    Conv1D(filters=64, kernel_size=2, activation='relu',
           input_shape=(n_features, 1), padding='same', name='conv1'),
    MaxPooling1D(pool_size=1, name='pool1'),
    Dropout(0.25, name='drop1'),

    # ── Block 2 ─────────────────────────────────────────────
    Conv1D(filters=128, kernel_size=2, activation='relu',
           padding='same', name='conv2'),
    MaxPooling1D(pool_size=1, name='pool2'),
    Dropout(0.25, name='drop2'),

    # ── Classifier head ─────────────────────────────────────
    Flatten(name='flatten'),
    Dense(128, activation='relu', name='dense1'),
    Dropout(0.30, name='drop3'),
    Dense(64, activation='relu', name='dense2'),
    Dense(1, activation='sigmoid', name='output')   # Binary output
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\nMODEL SUMMARY")
model.summary()


# ============================================================
# 9. TRAIN MODEL
# ============================================================
print("\n" + "="*55)
print("  STEP 9 — TRAINING MODEL")
print("="*55)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint('model/cnn_aqi_model.h5', monitor='val_accuracy',
                    save_best_only=True, verbose=0)
]

history = model.fit(
    X_train_cnn, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.20,
    callbacks=callbacks,
    verbose=1
)
print("Model training complete [OK]")
print("Best model saved -> model/cnn_aqi_model.h5")


# ============================================================
# 10. EVALUATE MODEL
# ============================================================
print("\n" + "="*55)
print("  STEP 10 — EVALUATION")
print("="*55)

y_prob = model.predict(X_test_cnn, verbose=0)
y_pred = (y_prob > 0.5).astype(int).flatten()

accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*40}")
print(f"  MODEL ACCURACY : {accuracy*100:.2f} %")
print(f"{'='*40}")
print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, y_pred,
                            target_names=['Safe (0)', 'Polluted (1)']))
print("CONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))


# ── Accuracy / Loss plots ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history.history['accuracy'],     label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Val Accuracy',   linewidth=2, linestyle='--')
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history.history['loss'],     label='Train Loss', linewidth=2, color='tomato')
axes[1].plot(history.history['val_loss'], label='Val Loss',   linewidth=2, linestyle='--', color='coral')
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/training_history.png', dpi=150)
plt.close()
print("\nChart saved -> outputs/training_history.png")


# ── Confusion matrix heatmap ─────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Safe', 'Polluted'],
            yticklabels=['Safe', 'Polluted'],
            linewidths=0.5, ax=ax)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual',    fontsize=12)
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=150)
plt.close()
print("Chart saved -> outputs/confusion_matrix.png")


# ============================================================
# 11. GIS-READY PREDICTIONS CSV
# ============================================================
print("\n" + "="*55)
print("  STEP 11 — GENERATING GIS OUTPUT")
print("="*55)

test_data = data.loc[idx_test].copy().reset_index(drop=True)
test_data['Actual_Class']    = y_test
test_data['Predicted_Class'] = y_pred
test_data['Predicted_Prob']  = y_prob.flatten().round(4)
test_data['AQI_Category']    = test_data['Predicted_Class'].map(
    {0: 'Safe', 1: 'Polluted'}
)

gis_output = test_data[['Latitude', 'Longitude', 'AQI',
                          'Actual_Class', 'Predicted_Class',
                          'Predicted_Prob', 'AQI_Category']].copy()
gis_output.columns = ['Latitude', 'Longitude', 'Actual_AQI',
                       'Actual_Class', 'Predicted_Class',
                       'Predicted_Prob', 'AQI_Category']

gis_output.to_csv('gis/predictions.csv', index=False)
print(f"GIS file saved -> gis/predictions.csv  ({len(gis_output):,} rows)")
print("  (Import directly into QGIS using 'Add Delimited Text Layer')")


# ============================================================
# 12. POWER BI DASHBOARD EXPORTS
# ============================================================
print("\n" + "="*55)
print("  STEP 12 — POWER BI READY EXPORTS")
print("="*55)

# ── Full predictions table ───────────────────────────────────
full_pred = data.copy()
all_X = scaler.transform(data[FEATURES].values)
all_X_cnn = all_X.reshape(all_X.shape[0], all_X.shape[1], 1)
all_prob = model.predict(all_X_cnn, verbose=0).flatten()
all_class = (all_prob > 0.5).astype(int)

full_pred['Predicted_Class']    = all_class
full_pred['Predicted_Prob']     = all_prob.round(4)
full_pred['AQI_Category']       = pd.Categorical(
    all_class, categories=[0, 1]
).rename_categories({0: 'Safe', 1: 'Polluted'})

full_pred.to_csv('dashboard/full_predictions.csv', index=False)
print("Saved -> dashboard/full_predictions.csv")

# ── KPI summary ─────────────────────────────────────────────
kpi = {
    'Total_Samples':      [len(full_pred)],
    'Safe_Count':         [int((all_class == 0).sum())],
    'Polluted_Count':     [int((all_class == 1).sum())],
    'Safe_Percent':       [round(float((all_class == 0).mean() * 100), 2)],
    'Polluted_Percent':   [round(float((all_class == 1).mean() * 100), 2)],
    'Model_Accuracy_Pct': [round(accuracy * 100, 2)],
    'Mean_AQI':           [round(float(data['AQI'].mean()), 2)],
    'Max_AQI':            [round(float(data['AQI'].max()), 2)],
    'Min_AQI':            [round(float(data['AQI'].min()), 2)],
}
pd.DataFrame(kpi).to_csv('dashboard/kpi_summary.csv', index=False)
print("Saved -> dashboard/kpi_summary.csv")

# ── Area-wise summary (if 'Area' column exists) ──────────────
if 'Area' in data.columns:
    area_summary = full_pred.groupby('Area').agg(
        Avg_AQI=('AQI', 'mean'),
        Polluted_Count=('Predicted_Class', 'sum'),
        Total=('Predicted_Class', 'count')
    ).reset_index()
    area_summary['Pollution_Rate_Pct'] = (
        area_summary['Polluted_Count'] / area_summary['Total'] * 100
    ).round(2)
    area_summary.to_csv('dashboard/area_summary.csv', index=False)
    print("Saved -> dashboard/area_summary.csv")


# ============================================================
# 13. SAMPLE PREDICTION  (single input)
# ============================================================
print("\n" + "="*55)
print("  STEP 13 — SAMPLE PREDICTION DEMO")
print("="*55)

sample_raw = np.array([[30, 65, 120, 180]])   # Temp, Humidity, PM2.5, PM10
sample_scaled = scaler.transform(sample_raw)
sample_cnn    = sample_scaled.reshape(1, n_features, 1)
sample_prob   = float(model.predict(sample_cnn, verbose=0)[0][0])
sample_class  = 'POLLUTED [!]' if sample_prob > 0.5 else 'SAFE [OK]'

print(f"  Input → Temperature=30, Humidity=65, PM2.5=120, PM10=180")
print(f"  Probability (Polluted): {sample_prob:.4f}")
print(f"  Prediction            : {sample_class}")


# ============================================================
# 14. FINAL SUMMARY
# ============================================================
print("\n" + "="*55)
print("  PROJECT COMPLETE — FILE SUMMARY")
print("="*55)
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git']]
    level = root.replace('.', '').count(os.sep)
    indent = ' ' * 4 * level
    print(f'{indent}{os.path.basename(root)}/')
    subindent = ' ' * 4 * (level + 1)
    for file in files:
        fpath = os.path.join(root, file)
        size  = os.path.getsize(fpath)
        print(f'{subindent}{file}  ({size:,} bytes)')

print("\n[SUCCESS] Air Pollution Prediction Pipeline finished successfully!")
print("   Open 'dashboard/' and 'gis/' files in Power BI / QGIS.")
