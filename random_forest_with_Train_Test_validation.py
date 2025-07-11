import pandas as pd
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ✅ List of all encoded well CSV filenames (without .csv extension)
file_names = [
    "AISH-007-08-206", "AISH-010-02-205", "AISH-017-08-202", "AISH-018-06-207",
    "AISH-022-07-106", "AISH-029-05-106", "AISH-040-08-209",
    "AISHWARIYA-1z", "AISHWARIYA-2z", "AISHWARIYA-3",
    "AISHWARIYA-4", "AISHWARIYA-5", "AISHWARIYA-6z"
]

# ✅ Feature columns and prediction target
numerical_features = ['GR_1', 'RHOB_1', 'TNPH_1', 'LLD_1']
target = 'VP_PRED'

# ✅ Split data: 10 for training, 1 for validation, 2 for testing
random.seed(42)
random.shuffle(file_names)
train_wells = file_names[:10]
val_wells = file_names[10:11]
test_wells = file_names[11:]

# ✅ Display selected wells for each set
print(f"📘 Train wells ({len(train_wells)}):", train_wells)
print(f"📙 Validation wells ({len(val_wells)}):", val_wells)
print(f"📕 Test wells ({len(test_wells)}):", test_wells)

# ✅ Function to load and prepare well data
def load_and_prepare(files):
    df_all = pd.DataFrame()
    for file in files:
        path = f"{file}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)

            # Drop rows with missing features or target
            df = df.dropna(subset=numerical_features + [target])

            # Convert one-hot encoded formations to integer (if needed)
            formation_columns = [col for col in df.columns if col.startswith("FORMATION_")]
            for col in formation_columns:
                df[col] = df[col].astype(int)

            df_all = pd.concat([df_all, df], ignore_index=True)
        else:
            print(f"❌ File not found: {path}")
    return df_all

# ✅ Load datasets
train_df = load_and_prepare(train_wells)
val_df = load_and_prepare(val_wells)
test_df = load_and_prepare(test_wells)

# ✅ Final input feature list
formation_columns = [col for col in train_df.columns if col.startswith("FORMATION_")]
features = numerical_features + formation_columns

# ✅ Model training
X_train = train_df[features]
y_train = train_df[target]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Predictions
X_val = val_df[features]
y_val = val_df[target]
y_val_pred = model.predict(X_val)

X_test = test_df[features]
y_test = test_df[target]
y_test_pred = model.predict(X_test)

# ✅ Evaluation function
def report_metrics(true, pred, label=""):
    mae = mean_absolute_error(true, pred)
    rmse = mean_squared_error(true, pred, squared=False)
    r2 = r2_score(true, pred)
    print(f"\n📊 {label} Evaluation:")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R²   = {r2:.4f}")

# ✅ Show metrics
report_metrics(y_val, y_val_pred, label="Validation")
report_metrics(y_test, y_test_pred, label="Test")

# ✅ Plot predictions
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_test_pred, alpha=0.3, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual VP_PRED")
plt.ylabel("Predicted VP_PRED")
plt.title("Random Forest - VP_PRED Prediction (Test Wells)")
plt.grid(True)
plt.tight_layout()
plt.show()
