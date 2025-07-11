import pandas as pd
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ List of encoded well CSV files to use for model training and evaluation
file_names = [
    "AISH-007-08-206_encoded", "AISH-010-02-205_encoded", "AISH-017-08-202_encoded", "AISH-018-06-207_encoded",
    "AISH-022-07-106_encoded", "AISH-029-05-106_encoded", "AISH-040-08-209_encoded", "AISHWARIYA-1z_encoded",
    "AISHWARIYA-2z_encoded", "AISHWARIYA-3_encoded", "AISHWARIYA-4_encoded", "AISHWARIYA-5_encoded", "AISHWARIYA-6z_encoded"
]

# ✅ Define input logs and prediction target
numerical_features = ['GR_1', 'RHOB_1', 'TNPH_1', 'LLD_1']
target = 'VP_PRED'

# ✅ Shuffle and split wells into train/validation/test
random.seed(42)
random.shuffle(file_names)
train_wells = file_names[:10]
val_wells = file_names[10:11]
test_wells = file_names[11:]

print(f"📘 Train wells     ({len(train_wells)}): {train_wells}")
print(f"📙 Validation well ({len(val_wells)}): {val_wells}")
print(f"📕 Test wells      ({len(test_wells)}): {test_wells}")

# ✅ Load and preprocess data
def load_and_prepare(files):
    df_all = pd.DataFrame()
    for file in files:
        path = f"{file}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.dropna(subset=numerical_features + [target], inplace=True)

            # Convert one-hot encoded formation columns to int
            formation_columns = [col for col in df.columns if col.startswith("FORMATION_")]
            for col in formation_columns:
                df[col] = df[col].astype(int)

            df_all = pd.concat([df_all, df], ignore_index=True)
        else:
            print(f"❌ File not found: {path}")
    return df_all

# ✅ Load data
train_df = load_and_prepare(train_wells)
val_df = load_and_prepare(val_wells)
test_df = load_and_prepare(test_wells)

# ✅ Define final feature set
formation_columns = [col for col in train_df.columns if col.startswith("FORMATION_")]
features = numerical_features + formation_columns

# ✅ Prepare training and test arrays
X_train, y_train = train_df[features], train_df[target]
X_val, y_val = val_df[features], val_df[target]
X_test, y_test = test_df[features], test_df[target]

# ✅ Initialize and train XGBoost model
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)
model.fit(X_train, y_train)

# ✅ Predict on validation and test sets
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# ✅ Evaluation report
def report_metrics(true, pred, label=""):
    mae = mean_absolute_error(true, pred)
    rmse = mean_squared_error(true, pred, squared=False)
    r2 = r2_score(true, pred)
    print(f"\n📊 {label} Evaluation:")
    print(f"MAE  = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R²   = {r2:.4f}")

report_metrics(y_val, y_val_pred, label="Validation")
report_metrics(y_test, y_test_pred, label="Test")

# ✅ Plot: Actual vs Predicted for test wells
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_test_pred, alpha=0.3, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual VP_PRED")
plt.ylabel("Predicted VP_PRED")
plt.title("XGBoost - VP_PRED Prediction (Test Set)")
plt.grid(True)
plt.tight_layout()
plt.show()
