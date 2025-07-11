import pandas as pd
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ Encoded CSV files (output from previous formation encoding step)
file_names = [
    "AISH-007-08-206_encoded", "AISH-010-02-205_encoded", "AISHWARIYA-3_encoded", "AISHWARIYA-4_encoded", "AISHWARIYA-5_encoded",
    "AISH-017-08-202_encoded", "AISH-018-06-207_encoded", "AISH-022-07-106_encoded", "AISH-029-05-106_encoded",
    "AISH-040-08-209_encoded", "AISHWARIYA-1z_encoded", "AISHWARIYA-2z_encoded", "AISHWARIYA-6z_encoded"
]

# ✅ Define log features and target
numerical_features = ['GR_1', 'RHOB_1', 'TNPH_1', 'LLD_1']
target = 'VP_PRED'

# ✅ Split into training and testing sets manually
train_wells = file_names[:10]
test_wells = file_names[10:]

print(f"📘 Train wells ({len(train_wells)}): {train_wells}")
print(f"📕 Test wells  ({len(test_wells)}): {test_wells}")

# ✅ Load and prepare data from a list of CSV files
def load_and_prepare(files, required_formation_cols=None):
    df_all = pd.DataFrame()
    for file in files:
        path = f"{file}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df.dropna(subset=numerical_features + [target])
            formation_cols = [col for col in df.columns if col.startswith("FORMATION_")]
            for col in formation_cols:
                df[col] = df[col].astype(int)

            if required_formation_cols:
                for col in required_formation_cols:
                    if col not in df.columns:
                        df[col] = 0
                df = df[numerical_features + required_formation_cols + [target]]

            df_all = pd.concat([df_all, df], ignore_index=True)
        else:
            print(f"❌ File not found: {path}")
    return df_all

# ✅ Load training data
train_df = load_and_prepare(train_wells)

# ✅ Extract formation columns for consistent test encoding
formation_columns = [col for col in train_df.columns if col.startswith("FORMATION_")]
features = numerical_features + formation_columns

# ✅ Load test data with aligned formation columns
def load_test_data(files, formation_columns):
    df_all = pd.DataFrame()
    for file in files:
        path = f"{file}.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = df.dropna(subset=numerical_features + [target])
            for col in formation_columns:
                if col in df.columns:
                    df[col] = df[col].astype(int)
                else:
                    df[col] = 0
            df_all = pd.concat([df_all, df], ignore_index=True)
        else:
            print(f"❌ File not found: {path}")
    return df_all

test_df = load_test_data(test_wells, formation_columns)

# ✅ Prepare X, y
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# ✅ Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Predict and evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("\n📊 VP_PRED Evaluation on Test Set:")
print(f"MAE  = {mae:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")

# ✅ Plot: Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.3, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Vp")
plt.ylabel("Predicted Vp")
plt.title("Random Forest - VP_PRED Prediction")
plt.grid(True)
plt.tight_layout()
plt.show()
