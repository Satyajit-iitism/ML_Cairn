import pandas as pd
import os

# ✅ List of well base filenames (CSV files must be named as <well>.csv)
well_files = [
    "AISH-007-08-206", "AISH-010-02-205", "AISH-017-08-202", "AISH-018-06-207",
    "AISH-022-07-106", "AISH-040-08-209", "AISHWARIYA-1z", "AISHWARIYA-2z",
    "AISHWARIYA-3", "AISHWARIYA-4", "AISHWARIYA-5", "AISHWARIYA-6z"
]

# ✅ List of formation names to retain for encoding
valid_formations = ['BH', 'FA1', 'FA3', 'BASE']

# ✅ Function to clean, filter, and one-hot encode formation column
def prepare_formation_for_model(input_csv, output_csv):
    try:
        # Load CSV
        df = pd.read_csv(input_csv)

        # Drop fully empty rows
        df.dropna(how='all', inplace=True)

        # Drop rows with missing FORMATION
        df = df.dropna(subset=['FORMATION'])

        # Clean FORMATION names
        df['FORMATION'] = df['FORMATION'].astype(str).str.strip().str.upper()

        # Filter only valid formation names
        df = df[df['FORMATION'].isin(valid_formations)]

        # One-hot encode FORMATION column
        df_encoded = pd.get_dummies(df, columns=['FORMATION'])

        # Save the cleaned DataFrame
        df_encoded.to_csv(output_csv, index=False)
        print(f"✅ Processed and saved: {output_csv}")

    except Exception as e:
        print(f"❌ Error processing {input_csv}: {e}")

# ✅ Main loop: process each well's CSV
for well in well_files:
    input_file = f"{well}.csv"
    output_file = f"{well}_encoded.csv"

    if os.path.exists(input_file):
        prepare_formation_for_model(input_file, output_file)
    else:
        print(f"⚠️ File not found: {input_file}")
