import pandas as pd

# Load your dataset
df = pd.read_csv("irrigation_dataset_with_rainfall.csv")

# Reorder columns: place 'rainfall' before 'result'
cols = list(df.columns)
cols.remove("rainfall")
cols.insert(cols.index("result"), "rainfall")
df = df[cols]

# Save as new CSV
df.to_csv("irrigation_dataset_with_rainfall_reordered.csv", index=False)

print("✅ New file saved as irrigation_dataset_with_rainfall_reordered.csv")
