import pandas as pd
import os

# ✅ Path where all 8 CSV files are stored
data_dir = r"C:\Users\Ishita\OneDrive\Desktop\MAJOR-1\CIC-IDS2017\data"

# ✅ List all CSV files in that folder
files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".csv")]

print("🔍 Found CSV files:")
for f in files:
    print(" -", f)

# ✅ Read and merge them
df_list = [pd.read_csv(f, low_memory=False) for f in files]
merged_df = pd.concat(df_list, ignore_index=True)

# ✅ Save the combined dataset (it will save next to your data folder)
output_path = os.path.join(os.path.dirname(data_dir), "CICIDS2017_merged.csv")
merged_df.to_csv(output_path, index=False)

print(f"\n✅ Merged all CSVs successfully! Saved to: {output_path}")
print("📊 Final shape:", merged_df.shape)
