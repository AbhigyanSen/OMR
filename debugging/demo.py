import pandas as pd
import json
import os

# Paths
SUMMARY_CSV = r"D:\Projects\OMR\new_abhigyan\debugging\TestData\Test_Series\summary.csv"
ICR_JSON = r"D:\Projects\OMR\new_abhigyan\debugging\annotate_Test_Series\output.json"
OUTPUT_CSV = r"D:\Projects\OMR\new_abhigyan\debugging\TestData\Test_Series\summary_icr.csv"

# Load summary CSV
df = pd.read_csv(SUMMARY_CSV)

# Load ICR results
with open(ICR_JSON, 'r', encoding='utf-8') as f:
    icr_data = json.load(f)

# Initialize new columns
df['ICR_Registration'] = None
df['ICR_Roll'] = None
df['ICR_Booklet'] = None

# Fill new columns using ICR data
for idx, row in df.iterrows():
    image_name = row['Image Name']

    if image_name in icr_data:
        icr_entry = icr_data[image_name]
        df.at[idx, 'ICR_Registration'] = icr_entry.get("Registration Number")
        df.at[idx, 'ICR_Roll'] = icr_entry.get("Roll Number")
        df.at[idx, 'ICR_Booklet'] = icr_entry.get("Question Booklet Number")

# Save the updated CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Saved updated CSV to: {OUTPUT_CSV}")