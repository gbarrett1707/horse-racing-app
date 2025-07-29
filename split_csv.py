import pandas as pd
import os

# Set the folder path to your Downloads/Racing Data Split
folder_path = os.path.expanduser("~/Downloads/Racing Data Split")
input_file = os.path.join(folder_path, "racing_data upload.csv")
chunk_size = 525000  # Number of rows per output file

# Load the CSV file
df = pd.read_csv(input_file)

# Split and save chunks
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i + chunk_size]
    output_file = os.path.join(folder_path, f"racing_data_part_{i // chunk_size + 1}.csv")
    chunk.to_csv(output_file, index=False)
    print(f"✅ Saved: {output_file}")

print("🎉 All parts saved successfully.")
