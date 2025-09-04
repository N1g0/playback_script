import os
import pandas as pd
from datetime import datetime, timedelta
import re
import glob

# --- Main processing pipeline ---

input_dir = "C:/Users/Nic/Desktop/Stat_PL/input"
output_dir = "C:/Users/Nic/Desktop/Stat_PL"
os.makedirs(output_dir, exist_ok=True)

# Collect all Excel and CSV files
data_files = glob.glob(os.path.join(input_dir, "*.csv")) + glob.glob(os.path.join(input_dir, "*.xlsx"))

# Prepare combined storage
all_data = {}

for input_file_path in data_files:
    print(f"Processing: {input_file_path}")

    if input_file_path.endswith(".csv"):
        df_raw = pd.read_csv(input_file_path, sep=';')
    else:
        df_raw = pd.read_excel(input_file_path)

    behave_dict = get_behavior_code_dict(df_raw)

    # Empty dataframe
    empty_df = empty_dataframe(schedule_dict)

    print()
    # Step 1: Reshape data
#    rs_df, _ = reshape_behavior_data(df_raw, schedule_dict, Cage_Compositions)
