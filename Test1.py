import pandas as pd
import os
import glob
from collections import defaultdict
import Methods

# --- Main processing pipeline ---

input_dir: str = "C:/Users/Nic/Desktop/Stat_PL/input"
output_dir: str = "C:/Users/Nic/Desktop/Stat_PL/Output"
os.makedirs(output_dir, exist_ok=True)

# Collect all Excel and CSV files
data_files: list[str] = glob.glob(os.path.join(input_dir, "*.csv")) + glob.glob(os.path.join(input_dir, "*.xlsx"))
print(data_files)

# --- Prepare storage ---
all_raw_dfs: dict[str, list[pd.DataFrame]] = defaultdict(list)

# --------------------------------------------------------------------------
#                          PHASE 1: Read and Store
# --------------------------------------------------------------------------

for input_file_path in data_files:
    print(f"Processing: {input_file_path}")

    # --- Skip All_Occurence files ---
    if "all_occurence" in input_file_path.lower():
        print(f"Skipping special file: {input_file_path}")
        continue

    # --- Read CSV or Excel ---
    try:
        if input_file_path.endswith(".csv"):
            df_raw: pd.DataFrame = pd.read_csv(input_file_path, sep=';')
        else:
            df_raw: pd.DataFrame = pd.read_excel(input_file_path)
    except Exception as e:
        print(f"🛑 Error reading file {input_file_path}: {e}")
        continue

    # --- Determine file type ---
    file_name = os.path.basename(input_file_path).lower()
    file_type: str | None = None

    if "_a_" in file_name:
        file_type = "A"
    elif "_b_" in file_name:
        file_type = "B"
    elif "_c_" in file_name:
        file_type = "C"

    if file_type:
        print(f"→ Detected {file_type} file: {input_file_path}")

        # Store raw_df for processing
        all_raw_dfs[file_type].append(df_raw)

    else:
        print(f"⚠️ Could not detect file type (A, B, or C) for: {input_file_path}. Skipping.")

# --------------------------------------------------------------------------
#                     PHASE 2: Process Each Raw DataFrame
# --------------------------------------------------------------------------
final_merged_dfs: dict[str, pd.DataFrame] = {}

for file_type, list_of_raw_dfs in all_raw_dfs.items():
    print(f"\nProcessing files for Type {file_type}...")

    processed_chunks = []

    for idx, raw_df in enumerate(list_of_raw_dfs):
        try:
            # Your method aligns the raw data to the fixed template timepoints
            time_sorted_df_raw, _ = Methods.adjust_msm_in_raw_empty(
                raw_df=raw_df,
                file_type=file_type,
                output_dir=output_dir,
                time_col_raw="1_TIme",
                time_col_empty="time"
            )
            print('time_sorted_df_raw: \n', time_sorted_df_raw)

            df_dec_ind: pd.DataFrame = Methods.detect_ind(time_sorted_df_raw)
            print('df_dec_ind: \n', df_dec_ind)
            # Fix date column
            df_dec_ind["date"] = pd.to_datetime(df_dec_ind["date"]).dt.strftime("%d-%m-%Y")
            # Fix time column
            df_dec_ind["1_TIme"] = pd.to_datetime(df_dec_ind["1_TIme"]).dt.strftime("%H:%M")



            processed_chunks.append(time_sorted_df_raw)

        except Exception as e:
            print(f"🛑 Error processing Type {file_type} (file #{idx + 1}): {e}")

