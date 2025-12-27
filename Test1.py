import pandas as pd
import os
import glob
from collections import defaultdict

# --- Main processing pipeline ---
import Methods

input_dir: str = "C:/Users/Nic/Desktop/Stat_PL/input"
output_dir: str = "C:/Users/Nic/Desktop/Stat_PL"
os.makedirs(output_dir, exist_ok=True)

# Collect all Excel and CSV files
data_files: list[str] = glob.glob(os.path.join(input_dir, "*.csv")) + glob.glob(os.path.join(input_dir, "*.xlsx"))
print(data_files)

# --- Prepare storage ---
all_raw_dfs: dict[str, list[pd.DataFrame]] = defaultdict(list)
empty_dfs: dict[str, pd.DataFrame] = {}

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

        # Create empty_df ONLY if it doesn't exist
        if file_type not in empty_dfs:
            empty_df: pd.DataFrame = Methods.empty_dataframe(Methods.schedule_dict)
            empty_dfs[file_type] = empty_df

            # Save reference empty CSV
            output_file_path = os.path.join(output_dir, f"reference_empty_{file_type}.csv")
            empty_df.to_csv(output_file_path, index=True, sep=';')
            print(f"✅ Reference empty CSV created for Type {file_type}: {output_file_path}")

        # Store raw_df for processing
        all_raw_dfs[file_type].append(df_raw)

    else:
        print(f"⚠️ Could not detect file type (A, B, or C) for: {input_file_path}. Skipping.")

# --------------------------------------------------------------------------
#                     PHASE 2: Process Each Raw DataFrame
# --------------------------------------------------------------------------

# Ensure output directory exists
output_dir: str = "combined_raw_csvs"
os.makedirs(output_dir, exist_ok=True)

final_merged_dfs: dict[str, pd.DataFrame] = {}

# TODO: Check if all three raw_dfs are saved in the same empty_df for each A, B and C
# Is empty_df updated or recreated and previous information from raw_df is lost

for file_type, list_of_raw_dfs in all_raw_dfs.items():
    print(f"\nProcessing files for Type {file_type}...")

    time_sorted_df: pd.DataFrame = empty_dfs[file_type]

    for idx, raw_df in enumerate(list_of_raw_dfs):
        try:
            # Fill in empty time points
            time_sorted_df_raw, time_sorted_df_empty = Methods.adjust_msm_in_raw_empty(raw_df=raw_df,
                                                                                       empty_df=time_sorted_df,
                                                                                       file_type=file_type,
                                                                                       output_dir=output_dir,
                                                                                       time_col_raw="1_TIme",
                                                                                       time_col_empty="time")

            # Construct output file path for each raw file
            suffix = f"_{file_type}_merged_{idx+1}"
            output_file_path: str = os.path.join(output_dir, f"final_data{suffix}.csv")

            # Save merged DataFrame
            time_sorted_df.to_csv(output_file_path, index=False, sep=';')
            print(f"✅ Merged CSV created: {output_file_path} (Shape: {time_sorted_df.shape})")

            # Store in dictionary
            final_merged_dfs[f"{file_type}_{idx+1}"] = time_sorted_df

        except Exception as e:
            print(f"🛑 Error during merge/save for Type {file_type} (file #{idx+1}): {e}")

    # --- Sort for msm and date ---
    time_sorted_df: pd.DataFrame = time_sorted_df.sort_values(['date', "msm"]).reset_index(drop=True)

    # Construct output file path for each raw file
    suffix = f"_{file_type}_merged_final"
    output_file_path: str = os.path.join(output_dir, f"final_data{suffix}.csv")

    # Save merged DataFrame
    time_sorted_df.to_csv(output_file_path, index=False, sep=';')
    print(f"✅ Merged CSV created: {output_file_path} (Shape: {time_sorted_df.shape})")
    # new Method!

    # Check empty_df data for missing data points, use vector operations
    # suggest with print() which possible data points from the left over points from raw_df can fill the missing spots

    # new Method
    # ggf. redo/fill missing data with left over points from raw_df

    # List = [ec5_uuid] or alternative when faster
    #ID_dict[input_file_path] = time_sorted_df[ec5_uuid]

    # Take row from df_raw when ID_dict[input_file_path] True
    raw_working_df: pd.DataFrame = ...
    # split each row into multiple individual rows and keep the ec5_uuid for every new row
    # header splits into multiple rows: ec5_uuid / creatred_at / TIme --> is the same for every new row
    # number of new rows depending on how often the following pattern occurs in header
    # Number1_ID_Behavior / Number2_Contact / Number3_AR / Number4_3M / Number5_Occurence / Number6_Notes
    # Name = >ID<_Behavior (from header)
    # Behaviour = value (from cell in column ID_Behaviour)
    # New header for entire df
    # ec5_uuid / created_at / TIme / Name / Behaviour / Contact / AR / 3M / Notes
    split_df: pd.DataFrame = ...

    # export split_df as excel_data to check for errors

    # itterating with vector through df with all raw data

        # Sorting into distance, behaviour and aggression data

        # = Methods.reshape_row_to_multiple(row)

        # Creation of distance and behaviour excel sheet


    # combining into one big dataframe all_data_behaviour, all_data_distance, all_data_occurence

    # Combining All-Occurence Data with play and Aggression Data from 2 min scanns