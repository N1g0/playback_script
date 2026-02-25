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
#TODO: Implement All Occurence data into processing pipeline
# Fact-check data if data processing was successful and no data are missing
# Build data search function?
# CHeck why B is not in the final data collection

# Define subfolders for organization
steps_dir = os.path.join(output_dir, "intermediate_steps")
os.makedirs(steps_dir, exist_ok=True)

# Global lists to store all data across all file types and chunks
all_behave_data = []
all_dist_data = []
all_occ_data = []

for file_type, list_of_raw_dfs in all_raw_dfs.items():
    print(f"\nProcessing files for Type {file_type}...")

    for idx, raw_df in enumerate(list_of_raw_dfs):
        try:
            # Time Adjustment
            time_sorted_df_raw, _ = Methods.adjust_msm_in_raw_empty(
                raw_df=raw_df,
                file_type=file_type,
                output_dir=output_dir,
                time_col_raw="1_TIme",
                time_col_empty="time"
            )
            # Save
            time_sorted_df_raw.to_csv(os.path.join(steps_dir, f"type{file_type}_file{idx}_step1_time.csv"), index=False)

            # Detection and Renaming (event_sorted_df)
            event_sorted_df = Methods.process_sort_event(time_sorted_df_raw, file_type)
            # Save
            event_sorted_df.to_csv(os.path.join(steps_dir, f"type{file_type}_file{idx}_step2_event.csv"), index=False)

            # Reshaping (The 3 tables)
            behave_df, dist_df, occ_df = Methods.process_sort_beh_dist(event_sorted_df)

            # Save (Individual file outputs)
            behave_df.to_csv(os.path.join(steps_dir, f"type{file_type}_file{idx}_step3_behave.csv"), index=False)
            dist_df.to_csv(os.path.join(steps_dir, f"type{file_type}_file{idx}_step3_dist.csv"), index=False)

            # Add to global lists for the final combined files
            all_behave_data.append(behave_df)
            all_dist_data.append(dist_df)
            all_occ_data.append(occ_df)

            print(f"✅ Finished file #{idx + 1}")

        except Exception as e:
            print(f"🛑 Error processing Type {file_type} (file #{idx + 1}): {e}")

# --- FINAL COMBINATION STEP ---
print("\nCombining all files into final outputs...")

if all_behave_data:
    final_behave = pd.concat(all_behave_data, ignore_index=True)
    final_behave.to_csv(os.path.join(output_dir, "final_combined_behaviour.csv"), index=False)
    print(f"📁 Saved final_combined_behaviour.csv ({len(final_behave)} rows)")

if all_dist_data:
    final_dist = pd.concat(all_dist_data, ignore_index=True)
    final_dist.to_csv(os.path.join(output_dir, "final_combined_distance.csv"), index=False)
    print(f"📁 Saved final_combined_distance.csv ({len(final_dist)} rows)")

if all_occ_data:
    final_occ = pd.concat(all_occ_data, ignore_index=True)
    final_occ.to_csv(os.path.join(output_dir, "final_combined_occurrence.csv"), index=False)
    print(f"📁 Saved final_combined_occurrence.csv ({len(final_occ)} rows)")