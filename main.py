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

# --- Prepare storage ---
all_raw_dfs: dict[str, list[pd.DataFrame]] = defaultdict(list)
all_occ_dfs: dict[str, list[pd.DataFrame]] = defaultdict(list)

# Global lists to store all data across all file types and chunks
all_behave_data: list = []
all_dist_data: list = []
all_occ_data: list = []

# --------------------------------------------------------------------------
#                          PHASE 1: Read and Store
# --------------------------------------------------------------------------

for input_file_path in data_files:
    print(f"Processing: {input_file_path}")

    # --- Skip All_Occurence files ---
    if "all_occurence" in input_file_path.lower():
        df_occ: pd.DataFrame = Methods.read_data(input_file_path)
        all_occ_data, file_type = Methods.file_name(input_file_path, df_occ, all_occ_dfs)
        all_occ_dfs = Methods.process_all_occurrence(df_occ, file_type, all_occ_dfs, output_dir)
        continue

    df_raw: pd.DataFrame = Methods.read_data(input_file_path)
    all_raw_dfs, file_type = Methods.file_name(input_file_path, df_raw, all_raw_dfs)

# --------------------------------------------------------------------------
#                     PHASE 2: Process Each Raw DataFrame
# --------------------------------------------------------------------------
#TODO: Concate both occ dfs into one and save in final_occ/fix df stucture first
# Fact-check data if data processing was successful and no data are missing
# Build data search function?

# Define subfolders for organization
steps_dir = os.path.join(output_dir, "intermediate_steps")
os.makedirs(steps_dir, exist_ok=True)

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
            print('occ_df: \n', occ_df)
            #TODO: Fix qaulifier and .append all_occ.dfs is Dataframe not dict anymore (change to one or the other)

            # Save (Individual file outputs)
            behave_df.to_csv(os.path.join(steps_dir, f"type{file_type}_file{idx}_step3_behave.csv"), index=False)
            dist_df.to_csv(os.path.join(steps_dir, f"type{file_type}_file{idx}_step3_dist.csv"), index=False)

            # Add to global lists for the final combined files
            all_behave_data.append(behave_df)
            all_dist_data.append(dist_df)
            all_occ_dfs[file_type].append(occ_df)

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

all_dfs_list = []
for cage_type in all_occ_dfs:
    all_dfs_list.extend(all_occ_dfs[cage_type])

# 2. Check if the flattened list has any data
if all_dfs_list:
    # Concatenate all dataframes into one long "final" dataframe
    final_occ = pd.concat(all_dfs_list, ignore_index=True)

    # Save to CSV
    output_path = os.path.join(output_dir, "final_combined_occurrence.csv")
    final_occ.to_csv(output_path, index=False)

    print(f"📁 Saved final_combined_occurrence.csv ({len(final_occ)} rows)")
else:
    print("⚠️ No data was found to concatenate.")