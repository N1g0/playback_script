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

# Global lists to store all data across all file types and chunks
all_behave_data: list = []
all_dist_data: list = []
all_occ_data: list = []
all_sanity: list = []

# --------------------------------------------------------------------------
#                          PHASE 1: Read and Store
# --------------------------------------------------------------------------

for input_file_path in data_files:
    print(f"Processing: {input_file_path}")

    # --- Process All_Occurence files ---
    if "all_occurence" in input_file_path.lower():
        df_occ: pd.DataFrame = Methods.read_data(input_file_path)
        all_raw_dfs, file_type = Methods.file_name(input_file_path, df_occ, all_raw_dfs)
        occ_df = Methods.process_all_occurrence(df_occ, file_type)

        print(f"✅ Processed {len(occ_df)} occurrences from Cage {file_type}")

        all_occ_data.append(occ_df)
        # Optional: Save to a file immediately
        occ_df.to_csv(f"{output_dir}/processed_occurrence_cage_{file_type}.csv", index=False)
        continue

    df_raw: pd.DataFrame = Methods.read_data(input_file_path)
    all_raw_dfs, file_type = Methods.file_name(input_file_path, df_raw, all_raw_dfs)

# --------------------------------------------------------------------------
#                     PHASE 2: Process Each Raw DataFrame
# --------------------------------------------------------------------------
#TODO:
# Sanitycheck: How many points are in every msm over all three cages and days (Überarbeiten und testen!!!)
# missing days used days for analysis -> Tabelle erstellen die die Aussortierten Tage / Zeiten abspeichert
# Intermediate step, with all datapoints (times) and the final process data with the fixed timewindow (apply_schedule_and_phase) (gefixt??)
# UNKNOWN checken (gefixt?) und phase überprüfen wann es nicht gecallt wird?
# DOCUMENTATION überarbeiten!

# Define subfolders for organization
steps_dir = os.path.join(output_dir, "intermediate_steps")
os.makedirs(steps_dir, exist_ok=True)
#print('all_raw_dfs: \n', all_raw_dfs)

for file_type, list_of_raw_dfs in all_raw_dfs.items():
    print(f"\nProcessing files for Type {file_type}...")
    #print('file_type:', file_type)
    #print('list_of_raw_dfs: \n', list_of_raw_dfs)
    for idx, raw_df in enumerate(list_of_raw_dfs):
        #print('idx: ', idx)
        #print('raw_df inner loop: \n', raw_df)
        try:
            # Time Adjustment
            time_sorted_df_raw = Methods.adjust_msm_in_raw_empty(
                raw_df=raw_df,
                file_type=file_type,
                output_dir=steps_dir,
                time_col_raw="1_TIme",
            )
            #print('time_sorted_df_raw: \n', time_sorted_df_raw)
            # Save
            time_sorted_df_raw.to_csv(os.path.join(steps_dir, f"type{file_type}_file{idx}_step1_time.csv"), index=False)

            #print('time_sorted_df_raw: \n', time_sorted_df_raw)

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


df_beh_missing, df_dist_missing, df_comparison = Methods.perform_sanity_check(
    all_behave_data,
    all_dist_data
)

# Export results to CSV files
df_beh_missing.to_csv(
    os.path.join(output_dir, 'behaviour_missing_duplicates.csv'), index=False
)
df_dist_missing.to_csv(
    os.path.join(output_dir, 'distance_missing_duplicates.csv'), index=False
)
df_comparison.to_csv(
    os.path.join(output_dir, 'missing_slots_comparison.csv'), index=False
)

print(f"\n📁 All reports successfully written to: '{output_dir}/'")


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
    all_occ = pd.concat(all_occ_data, ignore_index=True)
    all_occ.to_csv(os.path.join(steps_dir, "combined_all_occurrences.csv"), index=False)
    print(f"📁 Saved combined_occurrence.csv ({len(all_occ)} rows)")
    #TODO: compress_all_occ logic überprüfen!!!
    compressed_df: pd.DataFrame = Methods.compress_all_occ(all_occ)

    if not compressed_df.empty:
        compressed_df.to_csv(os.path.join(output_dir, "final_combined_all_occurrences.csv"), index=False)
        print(f"📁 Saved final_combined_occurrence.csv ({len(compressed_df)} rows)")
    else:
        print("⚠️ Compression resulted in an empty DataFrame.")

else:
    print("⚠️ No valid DataFrames found to concatenate.")
