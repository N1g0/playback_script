import os
import pandas as pd
from datetime import datetime, timedelta, date
import re
import glob
from dateutil import parser

# schedule_dict
schedule_dict = {
    # Playback trials
    ('Playback', '1'): ('11.03', '12.03'),
    ('Playback', '2'): ('23.03', '24.03'),
    ('Playback', '3'): ('03.04', '04.04'),
    ('Playback', '4'): ('21.04', '22.04'),
    ('Playback', '5'): ('26.04', '27.04'),
    ('Playback', '6'): ('14.05', '15.05'),
    ('Playback', '7'): ('30.05', '31.05'),

    # Control (Crow) trials
    ('Crow', '1'): ('14.03', '15.03'),
    ('Crow', '2'): ('19.03', '20.03'),
    ('Crow', '3'): ('08.04', '09.04'),
    ('Crow', '4'): ('16.04', '17.04'),
    ('Crow', '5'): ('30.04', '01.05'),
    ('Crow', '6'): ('04.05', '05.05'),
    ('Crow', '7'): ('19.05', '20.05'),

    # Baseline
    ('Baseline', '1'): ('29.03',),
    ('Baseline', '2'): ('06.03',)
}

Cage_Compositions = {
    'A': {
        '1': ['TK', 'NR', 'NY', 'MN', 'LN'],
        '2': ['NY', 'ST', 'SR', 'TK', 'TN'],
        '3': ['TN', 'TK', 'SR', 'ST']
    },
    'B': {
        '1': ['NH', 'ST', 'MS'],
        '2': ['NR', 'NH', 'MN', 'LN'],
        '3': ['MS', 'NH', 'NY', 'MN']
    },
    'C': {
        '1': ['GG', 'SB', 'TN', 'MZ', 'SR'],
        '2': ['MS', 'SB', 'GG', 'MZ'],
        '3': ['GG', 'LN', 'MZ', 'NR', 'SB']
    }
}

Cage_Comp_Dates = {
    '1': ['11.03', '12.03', '19.03', '20.03', '08.04', '09.04', '21.04', '22.04'],
    '2': ['23.03', '24.03', '26.04', '27.04', '04.05', '05.05', '19.05', '20.05', '30.05', '31.05'],
    '3': ['03.04', '04.04', '16.04', '17.04', '30.04', '01.05', '14.05', '15.05'],
}

def compare_df(raw_df: pd.DataFrame, empty_df: pd.DataFrame,
               time_col_raw="1_TIme", time_col_empty="time", tolerance="2min"):
    """
    Align raw_df to empty_df by nearest time points, filling empty_df with raw_df data.

    Handles MultiIndex in empty_df and duplicate columns from merge_asof.
    """

    # --- Copy to avoid modifying original data ---
    raw_df = raw_df.copy()
    empty_df = empty_df.copy()

    # --- Handle MultiIndex in empty_df ---
    if time_col_empty not in empty_df.columns:
        if time_col_empty in empty_df.index.names:
            empty_df = empty_df.reset_index(level=time_col_empty)
        else:
            raise KeyError(f"'{time_col_empty}' not found in columns or index of empty_df")

    print("Raw columns:", raw_df.columns)
    print("Empty columns:", empty_df.columns)

    # --- Convert time columns to datetime ---
    raw_df[time_col_raw] = pd.to_datetime(raw_df[time_col_raw], format="%H:%M", errors='coerce')
    empty_df[time_col_empty] = pd.to_datetime(empty_df[time_col_empty], format="%H:%M", errors='coerce')

    # --- Sort by time ---
    raw_df = raw_df.sort_values(time_col_raw).reset_index(drop=True)
    empty_df = empty_df.sort_values(time_col_empty).reset_index(drop=True)

    # --- Merge using nearest time ---
    merged = pd.merge_asof(
        empty_df,
        raw_df,
        left_on=time_col_empty,
        right_on=time_col_raw,
        direction="nearest",
        tolerance=pd.Timedelta(tolerance)
    )

    # --- Handle duplicate columns from merge (suffixes _x/_y) ---
    for col in raw_df.columns:
        if col in empty_df.columns:
            col_x = f"{col}_x"
            col_y = f"{col}_y"
            if col_x in merged.columns and col_y in merged.columns:
                # Prefer raw_df (_y) value, fallback to empty_df (_x) if missing
                merged[col] = merged[col_y].combine_first(merged[col_x])
                merged = merged.drop(columns=[col_x, col_y])

    # --- Fill 'date' from 'created_at' if exists ---
    if "created_at" in merged.columns:
        merged["date"] = pd.to_datetime(merged["created_at"], errors='coerce').dt.date

    # --- Identify unused raw times ---
    if time_col_raw in merged.columns:
        used_times = merged[time_col_raw].dropna().unique()
        unused_df = raw_df[~raw_df[time_col_raw].isin(used_times)]
        if not unused_df.empty:
            print("⚠️ Unused time points from raw_df:")
            print(unused_df[["ec5_uuid", "created_at", time_col_raw]])

    return merged








def get_behavior_code_dict(df):
    pattern = re.compile(r'\d+_(\w{2})_Behavior')
    return {
        col: match.group(1)
        for col in df.columns
        if (match := pattern.match(col))
    }


def format_date(date_input):
    """
    Parses a date string and returns a dictionary with both:
    - 'euro_format': 'DD.MM.YYYY'
    - 'iso_z_format': 'YYYY-MM-DDTHH:MM:SS.000Z'
    Returns None if the input is invalid.
    """
    if pd.isna(date_input) or str(date_input).strip() == '':
        return None

    date_str = str(date_input).strip()

    try:
        # ISO input with time (e.g., 2025-03-11T00:00:00Z)
        if 'T' in date_str:
            dt = datetime.fromisoformat(date_str.replace('Z', ''))

        # Dot-separated (e.g., 04.04.2025)
        elif '.' in date_str:
            dt = datetime.strptime(date_str, '%d.%m.%Y')

        # All other formats (e.g., 04/04/2025 or 4-4-25)
        else:
            dt = parser.parse(date_str, dayfirst=True)

        return {
            'euro_format': dt.strftime('%d.%m.%Y'),
            'iso_z_format': dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        }

    except Exception as e:
        print(f"Invalid date input: {date_str} -> {e}")
        return None


def minutes_since_midnight(time_str):
    try:
        # Try parsing with seconds first
        time_obj = pd.to_datetime(time_str, format='%H:%M:%S', errors='coerce')
        if pd.isna(time_obj):
            # Fallback: try without seconds
            time_obj = pd.to_datetime(time_str, format='%H:%M', errors='coerce')
            if pd.isna(time_obj):
                return None
        return time_obj.hour * 60 + time_obj.minute
    except Exception:
        return None




"""
def make_fill_columns(schedule_dict, behave_dict, Cage_Compositions):
    def fill_columns(row):
        try:
            row['msm'] = minutes_since_midnight(row.get('1_TIme'))

            formatted_date = format_date(row.get('created_at'))
            if formatted_date:
                # Store the ISO format string date for downstream usage
                row['date'] = formatted_date['iso_z_format']
            else:
                row['date'] = None

            row['phase'] = get_phase_from_time(row.get('1_TIme'))

            condition, trial, first_day = get_condition_and_trial(row['date'], schedule_dict)
            row['condition'] = condition
            row['trial'] = trial
            row['first_day'] = first_day

            row['group'] = get_exact_matching_cage_phase(behave_dict, Cage_Compositions)
            if row['group'] is None:
                row['group'] = get_group_from_date(row['date'], Cage_Comp_Dates, behave_dict)
            row['block_ID'] = block_ID(row)
        except Exception as e:
            print(f"Error in fill_columns: {e}")
        return row

    return fill_columns
"""






def flatten_dict_of_lists(dol):
    # Step 1: Check if 'distance' key exists in the dictionary
    if 'distance' not in dol:
        print("Key 'distance' not found in the dictionary.")
        return []

    # Step 2: Get the values associated with 'distance' key
    distance_values = dol['distance']

    # Step 3: Ensure 'distance' contains a valid list (non-empty)
    if not isinstance(distance_values, list) or len(distance_values) == 0:
        print("Invalid or empty list for 'distance'.")
        return []

    # Step 4: Find the length of the shortest list to avoid IndexError (if there are other keys, just skip them)
    min_len = len(distance_values)

    # Step 5: Flatten into a list of row dictionaries (only using 'distance' values)
    combined_rows = [{'distance': distance_values[i]} for i in range(min_len)]

    return combined_rows


def remove_double_dyad_from_list(rows):
    seen_dyads = set()
    filtered = []

    pattern = re.compile(r'\*')

    for entry in rows:
        row = entry.get('distance')
        if not row:
            print("Skipping invalid row (missing 'distance'):", entry)
            continue

        ind1 = row.get('Ind1')
        ind2 = row.get('partner')
        ec5_uuid = row.get('ec5_uuid')
        group = row.get('group', '')
        msm = row.get('msm')

        if isinstance(group, str) and pattern.search(group):
            print(f"Skipping due to group pattern (* found): {group}")
            continue

        if not (isinstance(ind1, str) and isinstance(ind2, str) and isinstance(ec5_uuid, str)):
            print(f"Skipping due to invalid types: Ind1={ind1}, partner={ind2}, ec5_uuid={ec5_uuid}")
            continue

        if msm is None:
            print(f"Skipping due to missing msm for dyad {ind1}-{ind2} and UUID {ec5_uuid}")
            continue

        normalized_dyad = '-'.join(sorted([ind1, ind2]))
        key = (ec5_uuid, normalized_dyad, msm)

        if key in seen_dyads:
            print(f"Duplicate dyad found for {normalized_dyad} with msm {msm} in UUID {ec5_uuid} — skipping duplicate.")
            continue

        seen_dyads.add(key)
        row['dyad'] = normalized_dyad
        filtered.append(row)

    return filtered


def clean_dict_of_lists(dol):
    # Flatten the dict of lists into a single list of row dictionaries
    flattened_rows = flatten_dict_of_lists(dol)

    # Remove duplicate dyads
    cleaned_rows = remove_double_dyad_from_list(flattened_rows)

    return cleaned_rows


def reshape_row_to_multiple(row, beha_dict):

    behaviour_lists = {
        'behaviour': [],
        'all_occurence': [],
    }
    distance_list = []

    # Metadata
    metadata = {
        'ec5_uuid': row.get('ec5_uuid'),
        'condition': row.get('condition'),
        'date': row.get('date'),
        'msm': row.get('msm'),
        'group': row.get('group'),
        'trial': row.get('trial'),
        'phase': row.get('phase'),
        'block_ID': row.get('block_ID')
    }
    notes = row.get('notes')

    # Extract & sort relevant columns
    behav_cols = [col for col in row.index if re.match(r'^\d+_', col)]
    behav_cols_sorted = sorted(behav_cols, key=lambda i: int(i.split('_')[0]))
    behavior_row = row[behav_cols_sorted]

    int1_col = 'Ind1'
    partner_col = 'partner'
    dyad_col = 'dyad'
    distance_col = 'distance'
    behaviour_col = 'behaviour'
    qualifier_col = 'qualifier'
    eating_col = 'eating'
    playing_col = 'playing'
    resting_col = 'resting'
    self_direct_col = 'self_direct'
    grooming_col = 'grooming'
    sitting_col = 'sitting'
    moving_col = 'moving'
    aggression_col = 'aggression'
    notes_col = 'notes'
    for i in range(0, len(behavior_row), 4):
        block = behavior_row.iloc[i:i + 4]
        if block.empty or len(block) < 4:
            continue
        int1 = block.index[0].split('_')[1] if len(block.index[0].split('_')) > 1 else 'UNKNOWN'
        behaviour, contact, arr, three_met = block.iloc[0], block.iloc[1], block.iloc[2], block.iloc[3]
        behaviour = str(behaviour).strip() if pd.notna(behaviour) else None

        # Determine group target names
        targets = contact if pd.notna(contact) else arr

        # Define the default behavior dictionary with all zeros
        default_behaviour_flags = {
            eating_col: 0,
            playing_col: 0,
            moving_col: 0,
            resting_col: 0,
            sitting_col: 0,
            self_direct_col: 0,
            grooming_col: 0,
        }

        default_behav_alloc_flags = {
            playing_col: 0,
            aggression_col: 0
        }

        # Handle group/dyadic behaviors
        if behaviour in {'PL', 'Pl'}:
            behaviour_lists['all_occurence'].append({
                **metadata,
                int1_col: int1,
                behaviour_col: behaviour,
                qualifier_col: '',
                partner_col: targets,
                notes_col: notes,
                **{**default_behav_alloc_flags, playing_col: 1},
            })

        elif behaviour in {'AG', 'DS', 'AR'}:
            if len(behaviour) == 2:
                behaviour_lists['all_occurence'].append({
                    **metadata,
                    int1_col: int1,
                    behaviour_col: behaviour[0],
                    qualifier_col: behaviour[1],
                    partner_col: targets,
                    notes_col: notes,
                    **{**default_behav_alloc_flags, aggression_col: 1},
                })

        # Grooming behaviours (qualifier matters)
        elif behaviour in {'GG', 'GR', 'GM'}:
            if len(behaviour) == 2:
                main_behaviour = behaviour[0]  # 'G'
                qualifier = behaviour[1]
                behaviour_lists['behaviour'].append({
                    **metadata,
                    int1_col: int1,
                    behaviour_col: main_behaviour,
                    qualifier_col: qualifier,
                    partner_col: targets,
                    **{**default_behaviour_flags, grooming_col: 1},
                })

        # Solo behaviours
        else:
            behaviour_flags = default_behaviour_flags.copy()
            solo_behaviour = True

            if behaviour == 'E':
                behaviour_flags[eating_col] = 1
            elif behaviour == 'RS':
                if len(behaviour) == 2:
                    main_behaviour = behaviour[0]
                    qualifier = behaviour[1]
                    behaviour_flags[resting_col] = 1
                    behaviour_flags[sitting_col] = 1
                    behaviour_lists['behaviour'].append({
                        **metadata,
                        int1_col: int1,
                        behaviour_col: main_behaviour,
                        qualifier_col: qualifier,
                        partner_col: None,
                        **behaviour_flags,
                    })
                    continue  # skip default append

            elif behaviour == 'RL':
                if len(behaviour) == 2:
                    main_behaviour = behaviour[0]
                    qualifier = behaviour[1]
                    behaviour_flags[resting_col] = 1
                    behaviour_lists['behaviour'].append({
                        **metadata,
                        int1_col: int1,
                        behaviour_col: main_behaviour,
                        qualifier_col: qualifier,
                        partner_col: None,
                        **behaviour_flags,
                    })
                    continue

                behaviour_flags[resting_col] = 1
            elif behaviour == 'M':
                behaviour_flags[moving_col] = 1
            elif behaviour == 'SD':
                behaviour_flags[self_direct_col] = 1
            else:
                solo_behaviour = False  # Unknown or unhandled behavior

            if solo_behaviour:
                behaviour_lists['behaviour'].append({
                    **metadata,
                    int1_col: int1,
                    behaviour_col: behaviour,
                    qualifier_col: None,
                    partner_col: None,
                    **behaviour_flags,
                })

        # Handle distance logic
        if pd.isna(contact) and pd.isna(arr) and pd.isna(three_met):
            for other in beha_dict.values():
                if other != int1:
                    distance_list.append({
                        **metadata,
                        int1_col: int1,
                        partner_col: other,
                        dyad_col: f"{int1}-{other}",
                        distance_col: 4
                    })
        else:
            levels = [(contact, 1), (arr, 2), (three_met, 3)]
            for group_value, dist in levels:
                if pd.notna(group_value):
                    for ind2 in group_value.split(','):
                        distance_list.append({
                            **metadata,
                            int1_col: int1,
                            partner_col: ind2.strip(),
                            dyad_col: f"{int1}-{ind2.strip()}",
                            distance_col: dist
                        })

    return {
        'distance': distance_list,
        'behaviour': behaviour_lists['behaviour'],
        'occurrence': behaviour_lists['all_occurence']
    }


def reshape_and_combine_all(df, beh_dict):
    combined_dict_of_lists = {}
    #print('beh_dict: ' + str(beh_dict))
    for _, row in df.iterrows():
        row_dict_of_lists = reshape_row_to_multiple(row, beh_dict)
        for key, value_list in row_dict_of_lists.items():
            if key not in combined_dict_of_lists:
                combined_dict_of_lists[key] = []
            combined_dict_of_lists[key].extend(value_list)

    # Clean "distance" field if exists
    if 'distance' in combined_dict_of_lists:
        combined_dict_of_lists['distance'] = clean_dict_of_lists(combined_dict_of_lists)

    return combined_dict_of_lists


def empty_dataframe(schedule_dict):

    # Define the columns
    columns = ['ec5_uuid', 'condition', 'date', 'msm']

    # Define time ranges in 2-minute intervals
    def generate_timepoints():
        times = []
        # Morning session
        t = datetime.strptime("09:30", "%H:%M")
        end = datetime.strptime("11:45", "%H:%M")
        while t <= end:
            times.append(t.strftime("%H:%M"))
            t += timedelta(minutes=2)

        # Afternoon session
        t = datetime.strptime("13:15", "%H:%M")
        end = datetime.strptime("14:45", "%H:%M")
        while t <= end:
            times.append(t.strftime("%H:%M"))
            t += timedelta(minutes=2)

        return times

    timepoints = generate_timepoints()

    # Build a MultiIndex from dyads and timepoints
    index = pd.MultiIndex.from_product(
        [schedule_dict.keys(), timepoints],
        names=["dyad", "time"]
    )

    # Create the empty DataFrame
    empty_df = pd.DataFrame(index=index, columns=columns)
    # Optional: reset index if you don't want MultiIndex
    # df = df.reset_index()
    print(empty_df)

    return empty_df


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

    # Check if all timeslots are filled
    checked_times_df = compare_df(df_raw, empty_df)

    # Step 1: Reshape data
    #rs_df, _ = reshape_behavior_data(df_raw, schedule_dict, Cage_Compositions)

    # Step 2: Rebuild fresh behavior dict from reshaped data
    #behave_dict = get_behavior_code_dict(rs_df)

    # Pick only needed columns (example based on your original code)
    #target_cols = [col for col in rs_df.columns if any(k in col for k in ['Contact', 'AR', '3M'])]
    #selected_columns = ['ec5_uuid', 'condition', 'date', 'msm', 'group', 'trial', 'phase', 'block_ID'] + target_cols + list(behave_dict.keys())
    #df_subset = rs_df[selected_columns]
    #print(df_subset)

    # Combine data for this file
    #combined = reshape_and_combine_all(df_subset, behave_dict)

    # Aggregate combined data globally
    #for key, data_list in combined.items():
    #    if key not in all_data:
    #        all_data[key] = []
    #    all_data[key].extend(data_list)

# After processing all files, save combined CSV files
for key, data_list in all_data.items():
    df = pd.DataFrame(data_list)
    output_file = os.path.join(output_dir, f"all_{key}.csv")
    print(f"Saving combined data to: {output_file}")
    df.to_csv(output_file, index=False)

print("All files processed and combined CSVs saved.")
