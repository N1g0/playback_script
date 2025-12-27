import os
import re
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

pd.set_option('display.max_columns', None)   # show all columns
pd.set_option('display.width', None)         # don't wrap columns
pd.set_option('display.max_colwidth', None)  # show full column content
# Dictionary with Background data

# schedule_dict
schedule_dict = {
    # Playback trials
    ('Playback', '1'): ('11.03.2025', '12.03.2025'),
    ('Playback', '2'): ('23.03.2025', '24.03.2025'),
    ('Playback', '3'): ('03.04.2025', '04.04.2025'),
    ('Playback', '4'): ('21.04.2025', '22.04.2025'),
    ('Playback', '5'): ('26.04.2025', '27.04.2025'),
    ('Playback', '6'): ('14.05.2025', '15.05.2025'),
    ('Playback', '7'): ('30.05.2025', '31.05.2025'),

    # Control (Crow) trials
    ('Crow', '1'): ('14.03.2025', '15.03.2025'),
    ('Crow', '2'): ('19.03.2025', '20.03.2025'),
    ('Crow', '3'): ('08.04.2025', '09.04.2025'),
    ('Crow', '4'): ('16.04.2025', '17.04.2025'),
    ('Crow', '5'): ('30.04.2025', '01.05.2025'),
    ('Crow', '6'): ('04.05.2025', '05.05.2025'),
    ('Crow', '7'): ('19.05.2025', '20.05.2025'),

    # Baseline
    ('Baseline', '1'): ('29.03.2025',),
    ('Baseline', '2'): ('06.03.2025',)
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

########################################################################################################################
########################################################################################################################

# loop through raw data time:
# calculate difference in timestep and see if it is 2 min or less/more
# show in Histogram good/bad data and where it occurs
# Glättungsalgorythmus für die schlechten datenpunkten
# füllen in soll tabelle_empty


def is_excluded(msm):
    """Checks if a given msm value falls within the excluded break periods."""
    # 10:30 - 10:45 (630 to 645 msm)
    if 630 <= msm <= 645:
        return True

    # 11:45 - 13:15 (705 to 795 msm)
    if 705 <= msm <= 795:
        return True

    # Later than 14:46 (886+ msm)
    if msm > 886:
        return True

    return False


print("\n--- FILTERED MISSING TIMESLOTS ---")
filtered_gaps = []


def analyze_time_gaps(
        df,
        start_time: float = 570,
        end_time: float = 885
                      ):
    # 1. Preparation: Ensure we are working with standard float numpy arrays
    # This prevents broadcasting issues with Pandas Nullable Int64 types
    df = df.sort_values("msm").reset_index(drop=True)
    observed = df["msm"].to_numpy(dtype=float)

    # 2. Generate Ideal Grid (2-minute intervals)
    #start_time = np.floor(observed.min())
    #end_time = np.ceil(observed.max())
    ideal_slots = np.arange(start_time, end_time + 2, 2)

    # 3. Vectorized Cost Matrix (Broadcasting)
    # observed (N, 1) vs ideal_slots (1, M)
    diff_matrix = np.abs(observed[:, np.newaxis] - ideal_slots)
    BIG = 1e6
    cost_matrix = np.where(diff_matrix <= 1.0, diff_matrix, BIG)

    # 4. Global Optimization (Hungarian Algorithm)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 5. Identifying valid matches
    actual_costs = cost_matrix[row_ind, col_ind]
    valid_mask = actual_costs < BIG

    matched_row_indices = row_ind[valid_mask]  # Indices of the original rows
    matched_slot_indices = col_ind[valid_mask]  # Indices of the ideal slots
    values_to_assign = ideal_slots[matched_slot_indices]

    # 6. Safe Assignment (The Fix)
    # Pre-allocate a full column of NaNs using Numpy
    corrected_values = np.full(len(df), np.nan)

    # Use standard integer indexing to place values exactly where they belong
    if len(matched_row_indices) > 0:
        corrected_values[matched_row_indices] = values_to_assign

    # Assign the completed array back to the DataFrame in one go
    df["corrected_msm"] = corrected_values

    # 7. Identify Missing Slots (Holes in the timeline)
    all_slot_indices = np.arange(len(ideal_slots))
    missing_indices = np.setdiff1d(all_slot_indices, matched_slot_indices)
    missing_slots = ideal_slots[missing_indices]

    print(f"File processed: {len(df)} rows.")
    print(f"  - Matches found: {len(matched_row_indices)}")

    return df, missing_slots


def adjust_msm_in_raw_empty(
    raw_df: pd.DataFrame,
    empty_df: pd.DataFrame,
    file_type: str,
    output_dir: str,
    time_col_raw: str = "1_TIme",
    time_col_empty: str = "time",
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # --- Copy inputs to avoid mutation ---
    raw_df = raw_df.copy()
    empty_df = empty_df.copy()
    raw_df = raw_df.rename(columns={"created_at": "date"})

    date_col = "date"

    # --- Parse dates ---
    raw_df[date_col] = pd.to_datetime(
        raw_df[date_col], format="%Y-%m-%dT%H:%M:%S.%fZ", errors="coerce"
    )
    empty_df[date_col] = pd.to_datetime(
        empty_df[date_col], format="%d.%m.%Y", errors="coerce"
    )

    # --- Ensure template time column exists ---
    if time_col_empty not in empty_df.columns:
        if time_col_empty in empty_df.index.names:
            empty_df = empty_df.reset_index(level=time_col_empty)
        else:
            raise KeyError(f"'{time_col_empty}' not found in empty_df")

    # --- Determine time format based on file type ---
    if file_type == 'A':
        raw_time_format = "%H:%M"
    elif file_type in ('B', 'C'):
        raw_time_format = "%H:%M:%S"
    else:
        raw_time_format = "%H:%M:%S"
        print(f"Warning: Unknown file type '{file_type}', defaulting to %H:%M:%S.")

    # --- Parse times ---
    raw_df[time_col_raw] = pd.to_datetime(
        raw_df[time_col_raw], format=raw_time_format, errors="coerce"
    )
    empty_df[time_col_empty] = pd.to_datetime(
        empty_df[time_col_empty], format="%H:%M", errors="coerce"
    )

    # --- Minutes-since-midnight helper ---
    def msm(df, col):
        td = df[col] - df[col].dt.normalize()
        return (td.dt.total_seconds() / 60).astype("Int64")

    raw_df["msm"] = msm(raw_df, time_col_raw)
    empty_df["msm"] = msm(empty_df, time_col_empty)

    # --- Drop invalid rows ---
    valid_empty_df = empty_df.dropna(subset=["msm", date_col]).copy()
    invalid_empty_df = empty_df[empty_df["msm"].isna() | empty_df[date_col].isna()].copy()
    raw_df = raw_df.dropna(subset=["msm", date_col]).copy()

    # Normalize dates (ensure same day alignment)
    raw_df[date_col] = raw_df[date_col].dt.normalize()
    valid_empty_df[date_col] = valid_empty_df[date_col].dt.normalize()

    # --- Sort ---
    raw_df = raw_df.sort_values(["date", "msm"]).reset_index(drop=True)
    valid_empty_df = valid_empty_df.sort_values(["date", "msm"]).reset_index(drop=True)

    #print('raw_df: ', raw_df)
    print('valid_empty_df: ', valid_empty_df)

    # Ensure date column is date-only
    raw_df['date'] = pd.to_datetime(raw_df['date']).dt.date

    os.makedirs(output_dir, exist_ok=True)

    for date, time_sorted_df in raw_df.groupby('date'):
        adjusted_df, gaps = analyze_time_gaps(time_sorted_df)
        for g in gaps:
            if not is_excluded(g):
                h, m = divmod(int(g), 60)
                print(f"MISSING DATA AT: {h:02d}:{m:02d}")
                filtered_gaps.append(g)

        if not filtered_gaps:
            print("No unexpected gaps found (all gaps occurred during excluded periods).")

        suffix = f"_{file_type}_{date}"

        output_file_path: str = os.path.join(
            output_dir,
            f"sorted_data{suffix}.csv"
        )

        # Save merged DataFrame
        adjusted_df.to_csv(
            output_file_path,
            index=False,
            sep=';'
        )
        print('adjusted_df: ', adjusted_df)

    return raw_df, valid_empty_df


def empty_dataframe(schedule_dict):

    # Define the columns
    columns = ['condition', 'date', 'msm']

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

    # Reset index for easier manipulation
    empty_df = empty_df.reset_index()

    # Extract condition (Playback, Crow, Baseline) and ID
    empty_df['condition'] = empty_df['dyad'].apply(lambda x: x[0])
    empty_df['id'] = empty_df['dyad'].apply(lambda x: x[1])

    # Map dates from schedule_dict
    def get_date_mapping(dyad):
        dates = schedule_dict[dyad]
        if len(dates) == 2:
            return {"morning": dates[0], "afternoon": dates[1]}
        else:
            return {"morning": dates[0], "afternoon": dates[0]}

    # Assign correct date depending on time (morning vs afternoon)
    empty_df['date'] = empty_df.apply(
        lambda row: get_date_mapping(row['dyad'])['morning']
        if row['time'] < '12:00'
        else get_date_mapping(row['dyad'])['afternoon'],
        axis=1
    )

    # --- Add "minutes since midnight" (msm) ---
    def time_to_msm(t_str):
        t = datetime.strptime(t_str, "%H:%M")
        return t.hour * 60 + t.minute

    empty_df['msm'] = empty_df['time'].apply(time_to_msm)

    # Drop helper columns
    empty_df = empty_df.drop(columns=['dyad', 'id'])

    return empty_df

#######################################################################################################################
# OLD CODE
#######################################################################################################################


from dateutil import parser


def detect_ind(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect dynamic individuals, merge columns with important information for later processing and
    return a DataFrame with columns ready for merging.
    """
    raw_df: pd.DataFrame = raw_df.copy()

    # --- Detect dynamic individuals ---
    behavior_pattern: re.Pattern = re.compile(r"^\d+_([A-Za-z]+)_Behavior$", re.IGNORECASE)
    column_groups: Dict[str, List[str]] = {}

    for col in raw_df.columns:
        match: re.Match | None = behavior_pattern.match(col)
        if match:
            identifier: str = match.group(1)
            related_cols: List[str] = [
                c for c in raw_df.columns
                if re.match(fr"^\d+_{identifier}(_Behavior|_Contact|_AR|_3M|_Notes)?$", c, re.IGNORECASE)
            ]
            column_groups[identifier] = related_cols

    # --- Insert label columns next to Behavior columns ---
    for identifier, cols in column_groups.items():
        behavior_col: str | None = next((c for c in cols if c.lower().endswith("_behavior")), None)
        if behavior_col:
            label_col: str = identifier
            if label_col not in raw_df.columns:
                raw_df.insert(raw_df.columns.get_loc(behavior_col), label_col, identifier)

    # --- Columns to keep for final merge ---
    base_cols: List[str] = ["ec5_uuid", "1_TIme", "highlight", "msm", "merge_date"]
    dynamic_cols: List[str] = []
    for identifier, cols in column_groups.items():
        dynamic_cols.append(identifier)  # Label column
        dynamic_cols.extend(cols)  # Behavior, Contact, AR, 3M, Notes

    # Remove duplicates while keeping order
    seen = set()
    cols_to_keep: List[str] = []
    for c in base_cols + dynamic_cols:
        if c not in seen and c in raw_df.columns:
            seen.add(c)
            cols_to_keep.append(c)

    # --- Return subset DataFrame ready to merge ---
    final_df: pd.DataFrame = raw_df[cols_to_keep].copy()
    return final_df


def get_exact_matching_cage_phase(behave_dict, cage_compositions):
    behavior_values = set(behave_dict.values())

    for cage, phases in cage_compositions.items():
        for phase, individuals in phases.items():
            if behavior_values.issubset(set(individuals)):
                return f"{cage}{phase}"

    return None


def get_condition_and_trial(date_str, schedule_dict):
    if pd.isna(date_str) or date_str == '':
        return None, None, None
    try:
        day_month = datetime.strptime(date_str, '%d-%m-%Y').strftime('%d.%m')
    except Exception as e:
        print(f"Invalid date format: {date_str} -> {e}")
        return None, None, None

    for (condition, trial), dates in schedule_dict.items():
        if day_month in dates:
            first_day = (day_month == dates[0])
            return condition, trial, first_day

    return None, None, None


def get_group_from_date(date_str, cage_dates_dict, behave_dict):
    # Convert input date to "DD.MM" format, allowing flexible input
    try:
        date_obj = parser.parse(date_str, dayfirst=True)
        short_date = date_obj.strftime("%d.%m")
    except (ValueError, TypeError):
        return ''  # Invalid date format

    # First, find the group the date belongs to
    matched_group = None
    for group, dates in cage_dates_dict.items():
        if short_date in dates:
            matched_group = group
            break

    # If the date is not found in any group, return empty
    if not matched_group:
        return ''

    # Define sets for each prefix group
    group_A_ids = {'TN', 'TK', 'LN', 'NR', 'NY', 'ST', 'MN', 'SR'}
    group_B_ids = {'MN', 'MS', 'NH', 'LN', 'NR', 'NY', 'ST'}
    group_C_ids = {'MZ', 'SB', 'GG', 'LN', 'NR', 'MS'}

    values = set(behave_dict.values())

    # Determine prefix based on behavior codes
    if 'GG' in values or 'SB' in values:
        prefix = 'C'
    elif values & group_A_ids:
        prefix = 'A'
    elif values & group_B_ids:
        prefix = 'B'
    elif values & group_C_ids:
        prefix = 'C'
    else:
        prefix = 'No data for that day'

    group_str = f"{prefix}{matched_group}" if prefix else matched_group
    return f"*{group_str}"


def get_phase_from_time(time_value):
    if pd.isna(time_value) or time_value == '':
        return None
    try:
        time_str = str(time_value).strip()
        # Try with seconds first
        try:
            time_obj = datetime.strptime(time_str, '%H:%M:%S').time()
        except ValueError:
            # Fallback to no seconds
            time_obj = datetime.strptime(time_str, '%H:%M').time()

        minutes = time_obj.hour * 60 + time_obj.minute

        if 570 <= minutes <= 630:  # 09:30 to 10:30
            return 'morn1'
        elif 645 <= minutes <= 705:  # 10:45 to 11:45
            return 'morn2'
        elif minutes >= 720:  # After 12:00
            return 'enrich'
        else:
            return 'extra'

    except Exception as e:
        print(f"Failed to parse time value '{time_value}': {e}")
        return None


def get_group_from_date_column(dates_series, cage_dates_dict, behave_dict):
    # Step 1: Build a reverse lookup from DD.MM to group
    date_to_group = {}
    for group, dates in cage_dates_dict.items():
        for d in dates:
            date_to_group[d] = group

    # Step 2: Determine shared prefix once (not per row)
    values = set(behave_dict.values())

    if 'GG' in values or 'SB' in values:
        prefix = 'C'
    elif values & {'TN', 'TK', 'LN', 'NR', 'NY', 'ST', 'MN', 'SR'}:
        prefix = 'A'
    elif values & {'MN', 'MS', 'NH', 'LN', 'NR', 'NY', 'ST'}:
        prefix = 'B'
    elif values & {'MZ', 'SB', 'GG', 'LN', 'NR', 'MS'}:
        prefix = 'C'
    else:
        prefix = None

    # Step 3: Vectorized transformation
    def map_func(date_str):
        try:
            date_obj = parser.parse(date_str, dayfirst=True)
            short_date = date_obj.strftime("%d.%m")
        except (ValueError, TypeError):
            return ''

        matched_group = date_to_group.get(short_date)
        if not matched_group:
            return ''

        group_str = f"{prefix}{matched_group}" if prefix else matched_group
        return f"*{group_str}"

    return dates_series.apply(map_func)


def generate_block_ID(df):
    # Start with lowercase phase (empty string where NaN)
    ID = df['phase'].fillna('').str.lower()
    ID = ID.where(ID.isin(['morn1', 'morn2', 'enrich']), '')

    # Append group
    ID += '_' + df['group'].fillna('')

    # Append trial
    ID += df['trial'].apply(lambda x: f"_trial{int(x)}" if pd.notnull(x) else '')

    # Append first_day
    ID += df['first_day'].apply(lambda x: '_first' if x is True else ('_sec' if x is False else ''))

    # Append condition
    condition_map = {'playback': '_play', 'crow': '_cont', 'baseline': '_base'}
    ID += df['condition'].fillna('').str.lower().map(condition_map).fillna('')

    return ID


def reshape_behavior_data(df, schedule_dict, Cage_Compositions):
    working_df = df.copy()
    # TODO: properly implementing the behave_dict
    behave_dict = df['ID']

    # Ensure required columns exist
    for col in ['msm', 'date', 'phase', 'group', 'trial', 'first_day', 'block_ID']:
        if col not in working_df.columns:
            working_df[col] = None
    """
    # Vectorized: minutes since midnight
    if '1_TIme' in working_df.columns:
        working_df['msm'] = working_df['1_TIme'].apply(minutes_since_midnight)

    # Vectorized: format date
    if 'created_at' in working_df.columns:
        working_df['date'] = working_df['created_at'].apply(
            lambda x: format_date(x)['iso_z_format'] if format_date(x) else None
        )
    """

    # Vectorized: get phase
    if '1_TIme' in working_df.columns:
        working_df['phase'] = working_df['1_TIme'].apply(get_phase_from_time)

    # Vectorized: condition, trial, first_day
    def get_schedule_info(date):
        condition, trial, first_day = get_condition_and_trial(date, schedule_dict)
        return pd.Series([condition, trial, first_day])

    working_df[['condition', 'trial', 'first_day']] = working_df['date'].apply(get_schedule_info)

    # Vectorized group detection fallback (still partially row-wise due to logic complexity)
    def get_group_vectorized(row):
        group = get_exact_matching_cage_phase(behave_dict, Cage_Compositions)
        if group is None:
            return get_group_from_date(row['date'], Cage_Comp_Dates, behave_dict)

    working_df['group'] = get_group_from_date_column(
        working_df['date'],
        Cage_Comp_Dates,
        behave_dict
    )

    # Vectorized block ID assignment
    working_df['block_ID'] = generate_block_ID(working_df)

    return working_df, behave_dict