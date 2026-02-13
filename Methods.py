import os
import re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Any

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
# Adjusting Timestamps and finding data gaps
########################################################################################################################


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


filtered_gaps = []


def time_to_msm(t_str: str) -> int:
    """Converts 'HH:MM' string to integer minutes since midnight."""
    t = datetime.strptime(t_str, "%H:%M")
    return t.hour * 60 + t.minute


def calculate_df_msm(df: pd.DataFrame, col: str) -> pd.Series:
    """Vectorized calculation of msm for a datetime column."""
    # Ensure the column is actually datetime objects
    dt_col = pd.to_datetime(df[col])
    td = dt_col - dt_col.dt.normalize()
    return (td.dt.total_seconds() / 60).astype("Int64")


def analyze_time_gaps(
    df: pd.DataFrame,
    start_time: float = 570.0,
    end_time: float = 885.0
) -> Tuple[pd.DataFrame, np.ndarray]:

    # 1. Preparation: Ensure we are working with standard float numpy arrays
    df: pd.DataFrame = df.sort_values("msm").reset_index(drop=True)
    observed: np.ndarray = df["msm"].to_numpy(dtype=float)

    # 2. Generate Ideal Grid (2-minute intervals)
    ideal_slots = np.array([time_to_msm(t) for t in generate_timepoints()])
    #print('ideal_slots:', ideal_slots)

    #ideal_slots_2: np.ndarray = np.arange(start_time, end_time + 2, 2)
    #print('ideal_slots_2:', ideal_slots_2)

    # 3. Vectorized Cost Matrix (Broadcasting)
    diff_matrix: np.ndarray = np.abs(observed[:, np.newaxis] - ideal_slots)
    BIG: float = 1e6
    cost_matrix: np.ndarray = np.where(diff_matrix <= 1.0, diff_matrix, BIG)

    # 4. Global Optimization (Hungarian Algorithm)
    row_ind: np.ndarray
    col_ind: np.ndarray
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 5. Identifying valid matches
    actual_costs: np.ndarray = cost_matrix[row_ind, col_ind]
    valid_mask: np.ndarray = actual_costs < BIG

    matched_row_indices: np.ndarray = row_ind[valid_mask]
    matched_slot_indices: np.ndarray = col_ind[valid_mask]
    values_to_assign: np.ndarray = ideal_slots[matched_slot_indices]

    # 6. Safe Assignment
    corrected_values: np.ndarray = np.full(len(df), np.nan)

    if len(matched_row_indices) > 0:
        corrected_values[matched_row_indices] = values_to_assign

    df["corrected_msm"] = corrected_values

    # 7. Identify Missing Slots (Holes in the timeline)
    all_slot_indices: np.ndarray = np.arange(len(ideal_slots))
    missing_indices: np.ndarray = np.setdiff1d(all_slot_indices, matched_slot_indices)
    missing_slots: np.ndarray = ideal_slots[missing_indices]

    print(f"File processed: {len(df)} rows.")
    print(f"  - Matches found: {len(matched_row_indices)}")

    return df, missing_slots


def save_missing_data(all_missing, output_dir, file_type):
    """
    all_missing: list of dicts
    output_dir: folder to save CSV
    """

    if not all_missing:
        print("No missing data to save.")
        return

    df = pd.DataFrame(all_missing)

    filename = f"missing_data_{file_type}.csv"

    output_path = os.path.join(output_dir, filename)

    df.to_csv(output_path, index=False, sep=";")

    print(f"Missing data saved to: {output_path}")


def adjust_msm_in_raw_empty(
    raw_df: pd.DataFrame,
    file_type: str,
    output_dir: str,
    time_col_raw: str = "1_TIme",
    time_col_empty: str = "time",
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # --- Copy inputs to avoid mutation ---
    raw_df = raw_df.copy()
    empty_df = empty_dataframe(schedule_dict)
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

    raw_df["msm"] = calculate_df_msm(raw_df, time_col_raw)
    empty_df["msm"] = calculate_df_msm(empty_df, time_col_empty)

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
    #print('valid_empty_df: ', valid_empty_df)

    # Ensure date column is date-only
    raw_df['date'] = pd.to_datetime(raw_df['date']).dt.date

    os.makedirs(output_dir, exist_ok=True)

    all_missing = []
    for date, time_sorted_df in raw_df.groupby('date'):
        adjusted_df, gaps = analyze_time_gaps(time_sorted_df)
        for g in gaps:
            if not is_excluded(g):
                h, m = divmod(int(g), 60)

                print(f"MISSING DATA in {file_type} / {date.strftime('%Y-%m-%d')} AT: {h:02d}:{m:02d}")

                all_missing.append({
                    "file_type": file_type,
                    "date": date.strftime("%Y-%m-%d"),
                    "time": f"{h:02d}:{m:02d}",
                    "gap_minutes": g
                })

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
        #print('adjusted_df: ', adjusted_df)

    save_missing_data(all_missing, output_dir, file_type)

    return raw_df, valid_empty_df


def generate_timepoints() -> List[str]:
    times: List[str] = []

    # Morning session
    t: datetime = datetime.strptime("09:30", "%H:%M")
    end: datetime = datetime.strptime("10:30", "%H:%M")
    while t <= end:
        times.append(t.strftime("%H:%M"))
        t += timedelta(minutes=2)

    # Playback break
    t: datetime = datetime.strptime("10:45", "%H:%M")
    end: datetime = datetime.strptime("11:45", "%H:%M")
    while t <= end:
        times.append(t.strftime("%H:%M"))
        t += timedelta(minutes=2)

    # Afternoon session
    t: datetime = datetime.strptime("13:15", "%H:%M")
    end: datetime = datetime.strptime("14:45", "%H:%M")
    while t <= end:
        times.append(t.strftime("%H:%M"))
        t += timedelta(minutes=2)

    return times


def empty_dataframe(
    schedule_dict: Dict[Tuple[Any, Any], List[str]]
) -> pd.DataFrame:

    # Define the columns
    columns: List[str] = ['condition', 'date', 'msm']

    # Define time ranges in 2-minute intervals

    timepoints: List[str] = generate_timepoints()

    # Build a MultiIndex from dyads and timepoints
    index: pd.MultiIndex = pd.MultiIndex.from_product(
        [schedule_dict.keys(), timepoints],
        names=["dyad", "time"]
    )

    # Create the empty DataFrame
    empty_df: pd.DataFrame = pd.DataFrame(index=index, columns=columns)

    # Reset index for easier manipulation
    empty_df: pd.DataFrame = empty_df.reset_index()

    # Extract condition (Playback, Crow, Baseline) and ID
    empty_df['condition']: pd.Series = empty_df['dyad'].apply(lambda x: x[0])
    empty_df['id']: pd.Series = empty_df['dyad'].apply(lambda x: x[1])

    # Map dates from schedule_dict
    def get_date_mapping(
        dyad: Tuple[Any, Any]
    ) -> Dict[str, str]:
        dates: List[str] = schedule_dict[dyad]
        if len(dates) == 2:
            return {"morning": dates[0], "afternoon": dates[1]}
        else:
            return {"morning": dates[0], "afternoon": dates[0]}

    # Assign correct date depending on time (morning vs afternoon)
    empty_df['date']: pd.Series = empty_df.apply(
        lambda row: get_date_mapping(row['dyad'])['morning']
        if row['time'] < '12:00'
        else get_date_mapping(row['dyad'])['afternoon'],
        axis=1
    )

    empty_df['msm']: pd.Series = empty_df['time'].apply(time_to_msm)

    # Drop helper columns
    empty_df: pd.DataFrame = empty_df.drop(columns=['dyad', 'id'])

    return empty_df


#######################################################################################################################
# Adding and Adjusting Columns
#######################################################################################################################


from dateutil import parser


def detect_ind(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # Base columns
    base_cols = ["ec5_uuid", "date", "1_TIme", "msm", "corrected_msm"]
    base_cols = [c for c in base_cols if c in df.columns]

    # Find behavior columns like: 4_MZ_Behavior
    pattern = re.compile(r"^(\d+)_([A-Za-z]+)_Behavior$", re.IGNORECASE)

    matches = []
    for col in df.columns:
        m = pattern.match(col)
        if m:
            order = int(m.group(1))
            ident = m.group(2)
            matches.append((order, ident, col))

    # Sort by numeric prefix
    matches.sort(key=lambda x: x[0])

    out_cols = base_cols.copy()

    # Build Ind_1 ... Ind_N
    for idx, (_, ident, behavior_col) in enumerate(matches, start=1):
        ind_col = f"Ind_{idx}"
        beh_col = f"Ind_{idx}_Behavior"

        # Insert label column
        df[ind_col] = ident

        # Rename behavior column
        df.rename(columns={behavior_col: beh_col}, inplace=True)

        out_cols.extend([ind_col, beh_col])

    return df[out_cols].copy()


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


# not used
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


def get_group_vectorized(dates_series, cage_dates_dict, behave_dict):
    # --- STEP 1: PRE-CALCULATE PREFIX (Do this once, not per row) ---
    values = set(behave_dict.values())

    # Define your logic sets
    group_A_ids = {'TN', 'TK', 'LN', 'NR', 'NY', 'ST', 'MN', 'SR'}
    group_B_ids = {'MN', 'MS', 'NH', 'LN', 'NR', 'NY', 'ST'}
    group_C_ids = {'MZ', 'SB', 'GG', 'LN', 'NR', 'MS'}

    if 'GG' in values or 'SB' in values:
        prefix = 'C'
    elif values & group_A_ids:
        prefix = 'A'
    elif values & group_B_ids:
        prefix = 'B'
    elif values & group_C_ids:
        prefix = 'C'
    else:
        prefix = ''  # Or 'No data' if you prefer

    # --- STEP 2: BUILD REVERSE LOOKUP DICT ---
    # Turns {Group: [Dates]} into {Date: Group} for instant O(1) access
    date_to_group = {d: group for group, dates in cage_dates_dict.items() for d in dates}

    # --- STEP 3: VECTORIZED DATA PROCESSING ---
    # 1. Convert all dates at once (C-speed)
    temp_dates = pd.to_datetime(dates_series, dayfirst=True, errors='coerce')

    # 2. Format to "DD.MM" string for all rows at once
    short_dates = temp_dates.dt.strftime("%d.%m")

    # 3. Map the groups (Instant lookup)
    matched_groups = short_dates.map(date_to_group)

    # 4. Construct final string using vectorized addition
    # Only act on rows where a group was found
    mask = matched_groups.notna()
    result = pd.Series('', index=dates_series.index)

    prefix_str = f"*{prefix}" if prefix else "*"
    result[mask] = prefix_str + matched_groups[mask].astype(str)

    return result


# not used
def get_group_from_date_column(dates_series, cage_dates_dict, behave_dict):
    # 1. Reverse lookup (Keep this, it's small and fast)
    date_to_group = {}
    for group, dates in cage_dates_dict.items():
        for d in dates:
            date_to_group[d] = group

    # 2. Determine prefix (Keep this, it's constant time)
    values = set(behave_dict.values())
    prefix = None
    if 'GG' in values or 'SB' in values:
        prefix = 'C'
    elif values & {'TN', 'TK', 'LN', 'NR', 'NY', 'ST', 'MN', 'SR'}:
        prefix = 'A'
    elif values & {'MN', 'MS', 'NH', 'LN', 'NR', 'NY', 'ST'}:
        prefix = 'B'
    elif values & {'MZ', 'SB', 'GG', 'LN', 'NR', 'MS'}:
        prefix = 'C'

    # 3. VECTORIZED DATE CONVERSION
    # Convert the entire series to datetime objects at once using C-speed
    temp_dates = pd.to_datetime(dates_series, dayfirst=True, errors='coerce')

    # Format the entire series to "DD.MM" strings at once
    short_dates = temp_dates.dt.strftime("%d.%m")

    # 4. VECTORIZED MAPPING
    # Map the group names from your dictionary
    matched_groups = short_dates.map(date_to_group)

    # 5. VECTORIZED STRING CONSTRUCTION
    # Use fillna to handle missing dates and then format the strings
    mask = matched_groups.notna()
    result = pd.Series('', index=dates_series.index)

    final_prefix = f"*{prefix}" if prefix else "*"
    result[mask] = final_prefix + matched_groups[mask].astype(str)

    return result


def generate_block_ID(df):
    # 1. Handle Phase: Vectorized filtering
    phase = df['phase'].fillna('').str.lower()
    phase = phase.where(phase.isin(['morn1', 'morn2', 'enrich']), '')

    # 2. Handle Group: Simple string addition
    group = '_' + df['group'].fillna('')

    # 3. Handle Trial: Use np.select instead of lambda
    # This avoids the slow "if pd.notnull" check per row
    trial_num = pd.to_numeric(df['trial'], errors='coerce')
    trial_str = np.select(
        [trial_num.notnull()],
        ['_trial' + trial_num.fillna(0).astype(int).astype(str)],
        default=''
    )

    # 4. Handle First Day: Vectorized mapping
    # Using a list of conditions (masks) and choices
    day_str = np.select(
        [df['first_day'] == True, df['first_day'] == False],
        ['_first', '_sec'],
        default=''
    )

    # 5. Handle Condition: Vectorized .map()
    condition_map = {'playback': '_play', 'crow': '_cont', 'baseline': '_base'}
    cond_str = df['condition'].fillna('').str.lower().map(condition_map).fillna('')

    # Final concatenation happens all at once
    return phase + group + trial_str + day_str + cond_str


def get_behavior_code_dict(df):
    pattern = re.compile(r"^Ind_(\d+)$")

    behave = []

    for col in df.columns:
        m = pattern.match(col)
        if m:
            behave.append((int(m.group(1)), col))

    behave.sort()

    behave_dict = {}
    for _, col in behave:
        val = df[col].dropna()
        if not val.empty:
            behave_dict[col] = val.iloc[0]

    return behave_dict


def reshape_behavior_data(df, behave_dict, schedule_dict, Cage_Compositions):
    working_df = df.copy()

    # 1. Phase mapping
    if '1_TIme' in working_df.columns:
        unique_times = working_df['1_TIme'].unique()
        phase_map = {t: get_phase_from_time(t) for t in unique_times}
        working_df['phase'] = working_df['1_TIme'].map(phase_map)
    print('working_df phase changed: \n', working_df)

    # 2. Schedule info via Merge
    sched_df = pd.DataFrame(schedule_dict).T.reset_index() # Adjust based on dict structure
    sched_df.columns = ['date', 'condition', 'trial', 'first_day']
    working_df = working_df.merge(sched_df, on='date', how='left')
    print('working_df shedule update: \n', working_df)

    # 3. Block ID
    working_df['block_ID'] = generate_block_ID(working_df)
    print('working_df Block ID added: \n', working_df)

    # 4. TODO: not properly implemented jet
    results = get_group_vectorized(working_df[''], Cage_Compositions, behave_dict)
    print('working_df group added: \n', working_df)

    return working_df, behave_dict


def process_sorted_data(sorted_df: pd.DataFrame, schedule_dict: dict, Cage_Compositions: dict):
    df_dec_ind: pd.DataFrame = detect_ind(sorted_df)
    print('df_dec_ind: \n', df_dec_ind)
    # Fix date column
    df_dec_ind["date"] = pd.to_datetime(df_dec_ind["date"]).dt.strftime("%d-%m-%Y")
    # Fix time column
    df_dec_ind["1_TIme"] = pd.to_datetime(df_dec_ind["1_TIme"]).dt.strftime("%H:%M")
    print('df_dec_ind date changed: \n', df_dec_ind)
    behave_dict: dict = get_behavior_code_dict(df_dec_ind)
    print('behave_dict: ', behave_dict)
    print(isinstance(df_dec_ind, type))

    resh_df: pd.DataFrame = reshape_behavior_data(df_dec_ind, behave_dict, schedule_dict, Cage_Compositions)

    # run follow up code to sort the data
