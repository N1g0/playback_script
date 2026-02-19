import os
import re
from datetime import datetime, timedelta, time
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Any, Match, Pattern, Optional, Union, Set
from dateutil import parser

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

def detect_ind(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # 1. Base columns
    base_cols = ["ec5_uuid", "date", "1_TIme", "msm", "corrected_msm"]
    base_cols = [c for c in base_cols if c in df.columns]

    # 2. Find behavior columns (e.g., 4_MZ_Behavior)
    pattern = re.compile(r"^(\d+)_([A-Za-z]+)_Behavior$", re.IGNORECASE)

    matches = []
    for col in df.columns:
        m = pattern.match(col)
        if m:
            order = int(m.group(1))
            ident = m.group(2)
            matches.append((order, ident, col))

    # Sort by the numeric prefix (4, 11, 18, etc.)
    matches.sort(key=lambda x: x[0])

    out_cols = base_cols.copy()

    # 3. Dynamic mapping of associated data
    for idx, (order, ident, behavior_col) in enumerate(matches, start=1):
        # New Column Names
        ind_col = f"Ind_{idx}"
        beh_col = f"Ind_{idx}_Behavior"
        con_col = f"Ind_{idx}_Contact"
        prox_col = f"Ind_{idx}_AR"
        dist_col = f"Ind_{idx}_3M"
        note_col = f"Ind_{idx}_Notes"

        # Assign Animal ID (MZ, SB, etc.)
        df[ind_col] = ident

        # Rename the columns by looking for the expected numeric index
        # MZ is 4, so Contact is 5, AR is 6, 3M is 7...
        rename_map = {
            behavior_col: beh_col,
            f"{order + 1}_Contact": con_col,
            f"{order + 2}_AR": prox_col,
            f"{order + 3}_3M": dist_col,
            f"{order + 5}_Notes": note_col
        }

        # Only rename if the column actually exists in raw data
        actual_rename = {k: v for k, v in rename_map.items() if k in df.columns}
        df.rename(columns=actual_rename, inplace=True)

        # Add all successfully renamed columns to our final output list
        out_cols.extend([ind_col])
        out_cols.extend(list(actual_rename.values()))

    # Ensure we don't crash if out_cols contains something missing
    final_cols = [c for c in out_cols if c in df.columns]
    return df[final_cols].copy()


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


def get_phase_from_time(time_value: Union[str, time, pd.Timestamp]) -> Optional[str]:
    """
    Categorizes a time value into experimental phases (morn1, morn2, enrich, extra).

    Args:
        time_value: The time entry (string 'HH:M', datetime object, or Timestamp).

    Returns:
        String representing the phase or None if parsing fails.
    """
    if pd.isna(time_value) or time_value == '':
        return None

    try:
        # 1. Handle cases where time_value is already a time-like object
        if isinstance(time_value, (time, pd.Timestamp)):
            time_obj = time_value
        else:
            # 2. Parse string values
            time_str: str = str(time_value).strip()
            try:
                time_obj = datetime.strptime(time_str, '%H:%M:%S').time()
            except ValueError:
                time_obj = datetime.strptime(time_str, '%H:%M').time()

        # 3. Convert to total minutes for easier comparison
        total_minutes: int = time_obj.hour * 60 + time_obj.minute

        # 4. Logic Gates for Phases
        if 570 <= total_minutes <= 630:  # 09:30 - 10:30
            return 'morn1'
        elif 645 <= total_minutes <= 705:  # 10:45 - 11:45
            return 'morn2'
        elif total_minutes >= 720:  # 12:00 onwards
            return 'enrich'
        else:
            return 'extra'

    except Exception as e:
        print(f"🛑 Failed to parse time value '{time_value}': {e}")
        return None


def get_group_vectorized(
        dates_series: pd.Series,
        cage_dates_dict: Dict[str, List[str]],
        behave_dict: Dict[str, str]
) -> pd.Series:
    """
    Vectorized mapping of dates to groups with a behavior-based group prefix.

    Args:
        dates_series: Column containing date strings (e.g., '03-04-2025').
        cage_dates_dict: Mapping of Cage names to lists of dates ['DD.MM'].
        behave_dict: Dictionary of {animal_id: behavior_code}.

    Returns:
        pd.Series: A series of formatted group strings like '*C_Cage1'.
    """
    # --- STEP 1: PRE-CALCULATE PREFIX ---
    # We look at the set of animal IDs found in the current file
    animal_ids: Set[str] = set(behave_dict.keys())

    group_A_ids: set = {'TN', 'TK', 'LN', 'NR', 'NY', 'ST', 'MN', 'SR'}
    group_B_ids: set = {'MN', 'MS', 'NH', 'LN', 'NR', 'NY', 'ST'}
    group_C_ids: set = {'MZ', 'SB', 'GG', 'LN', 'NR', 'MS'}

    # Determine prefix based on priority
    if 'GG' in animal_ids or 'SB' in animal_ids:
        prefix: str = 'C'
    elif animal_ids & group_A_ids:
        prefix: str = 'A'
    elif animal_ids & group_B_ids:
        prefix: str = 'B'
    elif animal_ids & group_C_ids:
        prefix: str = 'C'
    else:
        prefix: str = ''

    # --- STEP 2: BUILD REVERSE LOOKUP DICT ---
    # Converts {Group: [Dates]} -> {Date: Group}
    date_to_group: Dict[str, str] = {
        d: str(group) for group, dates in cage_dates_dict.items() for d in dates
    }

    # --- STEP 3: VECTORIZED DATA PROCESSING ---
    # Ensure dates are datetime objects for formatting
    temp_dates = pd.to_datetime(dates_series, dayfirst=True, errors='coerce')

    # Format to "DD.MM" to match cage_dates_dict keys
    short_dates = temp_dates.dt.strftime("%d.%m")

    # Map the groups via the dictionary
    matched_groups = short_dates.map(date_to_group)

    # Construct final result
    mask = matched_groups.notna()
    result = pd.Series('', index=dates_series.index)

    prefix_str = f"{prefix}_" if prefix else "*"
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


def generate_block_ID(df: pd.DataFrame) -> pd.Series:
    """
    Concatenates experimental metadata into a unique string identifier.
    Example output: 'morn1_GroupA_trial1_first_play'
    """
    # 1. Phase: morn1, morn2, etc.
    phase: pd.Series = df['phase'].fillna('').str.lower()
    phase = phase.where(phase.isin(['morn1', 'morn2', 'enrich']), '')

    # 2. Group: Needs to be handled carefully if empty
    group: pd.Series = '_' + df['group'].fillna('Unknown').astype(str)

    # 3. Trial: Extracting numeric trial ID
    trial_num = pd.to_numeric(df['trial'], errors='coerce')
    trial_str: np.ndarray = np.select(
        [trial_num.notnull()],
        ['_trial' + trial_num.fillna(0).astype(int).astype(str)],
        default=''
    )

    # 4. First Day vs Second Day
    day_str: np.ndarray = np.select(
        [df['first_day'] == True, df['first_day'] == False],
        ['_first', '_sec'],
        default=''
    )

    # 5. Condition Mapping
    condition_map: Dict[str, str] = {'playback': '_play', 'crow': '_cont', 'baseline': '_base'}
    cond_str: pd.Series = df['condition'].fillna('').str.lower().map(condition_map).fillna('')

    # Final result is a Series of concatenated strings
    return phase + group + trial_str + day_str + cond_str


def get_behavior_code_dict(df: pd.DataFrame) -> Dict[str, str]:
    """
    Creates a mapping between individual codes (e.g., 'MZ') and their
    corresponding behavior values (e.g., 'E') from the first row of the DataFrame.

    Args:
        df: A pandas DataFrame containing 'Ind_N' and 'Ind_N_Behavior' columns.

    Returns:
        A dictionary where keys are individual codes and values are behaviors.
    """
    # Type hint for the compiled regex pattern
    pattern: Pattern[str] = re.compile(r"^Ind_(\d+)$")
    behave_dict: Dict[str, str] = {}

    for col in df.columns:
        m: Optional[Match[str]] = pattern.match(col)

        if m:
            num: str = m.group(1)
            behavior_col: str = f"Ind_{num}_Behavior"

            # Check if the paired behavior column exists in the DataFrame
            if behavior_col in df.columns:
                code_key = df[col].iloc[0]
                behavior_val = df[behavior_col].iloc[0]

                # Ensure the key is valid (not NaN) before adding to dictionary
                if pd.notna(code_key) and pd.notna(behavior_val):
                    behave_dict[str(code_key)] = str(behavior_val)

    return behave_dict


def reshape_behavior_data(df: pd.DataFrame) -> pd.DataFrame:
    working_df = df.copy()
    behave_mapping = get_behavior_code_dict(working_df)

    # 1. Schedule & Trial Info (MUST come before block_ID)
    rows = []
    for (cond, trial), dates in schedule_dict.items():
        for d in dates:
            clean_date = d.replace('.', '-')
            # Note: We include trial and first_day logic here
            is_first_day = (d == dates[0])
            rows.append({
                'condition': cond,
                'trial': trial,
                'date': clean_date,
                'first_day': is_first_day
            })

    sched_df = pd.DataFrame(rows)
    working_df = working_df.merge(sched_df, on='date', how='left')

    # 2. Phase mapping (Required for block_ID)
    if '1_TIme' in working_df.columns:
        working_df['phase'] = working_df['1_TIme'].apply(get_phase_from_time)

    # 3. Group Mapping (Required for block_ID)
    working_df['group'] = get_group_vectorized(
        working_df['date'],
        Cage_Comp_Dates,
        behave_mapping
    )
    #print('working_df group added: \n', working_df.head())
    # 4. Now generate Block ID (Now that phase, group, trial, etc. all exist)
    working_df['block_ID'] = generate_block_ID(working_df)

    #print('working_df Block ID added: \n', working_df[['date', 'block_ID']].head())

    return working_df


def process_sort_event(sorted_df: pd.DataFrame):
    df_dec_ind: pd.DataFrame = detect_ind(sorted_df)
    #print('df_dec_ind: \n', df_dec_ind)
    # Fix date column
    df_dec_ind["date"] = pd.to_datetime(df_dec_ind["date"]).dt.strftime("%d-%m-%Y")
    # Fix time column
    df_dec_ind["1_TIme"] = pd.to_datetime(df_dec_ind["1_TIme"]).dt.strftime("%H:%M")
    #print('df_dec_ind date changed: \n', df_dec_ind)

    resh_df: pd.DataFrame = reshape_behavior_data(df_dec_ind)
    #print('resh_df: \n', resh_df)

    return resh_df


# --- HELPER: METADATA ---
def extract_metadata(row: pd.Series) -> Dict[str, Any]:
    return {
        'ec5_uuid': row.get('ec5_uuid'),
        'condition': row.get('condition'),
        'date': row.get('date'),
        'msm': row.get('msm'),
        'group': row.get('group'),
        'trial': row.get('trial'),
        'phase': row.get('phase'),
        'block_ID': row.get('block_ID'),
        'notes': row.get('notes')  # Global notes
    }


# --- HELPER: SOCIAL DISTANCES ---
def process_social_distances(int1: str, row: pd.Series, idx: int, beha_dict: dict, metadata: dict) -> List[Dict]:
    distances = []
    # Access renamed columns from detect_ind
    contact = row.get(f'Ind_{idx}_Contact')
    arr = row.get(f'Ind_{idx}_AR')
    three_met = row.get(f'Ind_{idx}_3M')

    if pd.isna(contact) and pd.isna(arr) and pd.isna(three_met):
        # Default distance 4 for all cage mates
        for other in beha_dict.values():
            if other != int1:
                distances.append({
                    **metadata, 'Ind1': int1, 'partner': other,
                    'dyad': f"{int1}-{other}", 'distance': 4
                })
    else:
        levels = [(contact, 1), (arr, 2), (three_met, 3)]
        for val, dist in levels:
            if pd.notna(val):
                for p in str(val).split(','):
                    p_clean = p.strip()
                    distances.append({
                        **metadata, 'Ind1': int1, 'partner': p_clean,
                        'dyad': f"{int1}-{p_clean}", 'distance': dist
                    })
    return distances


# --- HELPER: BEHAVIOR CLASSIFICATION ---
def classify_behavior(int1: str, behaviour: str, targets: Any, metadata: dict, ind_notes: Any) -> Dict:
    res = {'behaviour': [], 'occurrence': []}
    if not behaviour: return res

    behaviour = str(behaviour).strip()
    # Logic for qualifiers (second char of code)
    main_beh = behaviour[0] if len(behaviour) > 1 else behaviour
    qualifier = behaviour[1] if len(behaviour) > 1 else None

    # Binary Flag Templates
    solo_flags = {k: 0 for k in ['eating', 'playing', 'moving', 'resting', 'sitting', 'self_direct', 'grooming']}
    occ_flags = {'playing': 0, 'aggression': 0}

    # Occurrence: Play (PL) or Aggression (AG, DS, AR)
    if behaviour.upper().startswith('PL'):
        res['occurrence'].append(
            {**metadata, 'Ind1': int1, 'behaviour': 'PL', 'partner': targets, 'notes': ind_notes, **occ_flags,
             'playing': 1})
    elif behaviour.upper() in {'AG', 'DS', 'AR'}:
        res['occurrence'].append(
            {**metadata, 'Ind1': int1, 'behaviour': main_beh, 'qualifier': qualifier, 'partner': targets,
             'notes': ind_notes, **occ_flags, 'aggression': 1})

    # States: Grooming (GG, GR, GM)
    elif behaviour.upper() in {'GG', 'GR', 'GM'}:
        res['behaviour'].append(
            {**metadata, 'Ind1': int1, 'behaviour': main_beh, 'qualifier': qualifier, 'partner': targets, **solo_flags,
             'grooming': 1})

    # States: Solo
    else:
        current_flags = solo_flags.copy()
        if behaviour == 'E':
            current_flags['eating'] = 1
        elif behaviour == 'M':
            current_flags['moving'] = 1
        elif behaviour == 'SD':
            current_flags['self_direct'] = 1
        elif behaviour == 'RS':
            current_flags['resting'] = 1
            current_flags['sitting'] = 1
        elif behaviour == 'RL':
            current_flags['resting'] = 1

        res['behaviour'].append(
            {**metadata, 'Ind1': int1, 'behaviour': behaviour, 'qualifier': None, 'partner': None, **current_flags})

    return res


def reshape_behavior_data_to_tables(df: pd.DataFrame, beha_dict: Dict[str, str]):
    """
    MANAGER FUNCTION: Takes the full DataFrame, processes each row,
    and returns three separate DataFrames.
    """
    all_distances = []
    all_behaviours = []
    all_occurrences = []

    # Iterate through every row in your input DataFrame
    for _, row in df.iterrows():
        # Call the worker function for this specific row
        row_results = reshape_row_to_multiple(row, beha_dict)

        # Collect the lists returned by the worker
        all_distances.extend(row_results['distance'])
        all_behaviours.extend(row_results['behaviour'])
        all_occurrences.extend(row_results['occurrence'])

    # Convert the lists of dictionaries into tidy DataFrames
    return {
        "df_distance": pd.DataFrame(all_distances),
        "df_behaviour": pd.DataFrame(all_behaviours),
        "df_occurrence": pd.DataFrame(all_occurrences)
    }


def reshape_row_to_multiple(row: pd.Series, beha_dict: Dict[str, str]) -> Dict[str, List[Any]]:
    """
    WORKER FUNCTION: Processes one row and handles the 'NaN' string errors.
    """
    final_output = {'distance': [], 'behaviour': [], 'occurrence': []}

    # 1. Extract Metadata (Context for every new row generated)
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

    # 2. Loop through the 5 animals (Ind_1 to Ind_5)
    for i in range(1, 6):
        int1 = row.get(f'Ind_{i}')
        if pd.isna(int1):
            continue  # Skip empty animal slots

        # SAFETY: Convert to string and handle NaN to prevent regex/strip errors
        behaviour = str(row.get(f'Ind_{i}_Behavior', '')).strip()
        contact = row.get(f'Ind_{i}_Contact')
        arr = row.get(f'Ind_{i}_AR')
        three_met = row.get(f'Ind_{i}_3M')
        ind_notes = row.get(f'Ind_{i}_Notes')

        # --- PART A: SOCIAL DISTANCES ---
        # Logic: If no specific proximity is noted, assume distance 4 (Far)
        if pd.isna(contact) and pd.isna(arr) and pd.isna(three_met):
            for other_id in beha_dict.keys():
                if other_id != int1:
                    final_output['distance'].append({
                        **metadata, 'Ind1': int1, 'partner': other_id,
                        'dyad': f"{int1}-{other_id}", 'distance': 4
                    })
        else:
            # Map levels: 1=Contact, 2=AR, 3=3M
            for val, dist_score in [(contact, 1), (arr, 2), (three_met, 3)]:
                if pd.notna(val):
                    # Handle multiple partners like 'LN, NR'
                    for p in str(val).split(','):
                        p_clean = p.strip()
                        if p_clean:
                            final_output['distance'].append({
                                **metadata, 'Ind1': int1, 'partner': p_clean,
                                'dyad': f"{int1}-{p_clean}", 'distance': dist_score
                            })

        # --- PART B: BEHAVIOR CLASSIFICATION ---
        # We only process if there is a behavior code (e.g., 'E', 'PL', 'GG')
        if behaviour and behaviour.lower() != 'nan' and behaviour != '':
            # Determine target for social behaviors
            targets = contact if pd.notna(contact) else (arr if pd.notna(arr) else None)

            # Use classify_behavior helper (modular logic)
            beh_data = classify_behavior(int1, behaviour, targets, metadata, ind_notes)
            final_output['behaviour'].extend(beh_data['behaviour'])
            final_output['occurrence'].extend(beh_data['occurrence'])

    return final_output


def process_sort_beh_dist(df):
    behave_mapping = get_behavior_code_dict(df)

    # Reshape everything into the 3 target tables
    tables = reshape_behavior_data_to_tables(df, behave_mapping)

    # Access your final DataFrames
    df_dist = tables['df_distance']
    df_beh = tables['df_behaviour']
    df_occ = tables['df_occurrence']

    print(f"Processed {len(df_dist)} distance records and {len(df_beh)} behavior states.")

    return df_beh, df_dist, df_occ
