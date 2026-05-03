import os
import re
from datetime import datetime, timedelta, time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple, Any, Match, Pattern, Optional
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


def time_to_msm(time_value) -> int | None:
    """Convert various time formats to minutes since midnight."""

    try:
        # 1. If already numeric (e.g. corrected_msm)
        if isinstance(time_value, (int, float)):
            return int(time_value)

        # 2. If already a time-like object
        if isinstance(time_value, (time, pd.Timestamp)):
            time_obj = time_value

        else:
            # 3. Parse string values
            time_str = str(time_value).strip()

            try:
                time_obj = datetime.strptime(time_str, "%H:%M:%S").time()
            except ValueError:
                time_obj = datetime.strptime(time_str, "%H:%M").time()

        # 4. Convert to minutes
        return time_obj.hour * 60 + time_obj.minute

    except Exception as e:
        print(f"🛑 Failed to parse time value '{time_value}': {e}")
        return None


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
) -> pd.DataFrame:

    # --- Copy inputs to avoid mutation ---
    raw_df = raw_df.copy()
    raw_df = raw_df.rename(columns={"created_at": "date"})

    date_col = "date"

    # --- Parse dates ---
    raw_df[date_col] = pd.to_datetime(
        raw_df[date_col], format="%Y-%m-%dT%H:%M:%S.%fZ", errors="coerce"
    )

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

    raw_df["msm"] = calculate_df_msm(raw_df, time_col_raw)
    raw_df = raw_df.dropna(subset=["msm", date_col]).copy()

    # Normalize dates (ensure same day alignment)
    raw_df[date_col] = raw_df[date_col].dt.normalize()

    # --- Sort ---
    raw_df = raw_df.sort_values(["date", "msm"]).reset_index(drop=True)

    # Ensure date column is date-only
    raw_df['date'] = pd.to_datetime(raw_df['date']).dt.date

    os.makedirs(output_dir, exist_ok=True)

    updated_df: list = []
    all_missing: list = []
    for date, time_sorted_df in raw_df.groupby('date'):
        adjusted_df, gaps = analyze_time_gaps(time_sorted_df)
        for g in gaps:
            if not is_excluded(g):
                h, m = divmod(int(g), 60)

                #print(f"MISSING DATA in {file_type} / {date.strftime('%Y-%m-%d')} AT: {h:02d}:{m:02d}")

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

        updated_df.append(adjusted_df)

        # Save merged DataFrame
        adjusted_df.to_csv(
            output_file_path,
            index=False,
            sep=';'
        )

    time_corrected_df: pd.DataFrame = pd.concat(updated_df, ignore_index=True)

    save_missing_data(all_missing, output_dir, file_type)
    return time_corrected_df


def generate_timepoints() -> List[str]:
    times: List[str] = []
    # TODO: Check timepoints and find if all occation or not!

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
        # 1. Define New Column Names clearly
        ind_col = f"Ind_{idx}({ident})"
        beh_col = f"{ident}_Behavior"
        con_col = f"{ident}_Contact"
        prox_col = f"{ident}_AR"
        dist_col = f"{ident}_3M"
        note_col = f"{ident}_Notes"

        # 2. Assign the ID column (This always exists now)
        df[ind_col] = ident
        out_cols.append(ind_col)  # Add to list immediately

        # 3. Map Raw Names -> New Names
        rename_map = {
            behavior_col: beh_col,
            f"{order + 1}_Contact": con_col,
            f"{order + 2}_AR": prox_col,
            f"{order + 3}_3M": dist_col,
            f"{order + 4}_Notes": note_col
        }

        # 4. Only rename and track columns that actually exist in the CSV
        actual_rename = {k: v for k, v in rename_map.items() if k in df.columns}
        df.rename(columns=actual_rename, inplace=True)

        # 5. Add the successfully renamed values to our final output list
        out_cols.extend(actual_rename.values())

    # Final Filter: Keep only what is in the dataframe
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


def get_phase_from_time(time_value: float) -> Optional[str]:
    """
    Categorizes a time value into experimental phases (morn1, morn2, enrich, extra).

    Args:
        time_value: The time entry (string 'HH:M', datetime object, or Timestamp).

    Returns:
        String representing the phase or None if parsing fails.
    """
    if pd.isna(time_value) or time_value == '':
        return None
    msm: int = time_to_msm(time_value)

    # 4. Logic Gates for Phases
    if 570 <= msm <= 630:  # 09:30 - 10:30
        return 'morn1'
    elif 645 <= msm <= 705:  # 10:45 - 11:45
        return 'morn2'
    elif msm >= 720:  # 12:00 onwards
        return 'enrich'
    else:
        return 'extra'


def get_group(
        dates_series: pd.Series,
        cage_dates_dict: Dict[str, List[str]],
        behave_dict: Dict[str, str],
        file_type: str
) -> pd.Series:

    animal_ids = set(behave_dict.keys())
    prefix: str = file_type

    if not prefix:
        groups = {
            'A': {'TN', 'TK', 'LN', 'NR', 'NY', 'ST', 'MN', 'SR'},
            'B': {'MN', 'MS', 'NH', 'LN', 'NR', 'NY', 'ST'},
            'C': {'MZ', 'SB', 'GG', 'LN', 'NR', 'MS'}
        }
        if any(marker in animal_ids for marker in {'GG', 'SB'}):
            prefix = 'C'
        else:
            match_counts = {g: len(animal_ids & members) for g, members in groups.items()}
            max_matches = max(match_counts.values())
            if max_matches > 0:
                prefix = max(match_counts, key=match_counts.get)

    # Standardize dates and map cages
    clean_date_map = {}
    for cage, dates in cage_dates_dict.items():
        for d in dates:
            standard_d = str(d).replace('-', '.').replace('/', '.')
            clean_date_map[standard_d] = str(cage)

    temp_dates = pd.to_datetime(dates_series, dayfirst=True, errors='coerce')
    short_dates = temp_dates.dt.strftime("%d.%m")
    #TODO: Check UNKNOWN and if returned propperly!!!
    matched_cages = short_dates.map(clean_date_map).fillna("UNKNOWN_DATE")

    return matched_cages


def generate_block_ID(df: pd.DataFrame) -> pd.Series:
    """
    Concatenates experimental metadata into a unique string identifier.
    Example output: 'morn1_GroupA_trial1_first_play'
    """
    # 1. Phase: morn1, morn2, etc.
    phase: pd.Series = df['phase'].fillna('').str.lower()
    phase = phase.where(phase.isin(['morn1', 'morn2', 'enrich', 'extra']), '')

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
    # index e.g.: Ind_1(MZ)
    # Group 1: The digit (\d+)
    # Group 2: The initials within parentheses ([^)]+)
    pattern: Pattern[str] = re.compile(r"^Ind_(\d+)\(([^)]+)\)$")
    behave_dict: Dict[str, str] = {}

    for col in df.columns:
        m: Optional[Match[str]] = pattern.match(col)

        if m:
            # num = m.group(1) # Not strictly needed if using initials for behavior col
            initials: str = m.group(2)

            # The behavior column in your df follows the format 'MZ_Behavior'
            behavior_col: str = f"{initials}_Behavior"

            if behavior_col in df.columns:
                # Find first non-null for the ID code (e.g., 'MZ')
                first_code_idx = df[col].first_valid_index()
                # Find first non-null for the Behavior (e.g., 'E')
                first_behave_idx = df[behavior_col].first_valid_index()

                if first_code_idx is not None and first_behave_idx is not None:
                    code_key = df.at[first_code_idx, col]
                    behavior_val = df.at[first_behave_idx, behavior_col]

                    behave_dict[str(code_key)] = str(behavior_val)

    return behave_dict


def reshape_behavior_data(df: pd.DataFrame, file_type) -> pd.DataFrame:
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
    if 'corrected_msm' in working_df.columns:
        working_df['phase'] = working_df["corrected_msm"].apply(get_phase_from_time)
    # 3. Group Mapping (Refactored)
    # We call the function and assign the result to two new columns
    working_df['group'] = get_group(
        working_df['date'],
        Cage_Comp_Dates,
        behave_mapping,
        file_type
    )
    working_df['is_all_occasions'] = False
    # 4. Now generate Block ID (Now that phase, group, trial, etc. all exist)
    working_df['block_ID'] = generate_block_ID(working_df)

    return working_df


def process_sort_event(sorted_df: pd.DataFrame, file_type):
    df_dec_ind: pd.DataFrame = detect_ind(sorted_df)
    #print('df_dec_ind:', df_dec_ind)
    # Fix date column
    df_dec_ind["date"] = pd.to_datetime(df_dec_ind["date"]).dt.strftime("%d-%m-%Y")
    resh_df: pd.DataFrame = reshape_behavior_data(df_dec_ind, file_type)

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
    main_beh = behaviour

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
            {**metadata, 'Ind1': int1, 'behaviour': main_beh, 'partner': targets,
             'notes': ind_notes, **occ_flags, 'aggression': 1})

    # States: Grooming (GG, GR, GM)
    elif behaviour.upper() in {'GG', 'GR', 'GM'}:
        res['behaviour'].append(
            {**metadata, 'Ind1': int1, 'behaviour': main_beh, 'partner': targets, **solo_flags,
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
            current_flags['laying'] = 1

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
    final_output = {'distance': [], 'behaviour': [], 'occurrence': []}
    seen_dyads = set()

    metadata = {
        'ec5_uuid': row.get('ec5_uuid'),
        'condition': row.get('condition'),
        'date': row.get('date'),
        'msm': row.get('corrected_msm'),
        'group': row.get('group'),
        'is_all_occasions': row.get('is_all_occasions'),
        'trial': row.get('trial'),
        'phase': row.get('phase'),
        'block_ID': row.get('block_ID')
    }

    # 1. We need to find which "Ind_X" columns actually exist in the row
    # and map the Index (1-5) to the Initials (MZ, SB, etc.)
    index_to_initials = {}
    for col in row.index:
        # Match pattern like Ind_1(MZ)
        m = re.match(r"Ind_(\d+)\(([^)]+)\)", col)
        if m:
            index_to_initials[m.group(1)] = m.group(2)

    # 2. Loop through the found indices
    for i, initials in index_to_initials.items():
        # Get the ID (e.g., 'MZ') stored in the cell 'Ind_1(MZ)'
        full_col_name = f"Ind_{i}({initials})"
        int1 = row.get(full_col_name)

        if pd.isna(int1):
            continue

            # Use the INITIALS to get the behavior and proximity data
        behaviour = str(row.get(f'{initials}_Behavior', '')).strip()
        contact = row.get(f'{initials}_Contact')
        arr = row.get(f'{initials}_AR')
        three_met = row.get(f'{initials}_3M')
        # ind_notes = row.get(f'{initials}_Notes') # Check if this exists in your DF

        # --- PART A: SOCIAL DISTANCES ---
        if pd.isna(contact) and pd.isna(arr) and pd.isna(three_met):
            for other_id in beha_dict.keys():
                if other_id != int1:
                    # Create a sorted key: ("MN", "NY") instead of "MN-NY"
                    dyad_key = tuple(sorted([str(int1), str(other_id)]))

                    if dyad_key not in seen_dyads:
                        final_output['distance'].append({
                            **metadata, 'Ind1': int1, 'partner': other_id,
                            'dyad': f"{dyad_key[0]}-{dyad_key[1]}", 'distance': 4
                        })
                        seen_dyads.add(dyad_key)
        else:
            for val, dist_score in [(contact, 1), (arr, 2), (three_met, 3)]:
                if pd.notna(val):
                    for p in str(val).split(','):
                        p_clean = p.strip()
                        if p_clean:
                            dyad_key = tuple(sorted([str(int1), p_clean]))

                            if dyad_key not in seen_dyads:
                                final_output['distance'].append({
                                    **metadata, 'Ind1': int1, 'partner': p_clean,
                                    'dyad': f"{dyad_key[0]}-{dyad_key[1]}", 'distance': dist_score
                                })
                                seen_dyads.add(dyad_key)

        # --- PART B: BEHAVIOR CLASSIFICATION ---
        if behaviour and behaviour.lower() != 'nan' and behaviour != '':
            targets = contact if pd.notna(contact) else (arr if pd.notna(arr) else None)

            # Note: Ensure classify_behavior is updated to handle 'initials' if needed
            beh_data = classify_behavior(int1, behaviour, targets, metadata, None)
            final_output['behaviour'].extend(beh_data['behaviour'])
            final_output['occurrence'].extend(beh_data['occurrence'])

    return final_output


def process_sort_beh_dist(df):
    behave_mapping = get_behavior_code_dict(df)
    print('behave_mapping: ', behave_mapping)
    # Reshape everything into the 3 target tables
    tables = reshape_behavior_data_to_tables(df, behave_mapping)

    # Access your final DataFrames
    df_dist = tables['df_distance']
    df_beh = tables['df_behaviour']
    df_occ = tables['df_occurrence']

    print(f"Processed {len(df_dist)} distance records and {len(df_beh)} behavior states.")

    return df_beh, df_dist, df_occ


def to_msm(time_val):
    """Converts HH:MM or Timestamp to minutes since midnight."""
    if pd.isna(time_val): return 0
    s = str(time_val).strip()
    if ' ' in s: s = s.split(' ')[-1]
    try:
        parts = s.split(':')
        return int(parts[0]) * 60 + int(parts[1])
    except:
        return 0


def get_occurrence_metadata(row: pd.Series, prefix: str) -> Tuple[str, str]:
    """
    Refactored Metadata Extractor:
    Uses the individual's column prefix to grab related Contact and Notes.
    """
    # 1. Extract Contact/Partner (Check Contact first, then AR as fallback)
    partner = row.get(f"{prefix}_Contact")
    if pd.isna(partner) or str(partner).strip() == "":
        partner = row.get(f"{prefix}_AR")

    # 2. Extract Notes
    notes = row.get(f"{prefix}_Notes")

    # 3. Clean and Return
    partner_str = str(partner).strip() if pd.notna(partner) else ""
    notes_str = str(notes).strip() if pd.notna(notes) else ""

    return partner_str, notes_str


def parse_flexible_date(date_str):
    if pd.isna(date_str) or str(date_str).strip() == "":
        return pd.NaT

    date_str = str(date_str).strip()

    try:
        return pd.to_datetime(date_str, dayfirst=False, errors='coerce')

    except:
        return pd.NaT


def standardize_dataframe_dates(df):
    # 1. Identify the source column (created_at or date)
    source_col = 'created_at' if 'created_at' in df.columns else 'date'

    if source_col not in df.columns:
        print(f"⚠️ No column named '{source_col}' found.")
        return df

    # 2. Rename original column to 'date_old'
    # Use a dictionary to avoid errors if 'date_old' already exists
    df = df.rename(columns={source_col: 'date_old'})

    # 3. Parse dates (using the Month-First logic discussed)
    # df['date_old'] now contains your 04.05.2025 or 4/16/25
    df['date_dt'] = pd.to_datetime(df['date_old'], dayfirst=False, errors='coerce').ffill()

    # 4. Create the new 'date' column in "DD Month YYYY" format
    df['date'] = df['date_dt'].dt.strftime('%d %B %Y')

    # 5. Reorder columns so 'date' is where the original was
    # Find index of 'date_old' and place 'date' and 'date_dt' next to it
    cols = list(df.columns)
    old_idx = cols.index('date_old')

    # Remove 'date' and 'date_dt' from their current positions
    cols.remove('date')
    cols.remove('date_dt')

    # Insert them right after 'date_old'
    cols.insert(old_idx + 1, 'date')
    cols.insert(old_idx + 2, 'date_dt')

    df = df[cols]

    return df

# TODO: vershcieben in pipeline und intermediate step schaffen!!!
def apply_schedule_and_phase(df, schedule_dict, get_phase_from_time_func):
    """
    Integrates trial, condition, first_day, and phase into the working DataFrame.
    """
    # 1. Create a lookup table from the schedule_dict
    schedule_rows = []
    for (cond, trial), dates in schedule_dict.items():
        for i, d in enumerate(dates):
            # Standardizing date format to match your df['date_str'] or 'date'
            # Assuming your schedule_dict uses "04.04.2025" and df uses "04 April 2025"
            # We convert both to a standard datetime for the merge
            schedule_rows.append({
                'merge_date': pd.to_datetime(d, dayfirst=True),
                'condition': cond,
                'trial': trial,
                'first_day': (i == 0)  # True if it's the first date in the list
            })

    sched_df = pd.DataFrame(schedule_rows)

    # 2. Ensure the main df has a datetime column for merging
    if 'date_dt' not in df.columns:
        df['date_dt'] = pd.to_datetime(df['date'], errors='coerce')

    # 3. Merge schedule info into the main dataframe
    df = df.merge(sched_df, left_on='date_dt', right_on='merge_date', how='left')

    # Clean up merge column
    df.drop(columns=['merge_date'], inplace=True)

    # 4. Apply Phase mapping based on '1_TIme'
    if '1_TIme' in df.columns:
        # result looks like: 'Morning', 'Afternoon', etc.
        df['phase'] = df['1_TIme'].apply(get_phase_from_time_func)

    return df


def process_all_occurrence(df, file_type):
    """Main function to flatten and validate occurrence data."""

    # This creates 'date_dt' (Timestamp) and 'date' (String)
    df = standardize_dataframe_dates(df)
    df = detect_ind(df)
    behave_dict = get_behavior_code_dict(df)
    # Get Trial, Condition, and First_Day info
    df = apply_schedule_and_phase(df, schedule_dict, get_phase_from_time)

    # Pre-calculate Cages (Pass the Series, not a string)
    df['group'] = get_group(df['date_dt'], Cage_Comp_Dates, behave_dict, file_type)
    df['is_all_occasions'] = True
    #TODO: mark all-occation entrie!

    # Generate the unique Block Identifier
    df['block_ID'] = generate_block_ID(df)

    # 4. Identify Behavior Columns
    beh_cols = [c for c in df.columns if '_Behavior' in c]
    rows = []

    # 5. Process Rows
    for _, row in df.iterrows():
        # Metadata context for the row
        msm_val = to_msm(row.get('1_TIme', '00:00'))
        metadata = {
            'ec5_uuid': row.get('ec5_uuid', ''),
            'condition': row.get('condition', 'Unknown'),
            'date': row.get('date_dt', ''),
            'msm': msm_val,
            'group': row.get('group', file_type),
            'is_all_occasions': row.get('is_all_occasions', None),
            'trial': row.get('trial', ''),
            'phase': row.get('phase', ''),
            'first_day': row.get('first_day', False),
            'block_ID': row.get('block_ID', '')
        }
        for b_col in beh_cols:
            val = row[b_col]
            if pd.isna(val) or str(val).strip() == "":
                continue

            # b_col is e.g., '4_ST_Behavior' to '4_ST'
            #prefix = "_".join(b_col.split('_')[:2])
            #print('beh_cols: ', beh_cols)

            # Extract Ind ID for the 'Ind1' column (e.g., 'ST')
            ind_id = b_col.split('_')[0]
            #print('ind_id: ', ind_id)

            # Use the new Refactored Worker
            partner, notes = get_occurrence_metadata(row, ind_id)
            #print('partner / notes: ', partner, notes)
            # --- Behavioral Logic ---
            beh_raw = str(val).strip().upper()

            # Default values
            main_beh = beh_raw
            #qualifier = None
            is_playing = 0
            is_aggression = 0

            if beh_raw.startswith('PL'):
                is_playing = 1
                # For Play, usually kept as 'PL' or split if you have 'PLx'
                if len(beh_raw) > 2:
                    main_beh = beh_raw[:2]
                #    qualifier = beh_raw[2:]

            elif beh_raw in {'AG', 'DS', 'AR'}:
                is_aggression = 1
                # SPLIT LOGIC: A/G, D/S, A/R
                main_beh = beh_raw[:2]
               # qualifier = beh_raw[1]

            # Append to our list
            rows.append({
                **metadata,
                'Ind1': ind_id,
                'behaviour': main_beh,
               # 'qualifier': qualifier,
                'partner': partner if partner else None,
                'notes': notes if notes else None,
                'playing': is_playing,
                'aggression': is_aggression
            })

    # 6. Save
    if rows:
        # Create the DataFrame from the rows list
        occ_df: pd.DataFrame = pd.DataFrame(rows)

        return occ_df

    else:
        print(f"⚠️ No occurrences found for Cage {file_type}")


def read_data(input_file_path: str):
    """
    Read CSV or Excel
    :param input_file_path:
    :return: Pandas Dataframe
    """
    try:
        if input_file_path.endswith(".csv"):
            df_raw: pd.DataFrame = pd.read_csv(input_file_path, sep=None, engine='python')
        else:
            df_raw: pd.DataFrame = pd.read_excel(input_file_path)
        return df_raw

    except Exception as e:
        print(f"🛑 Error reading file {input_file_path}: {e}")
        return None


def file_name(input_file_path, df_raw, all_raw_list):
    # --- Determine file type ---
    file_name: str = os.path.basename(input_file_path).lower()
    file_type: str | None = None

    if "_a_" in file_name:
        file_type = "A"
    elif "_b_" in file_name:
        file_type = "B"
    elif "_c_" in file_name:
        file_type = "C"

    if file_type:
        print(f"→ Detected {file_type} file: {input_file_path}")

        all_raw_list[file_type].append(df_raw)

        return all_raw_list, file_type

    else:
        print(f"⚠️ Could not detect file type (A, B, or C) for: {input_file_path}. Skipping.")
        return None


def sanity_check_msm_coverage(df: pd.DataFrame, file_type: str) -> pd.DataFrame:

    if df.empty:
        print(f"⚠️ Empty dataframe for cage {file_type}")
        return pd.DataFrame()

    df = df.copy()

    # ✅ Always define msm column
    if 'corrected_msm' in df.columns:
        df['msm'] = df['corrected_msm']
    elif 'msm' not in df.columns:
        print("⚠️ No msm column found!")
        return pd.DataFrame()

    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    summary = (
        df.groupby(['date', 'msm'])
        .size()
        .reset_index(name='n_observations')
    )

    summary['cage'] = file_type

    return summary


def sanity_check_with_missing(all_sanity_df: pd.DataFrame) -> pd.DataFrame:

    if all_sanity_df.empty:
        print("⚠️ sanity_combined is empty!")
        return pd.DataFrame()

    required_cols = {'date', 'msm', 'cage'}
    missing_cols = required_cols - set(all_sanity_df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns in sanity data: {missing_cols}")

    expected = build_full_expected_grid()

    merged = expected.merge(
        all_sanity_df,
        on=['date', 'msm', 'cage'],
        how='left'
    )

    merged['n_observations'] = merged['n_observations'].fillna(0)

    def classify(n):
        if n == 1:
            return "OK"
        elif n == 0:
            return "MISSING"
        else:
            return "DUPLICATE"

    merged['status'] = merged['n_observations'].apply(classify)

    return merged


def build_full_expected_grid():
    rows = []

    timepoints = generate_timepoints()
    cages = ['A', 'B', 'C']

    for (condition, trial), dates in schedule_dict.items():
        for d in dates:
            date_obj = pd.to_datetime(d, dayfirst=True)

            for t in timepoints:
                msm = time_to_msm(t)

                for cage in cages:
                    rows.append({
                        "date": date_obj,
                        "msm": msm,
                        "cage": cage
                    })

    return pd.DataFrame(rows)


def classify_row(row):
    if row['n_missing'] == 0 and row['n_duplicate'] == 0:
        return "OK"
    elif row['n_duplicate'] > 0:
        return "DUPLICATE"
    else:
        return "MISSING"


def count_status(row):
    values = [row['A'], row['B'], row['C']]

    n_ok = sum(1 for v in values if v == 1)
    n_missing = sum(1 for v in values if v == 0)
    n_duplicate = sum(1 for v in values if v > 1)

    return pd.Series({
        'n_ok': n_ok,
        'n_missing': n_missing,
        'n_duplicate': n_duplicate
    })


def sanity_check(all_sanity: list, output_dir: str) -> None:
    sanity_combined = pd.concat(all_sanity, ignore_index=True)

    # 🔥 NEW: include missing data
    sanity_full = sanity_check_with_missing(sanity_combined)
    summary = sanity_full['status'].value_counts()
    sanity_pivot = sanity_full.pivot_table(
        index=['date', 'msm'],
        columns='cage',
        values='n_observations',
        fill_value=0
    ).reset_index()
    sanity_pivot['total'] = sanity_pivot[['A', 'B', 'C']].sum(axis=1)
    sanity_pivot[['n_ok', 'n_missing', 'n_duplicate']] = sanity_pivot.apply(
        count_status, axis=1
    )
    sanity_pivot['row_status'] = sanity_pivot.apply(classify_row, axis=1)

    daily_missing = (
        sanity_pivot.assign(
            A_missing=(sanity_pivot['A'] == 0),
            B_missing=(sanity_pivot['B'] == 0),
            C_missing=(sanity_pivot['C'] == 0),
        )
            .groupby('date')[['A_missing', 'B_missing', 'C_missing']]
            .sum()
            .reset_index()
            .rename(columns={
            'A_missing': 'A_missing_day',
            'B_missing': 'B_missing_day',
            'C_missing': 'C_missing_day'
        })
    )
    sanity_pivot = sanity_pivot.merge(daily_missing, on='date', how='left', validate='m:1')

    # Save
    sanity_pivot.to_csv(os.path.join(output_dir, "sanity_check.csv"), index=False)
