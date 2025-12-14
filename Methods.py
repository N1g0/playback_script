import re
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional, Sequence
import matplotlib.pyplot as plt
import math


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


def compare_df_msm_aware(
    raw_df: pd.DataFrame,
    empty_df: pd.DataFrame,
    file_type: str,
    ind: int = 0,
    time_col_raw: str = "1_TIme",
    time_col_empty: str = "time",
) -> pd.DataFrame:

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

    # ------------------------------------------------------------------
    # NEW SECTION: Maximum bipartite matching per date (±2 min tolerance)
    # ------------------------------------------------------------------

    print('raw_df: ', raw_df)
    print('valid_empty_df: ', valid_empty_df)

    result_rows = []

    for current_date in valid_empty_df[date_col].unique():
        tdf = (valid_empty_df[valid_empty_df[date_col] == current_date]
               .sort_values(by="msm")
               .reset_index())
        rdf = (raw_df[raw_df[date_col] == current_date]
               .sort_values(by="msm")
               .reset_index())

        print('tdf: ', tdf)
        print('rdf: ', rdf)
        if len(tdf) == 0 or len(rdf) == 0:
            result_rows.append(tdf.assign(match_raw_idx=pd.NA))
            continue

        t = tdf["msm"].to_numpy()
        d = rdf["msm"].to_numpy()

        print('t: ', t)
        print('d: ', d)

        # Build adjacency list: template index -> data indices compatible within ±2 minutes
        edges = defaultdict(list)
        for i, t_i in enumerate(t):
            left = np.searchsorted(d, t_i - 2, side="left")
            right = np.searchsorted(d, t_i + 2, side="right")
            for j in range(left, right):
                edges[i].append(j)

        # Kuhn algorithm for maximum bipartite matching
        match_to_template = [-1] * len(d)

        def try_kuhn(v, visited):
            for u in edges[v]:
                if visited[u]:
                    continue
                visited[u] = True

                if match_to_template[u] == -1 or try_kuhn(match_to_template[u], visited):
                    match_to_template[u] = v
                    return True
            return False

        for v in range(len(t)):
            visited = [False] * len(d)
            try_kuhn(v, visited)

        # Build match result: template idx -> raw idx (or NaN)
        template_to_raw = {v: None for v in range(len(t))}
        for raw_idx, tmpl_idx in enumerate(match_to_template):
            if tmpl_idx != -1:
                template_to_raw[tmpl_idx] = raw_idx
#######################################################################################################################
        #TODO: some mistake in the following sorting step!!!!!!!!
        matched = []
        for tmpl_idx, raw_idx in template_to_raw.items():
            if raw_idx is None:
                matched.append({
                    **tdf.loc[tmpl_idx].to_dict(),
                    "match_raw_idx": pd.NA
                })
            else:
                row = {
                    **tdf.loc[tmpl_idx].to_dict(),
                    "match_raw_idx": rdf.loc[raw_idx, "index"],
                }
                matched.append(row)

        result_rows.append(pd.DataFrame(matched))
    print('result_rows: ', result_rows)

    # Combine all results
    matched_template = pd.concat(result_rows, ignore_index=True)
    # Sort the final result by date and msm (or just msm if date_col is preserved)
    matched_template = matched_template.sort_values(by=[date_col, 'msm']).reset_index(drop=True)

    print(matched_template)
    # ------------------------------------------------------------------
    # Join matched template rows with raw_df rows
    # ------------------------------------------------------------------

    matched_template = matched_template.merge(
        raw_df.add_prefix("raw_"),
        left_on="match_raw_idx",
        right_on="raw_index",
        how="left"
    )

    # Add rows from template that had invalid time/date
    final = pd.concat([matched_template, invalid_empty_df], ignore_index=True)
    print('final: ', final)

    return final


def _calc_msm(series: pd.Series) -> pd.Series:
    """
    Return minutes-since-midnight as float (can include fractional minutes).
    Accepts a pd.Series of dtype datetime64[ns].
    """
    td = series - series.dt.normalize()
    return td.dt.total_seconds() / 60.0

# New method, test after fixing old one!
def match_templates_to_raw_multi_day(
    raw_df: pd.DataFrame,
    template_df: pd.DataFrame,
    file_type: str,
    time_col_raw: str = "1_TIme",
    time_col_template: str = "time",
    date_col_raw: str = "date",
    date_col_template: str = "date",
    tolerance_minutes: float = 2.0,
    raw_time_formats: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Match template rows (template_df) to raw rows (raw_df) across multiple dates.
    - Each template can match at most one raw row.
    - Each raw row can be used at most once.
    - Matching is allowed when |msm_template - msm_raw| <= tolerance_minutes.
    - Uses maximum cardinality bipartite matching per date (Kuhn/DFS).
    - Returns a combined DataFrame: template columns suffixed with "_tpl" and
      matched raw columns suffixed with "_raw". Unmatched raw fields are NaN.
    """
    # ---- Copy to avoid mutating inputs ----
    raw = raw_df.copy()
    tpl = template_df.copy()

    # ---- Date parsing (keep your formats) ----
    # If user didn't pass custom formats, use heuristics like your old function:
    if raw_time_formats is None:
        raw_time_formats = {}
    # Date parsing: attempt common formats used previously
    raw[date_col_raw] = pd.to_datetime(raw[date_col_raw], errors="coerce", utc=False)
    tpl[date_col_template] = pd.to_datetime(tpl[date_col_template], errors="coerce", utc=False)

    # Ensure time column exists in template (if in index, reset)
    if time_col_template not in tpl.columns and time_col_template in tpl.index.names:
        tpl = tpl.reset_index(level=time_col_template)

    # Decide raw time parsing based on file_type if time is not already datetime
    if not np.issubdtype(raw[time_col_raw].dtype, np.datetime64):
        if file_type == "A":
            raw_time_format = raw_time_formats.get("A", "%H:%M")
        elif file_type in ("B",):
            raw_time_format = raw_time_formats.get("B", "%H:%M:%S")
        elif file_type in ("C",):
            raw_time_format = raw_time_formats.get("C", "%H:%M:%S")
        else:
            raw_time_format = raw_time_formats.get("default", "%H:%M:%S")
        raw[time_col_raw] = pd.to_datetime(raw[time_col_raw], format=raw_time_format, errors="coerce")

    if not np.issubdtype(tpl[time_col_template].dtype, np.datetime64):
        tpl[time_col_template] = pd.to_datetime(tpl[time_col_template], format="%H:%M", errors="coerce")

    # ---- Compute msm as float minutes (preserves seconds) ----
    raw["_msm"] = _calc_msm(raw[time_col_raw])
    tpl["_msm"] = _calc_msm(tpl[time_col_template])

    # ---- Normalize dates (so matching is done per-calendar-day) ----
    raw["_day"] = pd.to_datetime(raw[date_col_raw]).dt.normalize()
    tpl["_day"] = pd.to_datetime(tpl[date_col_template]).dt.normalize()

    # ---- Drop invalid rows from template; keep them to append later ----
    tpl_valid = tpl.dropna(subset=["_msm", "_day"]).copy().reset_index(drop=True)
    tpl_invalid = tpl[tpl["_msm"].isna() | tpl["_day"].isna()].copy().reset_index(drop=True)

    # Raw drop rows without valid msm/day since they cannot be matched
    raw_valid = raw.dropna(subset=["_msm", "_day"]).copy().reset_index(drop=True)

    # ---- Prepare output storage ----
    matched_rows = []

    # iterate over all days present in template (only need to match dates in template)
    for current_day in tpl_valid["_day"].unique():
        tpl_day = tpl_valid[tpl_valid["_day"] == current_day].reset_index(drop=True)
        raw_day = raw_valid[raw_valid["_day"] == current_day].reset_index(drop=True)

        if tpl_day.shape[0] == 0:
            continue

        if raw_day.shape[0] == 0:
            # All templates unmatched for this day
            for i in range(len(tpl_day)):
                row_tpl = tpl_day.iloc[i].to_dict()
                # mark as unmatched: raw columns will be NaN after construction
                matched_rows.append({**{f"{k}_tpl": v for k, v in row_tpl.items()}, **{"_matched_raw_idx": pd.NA}})
            continue

        # Numeric arrays for fast search
        t_arr = tpl_day["_msm"].to_numpy()
        d_arr = raw_day["_msm"].to_numpy()

        # Build adjacency: template_idx -> list of raw_idx where abs diff <= tol
        edges = defaultdict(list)
        # d_arr sorted? ensure sorted for searchsorted. We'll sort raw_day and keep permutation map.
        order = np.argsort(d_arr)
        d_sorted = d_arr[order]
        # We must also remember mapping from sorted index -> raw_day index
        sorted_to_raw_idx = {s_idx: int(orig_idx) for s_idx, orig_idx in enumerate(order)}

        for ti, tval in enumerate(t_arr):
            left = np.searchsorted(d_sorted, tval - tolerance_minutes, side="left")
            right = np.searchsorted(d_sorted, tval + tolerance_minutes, side="right")
            for s_idx in range(left, right):
                raw_idx = sorted_to_raw_idx[s_idx]  # index in raw_day
                edges[ti].append(raw_idx)

        # Kuhn algorithm
        match_to_template = [-1] * len(raw_day)  # raw_idx -> template_idx or -1

        def _try_kuhn(v: int, visited: Sequence[bool]) -> bool:
            for raw_idx in edges.get(v, ()):
                if visited[raw_idx]:
                    continue
                visited[raw_idx] = True
                if match_to_template[raw_idx] == -1 or _try_kuhn(match_to_template[raw_idx], visited):
                    match_to_template[raw_idx] = v
                    return True
            return False

        # Run Kuhn for every template vertex
        for v in range(len(t_arr)):
            visited = [False] * len(raw_day)
            _try_kuhn(v, visited)

        # Invert match_to_template => template_idx -> raw_idx (or None)
        template_to_raw = {v: None for v in range(len(t_arr))}
        for r_idx, t_idx in enumerate(match_to_template):
            if t_idx != -1:
                template_to_raw[t_idx] = r_idx

        # Build matched_rows list of dicts
        for tpl_idx in range(len(t_arr)):
            tpl_row = tpl_day.iloc[tpl_idx].to_dict()
            matched_raw_idx = template_to_raw[tpl_idx]
            if matched_raw_idx is None:
                matched_rows.append({**{f"{k}_tpl": v for k, v in tpl_row.items()}, **{"_matched_raw_idx": pd.NA}})
            else:
                raw_row = raw_day.iloc[matched_raw_idx].to_dict()
                # prefix columns
                tpl_prefixed = {f"{k}_tpl": v for k, v in tpl_row.items()}
                raw_prefixed = {f"{k}_raw": v for k, v in raw_row.items()}
                # Also include matched raw index pointer (index in original raw_valid)
                raw_prefixed["_raw_matched_index"] = raw_day.index[matched_raw_idx]
                matched_rows.append({**tpl_prefixed, **raw_prefixed, "_matched_raw_idx": raw_day.index[matched_raw_idx]})

    # ---- Create DataFrame from matched_rows ----
    if len(matched_rows) > 0:
        matched_df = pd.DataFrame(matched_rows)
    else:
        matched_df = pd.DataFrame(columns=[*(f"{c}_tpl" for c in tpl_valid.columns), "_matched_raw_idx"])

    # ---- Attach unmatched fields for raw columns if they are missing ----
    # Ensure all raw columns with suffix _raw exist
    raw_cols = list(raw.columns)
    for c in raw_cols:
        pref = f"{c}_raw"
        if pref not in matched_df.columns:
            matched_df[pref] = pd.NA

    # ---- Append invalid template rows at the end (keep original columns with _tpl suffix) ----
    if not tpl_invalid.empty:
        tpl_invalid_prefixed = {f"{k}_tpl": tpl_invalid[k] for k in tpl_invalid.columns}
        invalid_rows = pd.DataFrame([{k: v for k, v in tpl_invalid_prefixed.items()}])
        # ensure same columns
        for col in matched_df.columns:
            if col not in invalid_rows.columns:
                invalid_rows[col] = pd.NA
        matched_df = pd.concat([matched_df, invalid_rows[matched_df.columns]], ignore_index=True, sort=False)

    # ---- Optional: compute difference column if both msm available ----
    # template msm column name:
    if "_msm_tpl" not in matched_df.columns and "_msm" in tpl_valid.columns:
        # If original added _msm with tpl suffix earlier it will be there; otherwise compute from tpl times
        pass

    # normalize column order a bit
    col_order = [c for c in matched_df.columns if c.endswith("_tpl")] + [c for c in matched_df.columns if c.endswith("_raw")] + [c for c in matched_df.columns if not (c.endswith("_tpl") or c.endswith("_raw"))]
    matched_df = matched_df.loc[:, col_order]

    return matched_df


def plot_matches_by_day(
    matched_df: pd.DataFrame,
    time_col_tpl: str = "_msm_tpl",
    time_col_raw: str = "_msm_raw",
    date_col_tpl: str = "date_tpl",
    figsize_per_plot: tuple = (8, 3),
    max_cols: int = 3,
    show_unmatched_raw: bool = True,
):
    """
    Create one subplot per date (template date) showing:
      - template times on y=1 (x = minutes since midnight)
      - raw times on y=0   (x = minutes since midnight)
      - lines between matched pairs
    matched_df should be the output of match_templates_to_raw_multi_day (it uses *_tpl and *_raw columns).
    If your dataframe uses different column names for msm columns, pass their names in time_col_tpl/time_col_raw.
    """
    # Determine dates present
    if date_col_tpl not in matched_df.columns:
        # try to find any column that looks like date_tpl
        possible = [c for c in matched_df.columns if c.endswith("_tpl") and "date" in c]
        if possible:
            date_col_tpl = possible[0]
        else:
            raise ValueError("Could not find a template date column in matched_df. Expected 'date_tpl' or similar.")

    # Build a column for the day normalized (if values are datetime)
    days = pd.to_datetime(matched_df[date_col_tpl]).dt.normalize()
    matched_df["_plot_day"] = days

    unique_days = matched_df["_plot_day"].dropna().unique()
    unique_days = np.sort(unique_days)

    if len(unique_days) == 0:
        raise ValueError("No valid dates found for plotting.")

    n_plots = len(unique_days)
    n_cols = min(max_cols, n_plots)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows), squeeze=False)
    axes = axes.flatten()

    for ax_idx, day in enumerate(unique_days):
        ax = axes[ax_idx]
        sub = matched_df[matched_df["_plot_day"] == day]

        # Template x (msm)
        if time_col_tpl not in sub.columns:
            # attempt to compute from template time column if present
            tpl_time_cols = [c for c in sub.columns if c.endswith("_tpl") and ("time" in c or "_msm" in c)]
            if len(tpl_time_cols) == 0:
                raise ValueError("Can't find template msm column for plotting.")
            time_col_tpl = tpl_time_cols[0]

        if time_col_raw not in sub.columns:
            # attempt to find a raw msm column
            raw_time_cols = [c for c in sub.columns if c.endswith("_raw") and ("time" in c or "_msm" in c)]
            if len(raw_time_cols) > 0:
                time_col_raw = raw_time_cols[0]

        x_tpl = sub[time_col_tpl].to_numpy(dtype=float)
        x_raw = sub[time_col_raw].to_numpy(dtype=float)

        # y positions
        y_tpl = np.ones_like(x_tpl) * 1.0
        y_raw = np.ones_like(x_raw) * 0.0

        # Scatter unmatched/matched differently:
        matched_mask = ~sub[time_col_raw].isna()
        unmatched_mask = sub[time_col_raw].isna()

        # Plot raw points at y=0 (colored by matched/unmatched)
        if show_unmatched_raw:
            # plot matched raw points
            ax.scatter(x_raw[matched_mask], y_raw[matched_mask], marker="o", label="raw (matched)", zorder=3)
            ax.scatter(x_raw[unmatched_mask], y_raw[unmatched_mask], marker="x", label="raw (unmatched)", alpha=0.4, zorder=2)
        else:
            ax.scatter(x_raw, y_raw, marker="o", label="raw", zorder=3)

        # Plot template points at y=1
        ax.scatter(x_tpl[matched_mask], y_tpl[matched_mask], marker="o", label="template (matched)", zorder=4)
        ax.scatter(x_tpl[unmatched_mask], y_tpl[unmatched_mask], marker="x", label="template (unmatched)", color="C3", zorder=2)

        # Draw lines for matched pairs
        for _, row in sub.iterrows():
            if pd.isna(row.get(time_col_raw)) or pd.isna(row.get(time_col_tpl)):
                continue
            xt = float(row[time_col_tpl])
            xr = float(row[time_col_raw])
            ax.plot([xt, xr], [1.0, 0.0], linewidth=0.8, alpha=0.7)

        ax.set_title(str(pd.to_datetime(day).date()))
        ax.set_ylim(-0.25, 1.25)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["raw", "template"])
        ax.set_xlabel("Minutes since midnight (msm)")
        ax.grid(axis="x", linestyle="--", linewidth=0.4)
        ax.legend(loc="upper right", fontsize="small")

    # hide any extra axes
    for i in range(len(unique_days), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()



'''
def compare_df_msm_aware(
    raw_df: pd.DataFrame,
    empty_df: pd.DataFrame,
    file_type: str,
    ind: int = 0,
    time_col_raw: str = "1_TIme",
    time_col_empty: str = "time",
) -> pd.DataFrame:
    """
    Align raw_df to empty_df by nearest time points (within tolerance),
    using 'minutes since midnight' (msm) for alignment.

    Handles:
      - Timezone mismatches
      - Unsorted merge keys
      - Empty merge subsets
      - German month abbreviations in date column
    """

    # --- Copy inputs to avoid mutation ---
    raw_df: pd.DataFrame = raw_df.copy()
    empty_df: pd.DataFrame = empty_df.copy()
    raw_df = raw_df.rename(columns={"created_at": "date"})

    #print('raw_df:', raw_df.head(10))
    #print('empty_df:', empty_df.head(10))

    date_col: str = "date"

    raw_df[date_col]: pd.Series = pd.to_datetime(
        raw_df[date_col], format="%Y-%m-%dT%H:%M:%S.%fZ", errors="coerce"
    )
    empty_df[date_col]: pd.Series = pd.to_datetime(
        empty_df[date_col], format="%d.%m.%Y", errors="coerce"
    )

    # --- Ensure time_col_empty exists ---
    if time_col_empty not in empty_df.columns:
        if time_col_empty in empty_df.index.names:
            empty_df: pd.DataFrame = empty_df.reset_index(level=time_col_empty)
        else:
            raise KeyError(f"'{time_col_empty}' not found in columns or index of empty_df")

    # --- Parse times ---
    if file_type == 'A':
        # Type A needs %H:%M (based on the screenshot)
        raw_time_format = "%H:%M"

    elif file_type == 'B':
        # You'll need to check the raw data for Type B,
        # but let's assume it's like C for now.
        raw_time_format = "%H:%M:%S"

    elif file_type == 'C':
        # Type C needs %H:%M:%S (based on previous success)
        raw_time_format = "%H:%M:%S"

    else:
        # Handle unexpected types or default
        raw_time_format = "%H:%M:%S"
        print(f"Warning: Unknown file type '{file_type}', defaulting to %H:%M:%S.")

    # --- Parse times ---
    raw_df[time_col_raw]: pd.Series = pd.to_datetime(
        raw_df[time_col_raw], format=raw_time_format, errors="coerce"  # Use the variable here
    )
    empty_df[time_col_empty]: pd.Series = pd.to_datetime(
        empty_df[time_col_empty], format="%H:%M", errors="coerce"
    )

    def msm(df: pd.DataFrame, col: str) -> pd.Series:
        """
        Calculate minutes since midnight for each timestamp in df[col].
        """
        td = df[col] - df[col].dt.normalize()  # timedelta since midnight
        return (td.dt.total_seconds() / 60)#.astype(int)

    raw_df["msm"]: pd.Series = msm(raw_df, time_col_raw)
    empty_df["msm"]: pd.Series = msm(empty_df, time_col_empty)

    raw_df["msm"]: pd.Series = raw_df["msm"].astype("Int64")
    empty_df["msm"]: pd.Series = empty_df["msm"].astype("Int64")

    # --- Filter valid rows ---
    valid_empty_df: pd.DataFrame = empty_df.dropna(subset=["msm", date_col]).copy()
    invalid_empty_df: pd.DataFrame = empty_df[empty_df["msm"].isna() | empty_df[date_col].isna()].copy()
    raw_df: pd.DataFrame = raw_df.dropna(subset=["msm", date_col]).copy()
    #print('raw_df:', raw_df.head(5))
    #print('invalid_empty_df: \n', invalid_empty_df.head(5))

    raw_df[date_col]: pd.Series = pd.to_datetime(raw_df[date_col]).dt.normalize()
    valid_empty_df[date_col]: pd.Series = pd.to_datetime(valid_empty_df[date_col]).dt.normalize()

    # --- Sort for merge_asof ---
    raw_df: pd.DataFrame = raw_df.sort_values("msm").reset_index(drop=True)
    valid_empty_df: pd.DataFrame = valid_empty_df.sort_values("msm").reset_index(drop=True)

    raw_df.columns = [
        col if col in ['msm', 'date'] else f"{col}_{ind}"
        for col in raw_df.columns
    ]

    # --- Merge ---
    merged_valid: pd.DataFrame = pd.merge_asof(
        valid_empty_df,
        raw_df,
        on="msm",
        by=date_col,
        tolerance=2,
        direction="nearest"
    )

    # --- Resolve duplicates ---
    cols_to_skip: set[str] = {time_col_empty, time_col_raw, "msm", "merge_date", date_col, date_col}
    common_cols: set[str] = set(raw_df.columns) & set(valid_empty_df.columns)

    for col in common_cols:
        if col not in cols_to_skip:
            col_x, col_y = f"{col}_x", f"{col}_y"
            if col_x in merged_valid.columns and col_y in merged_valid.columns:
                merged_valid[col]: pd.Series = merged_valid[col_y].combine_first(merged_valid[col_x])
                merged_valid.drop(columns=[col_x, col_y], inplace=True)

    merged_valid.drop(columns=["merge_date"], errors="ignore", inplace=True)

    final_merged: pd.DataFrame = pd.concat(
        [merged_valid, invalid_empty_df.drop(columns=["merge_date"], errors="ignore")],
        ignore_index=True,
    )

    return final_merged
'''


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