import pandas as pd
import os

def get_nearest_date(index, date):
    '''
    Returns the nearest date before the given date in the index.
    
    Args:
    - index (Index): Index of the DataFrame with datetime dates.
    - date (str or datetime): Date to find the nearest date before in the index.
    
    Returns:
    - datetime: The nearest date before the given date in the index.
    '''
    
    index_dates = pd.to_datetime(index)
    date = pd.to_datetime(date)
    nearest_date = index_dates[index_dates <= date].max()
    
    return nearest_date

def add_r1r2(df, r1=[1, 2, 3, 4, 9, 10, 11, 12], r2=[5, 6, 7, 8, 13, 14, 15, 16], both=[]):
    """
    Adds R1_Ph and R2_Ph columns (sequence of phases for each ring) to the DataFrame
    using vectorized operations.

    Args:
    - df (DataFrame): Input DataFrame containing traffic event data (TS_start, Code, ID, Cycle_start).
    - r1 (list): List of phases belonging to Ring 1.
    - r2 (list): List of phases belonging to Ring 2.
    - both (list): List of phases belonging to both rings (e.g., for split phasing).

    Returns:
    - DataFrame: Updated df with 'R1_Ph' and 'R2_Ph' columns.
    """

    # 1. Prepare the final ring lists including 'both' phases
    r1_final = list(set(r1 + both))
    r2_final = list(set(r2 + both))

    # 2. Filter the DataFrame to only Code 1 (Green Start) events
    df_ph = df[df.Code == 1].copy()

    # 3. Create a boolean mask indicating if the phase is in R1 or R2
    # Convert ID to string for concatenation later
    df_ph['ID_str'] = df_ph['ID'].astype(str)
    df_ph['in_R1'] = df_ph['ID'].isin(r1_final)
    df_ph['in_R2'] = df_ph['ID'].isin(r2_final)

    # 4. Define the vectorized logic to generate the comma-separated string for each group (Cycle_start)
    def create_ring_string(group):
        # Filter IDs that are in R1 for this Cycle_start
        r1_phases = group[group['in_R1']]['ID_str'].tolist()
        s1 = ','.join(r1_phases) if r1_phases else 'None'

        # Filter IDs that are in R2 for this Cycle_start
        r2_phases = group[group['in_R2']]['ID_str'].tolist()
        s2 = ','.join(r2_phases) if r2_phases else 'None'

        return pd.Series({'R1_Ph': s1, 'R2_Ph': s2})

    # 5. Apply the logic across all Cycle_start groups (This is the only 'apply'
    # but it operates on boolean masks and string joins, which is much faster than the old loop)
    df_res = (df_ph
              .groupby('Cycle_start')
              .apply(create_ring_string)
              .reset_index()
             )

    # 6. Merge the results back into the original DataFrame on 'Cycle_start'
    # Drop existing R1_Ph and R2_Ph columns if they exist to avoid conflicts during merge
    df = df.drop(columns=['R1_Ph', 'R2_Ph'], errors='ignore')
    df = pd.merge(df, df_res, on='Cycle_start', how='left')

    # 7. Forward fill the ring phases (important because Code 1 events don't happen every row)
    df.ffill(inplace=True)

    return df

def old_add_r1r2(df,r1=[1,2,3,4,9,10,11,12],r2=[5,6,7,8,13,14,15,16],both=[]):
    
    r1+=both
    r2+=both
    
    df_res = pd.DataFrame()

    loop = 0    
    for cs, dfgb in df[df.Code == 1].groupby('Cycle_start'):
        s1 = ''
        s2 = ''
        for n in dfgb.ID:
            if int(n) in r1:
                s1 += ',' + str(n)
            if int(n) in r2:
                s2 += ',' + str(n)
        s1 = s1[1:] if len(s1) > 0 else 'None'
        s2 = s2[1:] if len(s2) > 0 else 'None'

        df_res = pd.concat([df_res, pd.DataFrame([[s1, s2]],
                                                 index=[cs],
                                                 columns=['R1_Ph', 'R2_Ph'])])
    df_res = df_res.reset_index(names=['Cycle_start'])
    df = pd.merge(df, df_res, 'left', 'Cycle_start')
    
    return df

def comb_gyr_det(df_raw):
    '''
    Outputs df_comb, dataframe with green/yellow/red and detector events,
    durations, and time referenced to beginning of cycle.
    
    Args:
    - df_raw (DataFrame): Input dataframe.
    
    Returns:
    - df_comb (DataFrame): Processed dataframe.
    '''

    # GYR dataframe
    #df_gyr = df_raw.copy()[df_raw.loc[:, 'Code'].isin([1, 8, 9, 11])]
    #df_gyr.loc[:, 'TS_end'] = df_gyr.groupby('ID')['TS_start'].shift(-1)
    #df_gyr.loc[:, 't_cs'] = (df_gyr.TS_start - df_gyr.Cycle_start).dt.total_seconds()
    #df_gyr.loc[:, 't_ce'] = (df_gyr.TS_end - df_gyr.Cycle_start).dt.total_seconds()
    #df_gyr.loc[:, 'Duration'] = df_gyr.t_ce - df_gyr.t_cs
    #df_gyr.dropna(inplace=True)

    # Include code 12 in the initial filter
    df_gyr = df_raw.copy()[df_raw.loc[:, 'Code'].isin([1, 8, 9, 11, 12])]

    # First, apply the filtering logic within each ID-Cycle_start group
    def filter_group(group):
        # Check if code 11 exists in this group
        has_code_11 = 11 in group['Code'].values
        
        if has_code_11:
            # Drop code 12 rows and use code 11 for shift logic
            group = group[group['Code'] != 12]
        
        return group

    # Apply filtering within each Cycle_start group
    df_filtered = df_gyr.groupby(['ID', 'Cycle_start']).apply(filter_group).reset_index(drop=True)

    # Now sort by ID and TS_start to ensure proper chronological order across cycles
    df_filtered = df_filtered.sort_values(['ID', 'TS_start']).reset_index(drop=True)

    # Apply the shift logic across all events for each ID (not constrained by cycle)
    df_filtered['TS_end'] = df_filtered.groupby('ID')['TS_start'].shift(-1)

    # Continue with the remaining calculations
    df_filtered.loc[:, 't_cs'] = (df_filtered.TS_start - df_filtered.Cycle_start).dt.total_seconds()
    df_filtered.loc[:, 't_ce'] = (df_filtered.TS_end - df_filtered.Cycle_start).dt.total_seconds()
    df_filtered.loc[:, 'Duration'] = df_filtered.t_ce - df_filtered.t_cs
    df_filtered.dropna(inplace=True)

    # Sort final result by TS_start as desired
    df_filtered = df_filtered.sort_values('TS_start').reset_index(drop=True)

    df_gyr = df_filtered.copy()

    # Add Split column
    df_new = pd.DataFrame()
    for id in df_gyr.ID.unique():
        df = df_gyr.loc[(df_gyr.ID == id)]
        df = df.assign(Split=df['Duration'] + df['Duration'].shift(-1) + df['Duration'].shift(-2))
        #df.loc[:, 'Split'] = df.loc[:, 'Duration'] + df.loc[:, 'Duration'].shift(-1).copy() + df.loc[:, 'Duration'].shift(-2).copy()
        df.loc[df.Code != 1, 'Split'] = 0
        df_new = pd.concat([df_new, df])

    df_gyr = df_new.sort_values('TS_start')

    phase_key = [[1, 'G'],
                  [8, 'Y'],
                  [9, 'Rc'],
                  [11, 'R'],
                  [12, 'R']]

    overlap_key = [[61, 'G'],
                   [63, 'Y'],
                   [64, 'Rc'],
                   [65, 'R']]

    det_key = [[81, 'Off'],
                [82, 'On']]

    df_ph_stat = pd.DataFrame(phase_key, columns=['Code', 'Ph_Status']).set_index('Code')
    df_ol_stat = pd.DataFrame(overlap_key, columns=['Code', 'Ol_Status']).set_index('Code')
    df_det_stat = pd.DataFrame(det_key, columns=['Code', 'Det_Status']).set_index('Code')

    df_gyr = pd.merge(df_gyr, df_ph_stat, on='Code').sort_values('TS_start')
    
    # Overlap dataframe
    df_ol = df_raw.copy()[df_raw.Code.isin([61, 63, 64, 65])]
    df_ol.loc[:, 'TS_end'] = df_ol.groupby('ID')['TS_start'].shift(-1)
    df_ol.loc[:, 't_cs'] = (df_ol.TS_start - df_ol.Cycle_start).dt.total_seconds()
    df_ol.loc[:, 't_ce'] = (df_ol.TS_end - df_ol.Cycle_start).dt.total_seconds()
    df_ol.loc[:, 'Duration'] = df_ol.t_ce - df_ol.t_cs
    df_ol.dropna(inplace=True)

    df_ol = pd.merge(df_ol, df_ol_stat, on='Code').sort_values('TS_start')

    # Detector dataframe
    df_dt = df_raw.copy()[df_raw.Code.isin([81, 82])]
    df_dt.loc[:, 'TS_end'] = df_dt.groupby('ID')['TS_start'].shift(-1)
    df_dt.loc[:, 't_cs'] = (df_dt.TS_start - df_dt.Cycle_start).dt.total_seconds()
    df_dt.loc[:, 't_ce'] = (df_dt.TS_end - df_dt.Cycle_start).dt.total_seconds()
    df_dt.loc[:, 'Duration'] = (df_dt.TS_end - df_dt.TS_start).dt.total_seconds()
    df_dt.dropna(inplace=True)

    df_dt = pd.merge(df_dt, df_det_stat, on='Code').sort_values('TS_start')

    # Combine back into one dataframe
    df_comb = pd.concat([df_dt, df_gyr, df_ol]).sort_values('TS_start').reset_index(drop=True)

    # Cycle length
    df_comb = add_cyc_len(df_comb)

    return df_comb


def phase_status(df_comb, ph_list='all'):
    '''
    Adds columns with phase status (G, Y, Rc, R) to all events in df_comb.

    Args:
    - df_comb (DataFrame): Input DataFrame containing traffic event data.
    - ph_list (str or list): Phases to add columns. Default is 'all' to include all phases.

    Returns:
    - DataFrame: Updated df_comb with added phase status columns.
    '''

    if type(ph_list) == str:
        # Add phase status columns to detector events
        ph_list = df_comb[df_comb.loc[:, 'Code'].isin([1, 8, 9, 12])].ID.unique()

    ph_list.sort()
    for ph in ph_list:
        df_ph = df_comb[(df_comb.Code.isin([1, 8, 9, 12]) & (df_comb.ID == int(ph)))]
        df_ph = df_ph.assign(**{f'Ph {ph} Status': df_ph.sort_values('TS_start').Ph_Status})
        df_comb = pd.merge(df_comb, df_ph.loc[:, ['TS_start', f'Ph {ph} Status']], 'outer', on='TS_start').sort_values('TS_start')
    df_comb.ffill(inplace=True)

    return df_comb


def overlap_status(df_comb, ol_list='all'):
    '''
    Adds columns with overlap status (G, Y, Rc, R) to all events in df_comb.

    Args:
    - df_comb (DataFrame): Input DataFrame containing traffic event data.
    - ol_list (str or list): Overlaps to add columns. Default is 'all' to include all overlaps. 
    Accepts a list of integers or strings where 1 -> 'A', 2 -> 'B', ..., 16 -> 'P'.

    Returns:
    - DataFrame: Updated df_comb with added overlap status columns.
    '''

    if ol_list == 'all':
        # Add overlap status columns to detector events
        ol_list = df_comb[df_comb.loc[:, 'Code'].isin([61, 63, 64, 65])].ID.unique()

    # Normalize ol_list to integer IDs and keep mapping to letter for column names
    ol_ints = []
    ol_letters = []
    for ol in ol_list:
        try:
            ol_int = int(ol)
        except ValueError:
            # If ol is a letter, convert to int (A=1, B=2, ..., P=16)
            ol_int = ord(str(ol).upper()) - ord('A') + 1
        ol_letter = chr(ord('A') + ol_int - 1)
        ol_ints.append(ol_int)
        ol_letters.append(ol_letter)

    for ol_int, ol_letter in zip(ol_ints, ol_letters):
        df_ol = df_comb[(df_comb.Code.isin([61, 63, 64, 65]) & (df_comb.ID == ol_int))]
        df_ol = df_ol.assign(**{f'OL{ol_letter} Status': df_ol.sort_values('TS_start').Ol_Status})
        df_comb = pd.merge(df_comb, df_ol.loc[:, ['TS_start', f'OL{ol_letter} Status']], 'outer', on='TS_start').sort_values('TS_start')
    df_comb.ffill(inplace=True)

    return df_comb


def detector_status(df_comb, det_list='all'):
    '''
    Adds columns with detector status (On, Off) to all events in df_comb.

    Args:
    - df_comb (DataFrame): Input DataFrame containing traffic event data.
    - det_list (str or list): Detectors to add columns. Default is 'all' to include all detectors.

    Returns:
    - DataFrame: Updated df_comb with added detector status columns.
    '''

    if type(det_list) == str:
        # Add detector status columns to detector events
        det_list = df_comb[df_comb.loc[:, 'Code'].isin([81, 82])].ID.unique()

    det_list.sort()
    for det in det_list:
        df_det = df_comb[(df_comb.Code.isin([81, 82]) & (df_comb.ID == int(det)))]
        #df_det.loc[:, f'Det {det} Status'] = df_det.sort_values('TS_start').Det_Status
        df_det = df_det.assign(**{f'Det {det} Status': df_det.sort_values('TS_start').Det_Status})
        df_comb = pd.merge(df_comb, df_det.loc[:, ['TS_start', f'Det {det} Status']], 'outer', on='TS_start').sort_values('TS_start')
    df_comb.ffill(inplace=True)

    return df_comb

def filter_month(df, month, dt_col='Cycle_start'):
    '''
    Filters DataFrame by a specific month.
    
    Args:
    - df (DataFrame): Input DataFrame with datetime column.
    - month (int): Target month (1 to 12).
    - dt_col (str): Name of the datetime column in the DataFrame.
    
    Returns:
    - DataFrame: Filtered DataFrame for the specified month.
    '''
    df.loc[:, 'Month'] = pd.Series(pd.DatetimeIndex(df.loc[:, dt_col]).month, index=df.index)
    df = df[df.Month == month].drop('Month', axis=1).reset_index(drop=True)
    
    return df

def filter_date(df, begin, end, dt_col='Cycle_start', add_day=True):
    '''
    Filters DataFrame by a date range.
    
    Args:
    - df (DataFrame): Input DataFrame with datetime column.
    - begin (str): Start date of the window (e.g., '6-1-2023').
    - end (str): End date of the window (e.g., '6-15-2023').
    - dt_col (str): Name of the datetime column in the DataFrame.
    - add_day (bool): If True, the end time is set to '23:59:59.9'.
    
    Returns:
    - DataFrame: Filtered DataFrame for the specified date range.
    '''
    reindex = None

    if df.index.name == dt_col:
        df = df.reset_index()
        reindex = True

    if add_day:
        df = df[(df[dt_col] >= begin) & (df[dt_col] <= (str(end) + ' 23:59:59.9'))]
    else:
        df = df[(df[dt_col] >= begin) & (df[dt_col] <= end)]

    if reindex:
        df = df.set_index(dt_col)

    return df

def filter_time(df, start_time, stop_time, dt_col='Cycle_start'):
    '''
    Filters DataFrame by a time window.
    
    Args:
    - df (DataFrame): Input DataFrame with datetime column.
    - start_time (str): Start time of the window (e.g., '9:00').
    - stop_time (str): End time of the window (e.g., '17:00').
    - dt_col (str): Name of the datetime column in the DataFrame.
    
    Returns:
    - DataFrame: Filtered DataFrame for the specified time window.
    '''
    if df.index.name == dt_col:
        df = df.between_time(start_time, stop_time)
    else:
        df = df.set_index(dt_col).between_time(start_time, stop_time).reset_index()

    return df

def filter_day(df, days, dt_col='Cycle_start'):
    '''
    Filters DataFrame by days of the week.
    
    Args:
    - df (DataFrame): Input DataFrame with datetime column.
    - days (list): List of days to filter (e.g., ['Wednesday'], ['Saturday', 'Sunday']).
    - dt_col (str): Name of the datetime column in the DataFrame.
    
    Returns:
    - DataFrame: Filtered DataFrame for the specified days of the week.
    '''
    days_code = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
                 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

    days_n = [days_code[d] for d in days]

    df.loc[:, 'Day'] = pd.Series(pd.DatetimeIndex(df.loc[:, dt_col]).dayofweek, index=df.index)
    df = df.loc[df.Day.isin(days_n)]
    df = df.drop('Day', axis=1)

    return df

def add_cyc_len(df):
    """
    Adds a column for cycle length to the DataFrame.

    Args:
        df (DataFrame): Input DataFrame with 'Cycle_start' column.

    Returns:
        DataFrame: DataFrame with a single 'Cycle_len' column.
    """
    cl_key = pd.concat([
        pd.Series(df.Cycle_start.unique(), name='Cycle_start'),
        pd.Series(df.Cycle_start.unique(), name='Cycle_end').shift(-1)
    ], axis=1)

    cl_key['Cycle_len'] = (cl_key['Cycle_end'] - cl_key['Cycle_start']).dt.total_seconds()

    # Merge while avoiding duplicate Cycle_len columns
    df = df.drop(columns=['Cycle_len'], errors='ignore')  # ensure no existing one
    df = pd.merge(df, cl_key[['Cycle_start', 'Cycle_len']], on='Cycle_start', how='left')

    return df

def create_int_directory(dir_name, base_path):
    """
    Creates a directory structure for intersection data with required subdirectories and configuration file.
    
    Args:
        dir_name (str): Name of the main directory to create
        base_path (str): Path where the directory should be created
        
    Returns:
        str: Full path to the created directory
    """
    # Create full path
    full_path = os.path.join(base_path, "Intersections", dir_name)
    
    # Create main directory
    os.makedirs(full_path, exist_ok=True)
    
    # Create subdirectories
    subdirs = {
        'Data': {'DATZ': {}, 'CSV' : {}, 'DataFrames' : {}},  # Data subdirectories
        'Plotting': {},
        'Counts': {},
        'Configuration': {},
        'Video Processing': {}
    }
    
    # Create directory structure
    for subdir, nested_dirs in subdirs.items():
        subdir_path = os.path.join(full_path, subdir)
        os.makedirs(subdir_path, exist_ok=True)
        
        # Create nested directories
        for nested_dir in nested_dirs:
            nested_path = os.path.join(subdir_path, nested_dir)
            os.makedirs(nested_path, exist_ok=True)
    
    # Create int_cfg.csv in Configuration folder
    config_path = os.path.join(full_path, 'Configuration', 'int_cfg.csv')
    
    # Create column headers (1 through 10)
    columns = ['1/1/2020']
    
    # Create row indices
    movement_rows = [d + m for d in ['EB', 'WB', 'NB', 'SB'] for m in ['L', 'T', 'R']]
    additional_rows = [
        'Detector', 'Phase', 'Status',
        'P2 Arrival', 'P4 Arrival', 'P6 Arrival', 'P8 Arrival',
        'P1 Occupancy', 'P2 Occupancy', 'P3 Occupancy', 'P4 Occupancy',
        'P5 Occupancy', 'P6 Occupancy', 'P7 Occupancy', 'P8 Occupancy',
        'P1 Stop Bar', 'P2 Stop Bar', 'P3 Stop Bar', 'P4 Stop Bar',
        'P5 Stop Bar', 'P6 Stop Bar', 'P7 Stop Bar', 'P8 Stop Bar',
        'R1', 'R2', 'B'
    ]
    index = ['TM:'] * 12 + ['Exc:'] * 3 + ['Plt:'] * 20 + ['RB:'] * 3
    
    # Create DataFrame with multi-level index
    df = pd.DataFrame('', index=pd.MultiIndex.from_tuples(zip(index, movement_rows + additional_rows)), columns=columns)
    
    # Save to CSV
    df.to_csv(config_path)