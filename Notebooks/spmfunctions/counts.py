import pandas as pd
from .misc_tools import comb_gyr_det, phase_status
import os

def cyc_counts(df, bin_len=60):
    '''
    Outputs dataframe of the number of cycles per hour.

    Args:
    - df (DataFrame): Input DataFrame containing cycle start times.
    - bin_len (int): Length in minutes of bins for counts. Default is 60.

    Returns:
    - DataFrame: Number of cycles per hour.
    '''

    df.loc[:, 'Hour'] = pd.Series(pd.DatetimeIndex(df.Cycle_start).hour, index=df.index)
    df = df.loc[:, ['Cycle_start', 'Hour']].groupby('Cycle_start').min()
    df = df.resample(f'{bin_len}T').count()

    return df


def det_counts(df_raw, int_cfg, bin_len=60, hourly=False, ped=True, cnt_dir=None):
    '''
    Outputs detector counts and movement counts based on the detector configuration file.

    Args:
    - df_raw (DataFrame): Input DataFrame containing raw traffic data.
    - int_cfg (dict): Dictionary containing intersection configuration.
    - bin_len (int): Length, in minutes, of bins. Default is 60.
    - hourly (bool): If True, convert counts to hourly flow rate. Default is False.
    - ped (bool): If True, include pedestrian counts. Default is True.
    - cnt_dir (str): Path to directory to store output. If provided, will write to file.

    Returns:
    - DataFrame: Detector and movement counts OR writes to file and returns None if cnt_dir provided.
    '''

    # Read in detector configuration
    df_mvmt = int_cfg['Movements']
    df_exc = int_cfg['Exclusions']

    # Process exclusions
    exc_ph = set()
    for idx, row in df_exc.iterrows():
        exc_ph.update(map(int, row['Phase'].split(',')))
    exc_ph = list(exc_ph)

    if exc_ph:
        df = phase_status(comb_gyr_det(df_raw), exc_ph)
    else:
        df = df_raw

    # Initiate count DataFrame
    df_cnt = df[(df.Code == 82)].copy()

    if hourly:
        hourly_factor = 60 / bin_len
    else:
        hourly_factor = 1

    # Exclude detector events per detector configuration
    for idx, row in df_exc.iterrows():
        next_idx = df_exc.index[df_exc.index.get_loc(idx) + 1] if df_exc.index.get_loc(idx) + 1 < len(df_exc.index) else df_cnt.index.max()
        det_list = list(map(int, row['Detector'].split(',')))
        ph_list = list(map(int, row['Phase'].split(',')))
        st_list = row['Status'].split(',')

        for det, ph, st in zip(det_list, ph_list, st_list):
            idx_ts = pd.Timestamp(idx)
            next_idx_ts = pd.Timestamp(next_idx)
            df_cnt.set_index('TS_start', inplace=True)
            df_cnt.index = pd.to_datetime(df_cnt.index)
            df_cnt.drop(df_cnt[(df_cnt.index >= idx_ts) & (df_cnt.index < next_idx_ts) & (df_cnt.ID == det) & (df_cnt[f'Ph {ph} Status'] == st)].index, inplace=True)
            df_cnt.reset_index(inplace=True)

    if bin_len == 'cycle':
        hourly_factor = 1
        df_cnt = df_cnt.loc[:, ['Cycle_start', 'ID', 'Code']].groupby(['Cycle_start', 'ID']).count().unstack()
        df_cnt.columns = df_cnt.columns.droplevel()
    else:
        # Groupby detector and resample to bins
        df_cnt = df_cnt.loc[:, ['TS_start', 'Code', 'ID']].set_index('TS_start').groupby('ID')
        df_cnt = df_cnt.resample(f'{bin_len}min').count()
        df_cnt = df_cnt.loc[:, 'Code'].unstack(level='ID')

    # Sum detectors to movements per detector configuration
    original_index = df_cnt.index.copy()
    df_cnt = df_cnt.join(df_mvmt, how='outer')
    df_mvmt_cols = df_mvmt.columns

    # Forward fill only the columns from df_mvmt
    df_cnt[df_mvmt_cols] = df_cnt[df_mvmt_cols].ffill()
    df_cnt = df_cnt.loc[original_index]

    def unpack_and_sum(row):
        for col in df_mvmt_cols:
            dets = []
            for val in str(row[col]).split(','):
                val = str(val) if val.isdigit() else val
                if val.isdigit() and int(val) in df_cnt.columns:
                    dets.append(int(val))
            row[col] = df_cnt.loc[row.name, dets].sum() * hourly_factor
        return row

    df_cnt = df_cnt.apply(unpack_and_sum, axis=1)

    # Add total entering vehicles 'TEV', Hour, and Minute columns to DataFrame
    df_cnt.loc[:, 'TEV'] = df_cnt.loc[:, df_mvmt.columns].sum(axis=1)
    df_cnt.index.name = 'Time'

    if ped:
        df_pedct = ped_counts(df_raw, bin_len, hourly)
        if not df_pedct.empty:
            df_cnt = pd.merge(df_cnt, df_pedct, how='outer', on='Time')

    if cnt_dir:
        # Format filename and filepath
        start_date = df_cnt.index.min().strftime('%Y_%m_%d')
        end_date = df_cnt.index.max().strftime('%Y_%m_%d')
        bin_str = 'Cycle' if bin_len == 'cycle' else f'{bin_len}min'
        hrly_str = 'hrly_' if hourly else ''
        filename = f"Counts_{bin_str}{hrly_str}_{start_date}-{end_date}.csv"
        filepath = os.path.join(cnt_dir, filename)

        df_cnt.to_csv(filepath)
        return None

    else:
        return df_cnt


def ped_counts_old(df_raw, bin_len=60, hourly=False):
    '''
    Outputs dataframe with counts of pedestrian actuations by phase per time bin.

    Args:
    - df_raw (DataFrame): Input DataFrame containing raw traffic data.
    - bin_len (int or str): Length, in minutes, of bins. Default is 60. If, 'cycle',
          count will be on cycle-by-cycle basis.
    - hourly (bool): If True, reports binned data in hourly rate. Default is False.

    Returns:
    - DataFrame: Counts of pedestrian actuations by phase per time bin.
    '''

    df_p = df_raw.loc[(df_raw.Code == 45) | (df_raw.Code == 21)]
    if len(df_p) == 0:
        return pd.DataFrame()

    df_p.loc[:, 'P_start'] = df_p.loc[df_p.Code == 21].TS_start
    df_gb = df_p.groupby('ID')
    df_p.loc[:, 'P_start'] = df_gb['P_start'].bfill()
    df_unq = df_p.loc[df_p.Code == 45, ['Code', 'ID', 'P_start']].drop_duplicates().index

    if hourly:
        hourly_factor = 60 / bin_len
    else:
        hourly_factor = 1
    
    if bin_len == 'cycle':
        df_p = df_p.loc[df_unq, ['Cycle_start', 'Code', 'ID']].groupby(['Cycle_start','ID']).count().unstack()
        df_p.columns = df_p.columns.droplevel()        
    else:
        # Groupby detector and resample to bins
        df_p = df_p.loc[df_unq,['TS_start','Code','ID']].set_index('TS_start').groupby('ID')
        df_p = df_p.resample(f'{bin_len}T').count()
        df_p = df_p.loc[:, 'Code'].unstack(level='ID')
    
    df_p = df_p.fillna(0).astype('int')

    # Add total column to DataFrame
    df_p.loc[:, 'Total'] = df_p.sum(axis=1)
    df_p.index.name = 'Time'
    df_p.rename(columns=dict([[c, f'Ped {c}'] for c in df_p.columns]), inplace=True)

    return df_p


def ped_counts(df_raw, bin_len=60, hourly=False):
    """
    Count the number of pedestrian phases served (code 21) that were
    preceded by a pedestrian call (code 45) for the same ID.

    Args:
        df_raw (pd.DataFrame): Input dataframe with columns at least ['TS_start', 'Cycle_start', 'Code', 'ID'].
        bin_len (int or str): Bin length in minutes, or 'cycle' for per-cycle counts.
        hourly (bool): If True, output counts scaled to hourly rate.

    Returns:
        pd.DataFrame: Counts of legitimate ped actuations by phase and time bin/cycle.
    """

    df_p = df_raw.loc[df_raw['Code'].isin([21, 45])].copy()
    if df_p.empty:
        return pd.DataFrame()

    df_p = df_p.sort_values(['ID', 'TS_start'])

    # Determine which Code 21 events were preceded by a Code 45 since the last Code 21
    legit_21 = []
    for pid, grp in df_p.groupby('ID'):
        last_21_time = None
        seen_call = False

        for _, row in grp.iterrows():
            if row['Code'] == 45:
                seen_call = True
            elif row['Code'] == 21:
                if seen_call:
                    legit_21.append(row.name)
                seen_call = False
                last_21_time = row['TS_start']

    df_legit_21 = df_p.loc[legit_21]

    if hourly:
        hourly_factor = 60 / bin_len if bin_len != 'cycle' else 1
    else:
        hourly_factor = 1

    if bin_len == 'cycle':
        # Count per cycle where service occurred
        df_counts = (df_legit_21.groupby(['Cycle_start', 'ID'])
                                .size()
                                .unstack(fill_value=0))
    else:
        # Resample by time bins based on service time
        df_counts = (df_legit_21.set_index('TS_start')
                                   .groupby('ID')
                                   .resample(f'{bin_len}T')
                                   .size()
                                   .unstack(level='ID')
                                   .fillna(0)
                                   .astype(int))
    
    # Scale to hourly if needed
    if hourly:
        df_counts = df_counts * hourly_factor

    # Add total column
    df_counts['Total'] = df_counts.sum(axis=1)
    df_counts.index.name = 'Time'
    df_counts.rename(columns={c: f'Ped {c}' for c in df_counts.columns}, inplace=True)

    return df_counts
