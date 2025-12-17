import pandas as pd
import os
import datetime as dt
#from statsmodels.stats.api import anova_lm
#from statsmodels.formula.api import ols
import bokeh
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
from bokeh.palettes import Viridis256
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


#---------------------------------------------------------------------------------------------------

def read_det_cfg(path_det, f_det='dets.csv'):
    '''
    Reads detector configuration from a CSV file.
    
    Args:
    - path_det (str): Path to the directory containing the detector configuration file.
    - f_det (str): Filename of the detector configuration CSV file.
    
    Returns:
    - List: Contains DataFrames for movement, exclusion, arrival, and stopbar configurations.
    '''
    f_path = os.path.join(path_det, f_det)
    df_dk = pd.read_csv(f_path, index_col=0)

    # Extract different configurations
    df_mvmt = df_dk.copy().iloc[:12].dropna(axis=1, how='all')  # Movement configuration
    df_exc = df_dk.copy().iloc[13:16].dropna(axis=1)  # Exclusion configuration
    df_adv = df_dk.copy().iloc[18:22].dropna(axis=1, how='all')  # Arrivals configuration
    df_sb = df_dk.copy().iloc[24:32].dropna(axis=1, how='all')  # Stopbar configuration
    
    return [df_mvmt, df_exc, df_adv.dropna(how='all'), df_sb.dropna(how='all')]

#---------------------------------------------------------------------------------------------------

def count_by_hour(df_p):
    '''
    Counts the number of occurrences in a DataFrame by hour.
    
    Args:
    - df_p (DataFrame): Input DataFrame with a datetime index.
    
    Returns:
    - DataFrame: Count of occurrences by hour.
    '''
    df_p.loc[:, 'Hour'] = pd.Series(df_p.index.hour, index=df_p.index)
    return df_p.groupby('Hour').count()

#---------------------------------------------------------------------------------------------------

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

#---------------------------------------------------------------------------------------------------

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

#---------------------------------------------------------------------------------------------------

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

#---------------------------------------------------------------------------------------------------

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

#---------------------------------------------------------------------------------------------------

def add_cyc_len(df):
    '''
    Adds a column for cycle length to the DataFrame.
    
    Args:
    - df (DataFrame): Input DataFrame with 'Cycle_start' column.
    
    Returns:
    - DataFrame: DataFrame with an additional 'Cycle_len' column.
    '''
    cl_key = pd.concat([pd.Series(df.Cycle_start.unique(), name='Cycle_start'),
                        pd.Series(df.Cycle_start.unique(), name='Cycle_end').shift(-1)],
                       axis=1)
    cl_key.loc[:, 'Cycle_len'] = (cl_key.Cycle_end - cl_key.Cycle_start).dt.total_seconds()

    df = pd.merge(df, cl_key.loc[:, ['Cycle_start', 'Cycle_len']], on='Cycle_start')

    return df

#---------------------------------------------------------------------------------------------------

def read_df_raw(f_path):
    '''
    Reads data from csv into dataframe and outputs df_raw.
    
    Args:
    - f_path (str): Path, including filename, of "compiled" CSV. 
      If ACHD, should have 4 columns (TS, Code, Description, and ID),
      and if ITD, should have 3. Neither should have headers.
      
    Returns:
    - df_raw (DataFrame): Processed dataframe.
    '''

    # Check if the file path indicates ACHD data
    if 'ACHD' in f_path:
        # Read CSV with specific columns and without headers
        df_raw = pd.read_csv(f_path, usecols=[0, 1, 3], 
                             names=['TS_start', 'Code', 'ID'], 
                             header=None)\
                   .drop_duplicates()
        
        # Convert 'TS_start' column to datetime
        df_raw['TS_start'] = pd.to_datetime(df_raw['TS_start'], 
                                            infer_datetime_format=True)

        # Sort, drop duplicates, and reset index
        df_raw = df_raw.sort_values(['TS_start', 'Code', 'ID'])\
                       .drop_duplicates()\
                       .reset_index(drop=True)

        # Identify rows of last barrier
        r1=[1,2,5,6]
        r2=[3,4,7,8]        
        
        #idx_bt = df_raw.loc[(df_raw.Code == 1) | 
        #                    ((df_raw.Code == 31) & (df_raw.ID == 1))]
        #idx_bt = idx_bt[(idx_bt.Code == 1) & 
        #                (idx_bt.Code.shift(1, fill_value=0) == 31) & 
        #                (idx_bt.ID.shift(2).isin(r2))]

        dfg = df_raw[df_raw.Code==1].groupby('TS_start').apply(lambda x: list(x.ID))
        idx_bt = dfg[(dfg.apply(lambda x: all(elem in r1 for elem in x)))&
                    (dfg.shift().apply(lambda x: all(elem in r2 for elem in x) if isinstance(x, list) else False))]

        idx_bt = idx_bt.reset_index().drop([0], axis=1)        

    else:
        # Read CSV with specific columns and without headers
        df_raw = pd.read_csv(f_path, names=['TS_start', 'Code', 'ID'], header=None).drop_duplicates()

        # Convert 'TS_start' column to datetime
        df_raw['TS_start'] = pd.to_datetime(df_raw['TS_start'], infer_datetime_format=True)

        # Sort, drop duplicates, and reset index
        df_raw = df_raw.sort_values(['TS_start', 'Code', 'ID'])\
                       .drop_duplicates()\
                       .reset_index(drop=True)

        # Identify last barrier
        idx_bt = df_raw.loc[(df_raw.Code == 31) & 
                            (df_raw.ID == df_raw[df_raw.Code == 31].ID.max())]

    # Format Timestamp column and add column for when each cycle starts
    idx_bt.loc[:, 'Cycle_start'] = idx_bt.loc[:, 'TS_start'].copy()
    df_raw = pd.merge(df_raw, idx_bt.loc[:, ['TS_start', 'Cycle_start']], 
                      on='TS_start', how='left')\
               .sort_values(['TS_start', 'Code', 'ID'])\
               .ffill()

    # Coord plan
    df_cp = df_raw.loc[df_raw.Code == 131, ['Cycle_start', 'ID']]
    df_cp = df_cp.rename({'ID': 'Coord_plan'}, axis=1)

    df_raw = pd.merge(df_raw, df_cp, how='left', on='Cycle_start')
    df_raw = df_raw.ffill()
    df_raw.loc[:, 'Coord_plan'] = df_raw.Coord_plan.fillna(0)

    return df_raw.dropna().sort_values('TS_start').reset_index(drop=True)

#---------------------------------------------------------------------------------------------------

def fix_df_raw_ACHD(df_raw):
    '''
    Starting with df_raw with TS_start already converted to datetime,
    drops extra columns and repeats read_df_raw starting with adding Cycle_start column.
    
    Args:
    - df_raw (DataFrame): Initial dataframe.
    
    Returns:
    - df_raw (DataFrame): Processed dataframe.
    '''

    # Keep only specific columns and drop duplicates
    df_raw = df_raw.loc[:, ['TS_start', 'Code', 'ID']].drop_duplicates()

    # Sort, drop duplicates, and reset index
    df_raw = df_raw.sort_values(['TS_start', 'Code', 'ID']).reset_index(drop=True)

    # Identify specific rows in the ACHD case
    idx_bt = df_raw.loc[(df_raw.Code == 1) | ((df_raw.Code == 31) & (df_raw.ID == 1))]
    idx_bt = idx_bt[(idx_bt.Code == 1) & (idx_bt.Code.shift(1, fill_value=0) == 31) & (idx_bt.ID.shift(2).isin([3, 4, 7, 8]))]

    # Format Timestamp column and add column for when each cycle starts
    idx_bt.loc[:, 'Cycle_start'] = idx_bt.loc[:, 'TS_start']
    df_raw = pd.merge(df_raw, idx_bt.loc[:, ['TS_start', 'Cycle_start']], 
                      on='TS_start', how='left')\
               .sort_values(['TS_start', 'Code', 'ID'])\
               .ffill()

    # Coord plan
    df_cp = df_raw.loc[df_raw.Code == 131, ['Cycle_start', 'ID']]
    df_cp = df_cp.rename({'ID': 'Coord_plan'}, axis=1)

    df_raw = pd.merge(df_raw, df_cp, how='left', on='Cycle_start')
    df_raw = df_raw.ffill()
    df_raw.loc[:, 'Coord_plan'] = df_raw.Coord_plan.fillna(0)

    return df_raw.dropna().reset_index(drop=True)

#---------------------------------------------------------------------------------------------------

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
    df_gyr = df_raw.copy()[df_raw.loc[:, 'Code'].isin([1, 8, 10, 11])]
    df_gyr.loc[:, 'TS_end'] = df_gyr.groupby('ID')['TS_start'].shift(-1)
    df_gyr.loc[:, 't_cs'] = (df_gyr.TS_start - df_gyr.Cycle_start).dt.total_seconds()
    df_gyr.loc[:, 't_ce'] = (df_gyr.TS_end - df_gyr.Cycle_start).dt.total_seconds()
    df_gyr.loc[:, 'Duration'] = df_gyr.t_ce - df_gyr.t_cs
    df_gyr.dropna(inplace=True)

    # Add Split column
    df_new = pd.DataFrame()
    for id in df_gyr.ID.unique():
        df = df_gyr.loc[(df_gyr.ID == id)]
        df.loc[:, 'Split'] = df.loc[:, 'Duration'] + df.loc[:, 'Duration'].shift(-1) + df.loc[:, 'Duration'].shift(-2)
        df.loc[df.Code != 1, 'Split'] = 0
        df_new = pd.concat([df_new, df])

    df_gyr = df_new.sort_values('TS_start')

    status_key = [[1, 'G'],
                  [8, 'Y'],
                  [10, 'Rc'],
                  [11, 'R']]

    det_key = [[81, 'Off'],
               [82, 'On']]

    df_ph_stat = pd.DataFrame(status_key, columns=['Code', 'Ph_Status']).set_index('Code')
    df_det_stat = pd.DataFrame(det_key, columns=['Code', 'Det_Status']).set_index('Code')

    df_gyr = pd.merge(df_gyr, df_ph_stat, on='Code').sort_values('TS_start')

    # Detector dataframe
    df_dt = df_raw.copy()[df_raw.Code.isin([81, 82])]
    df_dt.loc[:, 'TS_end'] = df_dt.groupby('ID')['TS_start'].shift(-1)
    df_dt.loc[:, 't_cs'] = (df_dt.TS_start - df_dt.Cycle_start).dt.total_seconds()
    df_dt.loc[:, 't_ce'] = (df_dt.TS_end - df_dt.Cycle_start).dt.total_seconds()
    df_dt.loc[:, 'Duration'] = (df_dt.TS_end - df_dt.TS_start).dt.total_seconds()
    df_dt.dropna(inplace=True)

    df_dt = pd.merge(df_dt, df_det_stat, on='Code').sort_values('TS_start')

    # Combine back into one dataframe
    df_comb = pd.concat([df_dt, df_gyr]).sort_values('TS_start').reset_index(drop=True)

    # Cycle length
    df_comb = add_cyc_len(df_comb)

    return df_comb

#---------------------------------------------------------------------------------------------------

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
        ph_list = df_comb[df_comb.loc[:, 'Code'].isin([1, 8, 10, 11])].ID.unique()

    ph_list.sort()
    for ph in ph_list:
        df_ph = df_comb[(df_comb.Code.isin([1, 8, 10, 11]) & (df_comb.ID == int(ph)))]
        df_ph.loc[:, f'Ph {ph} Status'] = df_ph.sort_values('TS_start').Ph_Status
        df_comb = pd.merge(df_comb, df_ph.loc[:, ['TS_start', f'Ph {ph} Status']], 'outer', on='TS_start').sort_values('TS_start')
    df_comb.ffill(inplace=True)

    return df_comb

#---------------------------------------------------------------------------------------------------

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
        df_det.loc[:, f'Det {det} Status'] = df_det.sort_values('TS_start').Det_Status
        df_comb = pd.merge(df_comb, df_det.loc[:, ['TS_start', f'Det {det} Status']], 'outer', on='TS_start').sort_values('TS_start')
    df_comb.ffill(inplace=True)

    return df_comb

#---------------------------------------------------------------------------------------------------

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

#---------------------------------------------------------------------------------------------------

def det_counts(df_raw, path_det, f_det='dets.csv', bin_len=60, hourly=False, ped=True):
    '''
    Outputs detector counts and movement counts based on the detector configuration file.

    Args:
    - df_raw (DataFrame): Input DataFrame containing raw traffic data.
    - path_det (str): Path to the detector configuration file.
    - f_det (str): Filename of the detector configuration CSV file. Default is 'dets.csv'.
    - bin_len (int): Length, in minutes, of bins. Default is 60.
    - hourly (bool): If True, convert counts to hourly flow rate. Default is False.
    - ped (bool): If True, include pedestrian counts. Default is True.

    Returns:
    - DataFrame: Detector and movement counts.
    '''

    # Read in detector configuration
    df_mvmt, df_exc, df_adv, df_sb = read_det_cfg(path_det, f_det)

    # If exclude phases, generate df_comb with detector status columns
    exc_ph = (df_exc.loc['Phase'].unique())
    if len(exc_ph) > 0:
        df = phase_status(comb_gyr_det(df_raw), exc_ph)
    else:
        df = df_raw

    # Initiate count DataFrame
    df_cnt = df[(df.Code == 82)]

    if hourly:
        hourly_factor = 60 / bin_len
    else:
        hourly_factor = 1

    # Exclude detector events per detector configuration
    for col in df_exc.columns:
        det = int(df_exc.loc['Detector', col])
        ph = int(df_exc.loc['Phase', col])
        st = df_exc.loc['Status', col].split(',')

        df_cnt.drop(df_cnt[(df.ID == det) & (df_cnt[f'Ph {ph} Status'].isin(st))].index, inplace=True)

        
    if bin_len=='cycle':
        hourly_factor = 1
        df_cnt = df_cnt.loc[:,['Cycle_start','ID','Code']].groupby(['Cycle_start','ID']).count().unstack()
        df_cnt.columns = df_cnt.columns.droplevel()
        
    else:    
        # Groupby detector and resample to bins
        df_cnt = df_cnt.loc[:, ['TS_start', 'Code', 'ID']].set_index('TS_start').groupby('ID')
        df_cnt = df_cnt.resample(f'{bin_len}T').count()
        df_cnt = df_cnt.loc[:, 'Code'].unstack(level='ID')
    


    # Sum detectors to movements per detector configuration
    for mvmt, row in df_mvmt.iterrows():
        dets = [x for x in row.dropna().astype(int).tolist() if x in df_cnt.columns]
        df_cnt.loc[:, mvmt] = df_cnt.loc[:, dets].sum(axis=1).astype(int) * hourly_factor

    # Add total entering vehicles 'TEV', Hour, and Minute columns to DataFrame
    df_cnt.loc[:, 'TEV'] = df_cnt.loc[:, df_mvmt.index.tolist()].sum(axis=1)
    df_cnt.index.name = 'Time'

    df_cnt.rename(columns=dict([[c, f'Det {c}'] for c in df_cnt.columns[:-13]]), inplace=True)

    if ped:
        df_pedct = ped_counts(df_raw, bin_len, hourly)
        if not df_pedct.empty:
            df_cnt = pd.merge(df_cnt, df_pedct, how='outer', on='Time')

    return df_cnt

#---------------------------------------------------------------------------------------------------

def ped_counts(df_raw, bin_len=60, hourly=False):
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

#---------------------------------------------------------------------------------------------------

def add_r1r2(df,r1=[1,2,3,4],r2=[5,6,7,8],both=[]):
    
    r1+=both
    r2+=both
    
    df_res = pd.DataFrame()

    loop = 0    
    for cs, dfgb in df[df.Code == 1].groupby('Cycle_start'):
        s1 = ''
        s2 = ''
        for n in dfgb.ID:
            if int(n) in r1:
                s1 += str(n)
            if int(n) in r2:
                s2 += str(n)

        df_res = pd.concat([df_res, pd.DataFrame([[s1, s2]],
                                                 index=[cs],
                                                 columns=['R1_Ph', 'R2_Ph'])])
    df_res = df_res.reset_index(names=['Cycle_start'])
    df = pd.merge(df, df_res, 'left', 'Cycle_start')
    
    return df


#---------------------------------------------------------------------------------------------------


def ring_dfs(df_comb,both=[]):
    '''
    Generates ring dataframes based on the input traffic events dataframe.

    Args:
    - df_comb (DataFrame): Input DataFrame containing traffic event data.
    - both (list): List of phases to be included in both rings, e.g. for split phasing. Default is empty list.

    Returns:
    - dict: Dictionary containing ring dataframes for Ring 1 and Ring 2.
    '''

    df_res = pd.DataFrame(columns=['Cycle_start'])
    df_comb = add_r1r2(df_comb,both=both)

    resdict = {}
    for r in [1, 2]:
        seq = df_comb.loc[:, f'R{r}_Ph'].unique()
        df_seq = pd.DataFrame([list(x) for x in seq], index=seq)
        df_seq.columns = [f' ({x})' for x in range(1, len(df_seq.columns) + 1)]
        df_seq = df_seq.astype('str') + df_seq.columns
        cols = []
        for c in df_seq.columns:
            for x in (df_seq.loc[:, c].dropna().unique()):
                for y in ['G', 'Y', 'Rc']:
                    if 'None' not in x:
                        cols.append(f'{y} {x}')

        df_ring = pd.DataFrame(columns=cols)

        for seq, row in df_seq.iterrows():
            idx = []
            for v in row:
                if v.startswith('None'):
                    continue
                for st in ['G', 'Y', 'Rc']:
                    idx.append(f'{st} {v}')

            df = df_comb[(df_comb.ID.isin([int(x) for x in seq])) & (df_comb.Code.isin([1, 8, 10])) &
                         (df_comb.loc[:, f'R{r}_Ph'] == seq)]

            for cs, dfgb in df.groupby('Cycle_start'):
                srs = dfgb.Duration
                if len(srs) != len(idx):
                    continue
                srs.index = idx
                srs.name = cs
                df_ring = pd.concat([df_ring, pd.DataFrame(srs).T])

        df_ring = df_ring.fillna(0)
        df_ring.index.name = 'Cycle_start'
        df_ring = pd.merge(df_ring, df_comb.loc[:, ['Cycle_start', 'Coord_plan']], how='left', on='Cycle_start').drop_duplicates()
        df_ring.loc[:, 'Coord_plan'] = df_ring.Coord_plan

        resdict[r] = df_ring

    return resdict

#---------------------------------------------------------------------------------------------------

def plot_coord(ring_dict,df_comb,det_path,path_out,sfx='',plt='plotly'):
    if plt=='plotly':
        plot_coord_plotly(ring_dict,df_comb,det_path,path_out,sfx=sfx)
    elif plt=='bokeh':
        plot_coord_bokeh(ring_dict,df_comb,det_path,path_out,sfx=sfx)
        
#---------------------------------------------------------------------------------------------------    
    
def plot_term(df,save_path,intx,line=True,n_con=10,plt='plotly'):
    if plt=='plotly':
        plot_term_plotly(df,save_path,intx,line=line,n_con=n_con)
    elif plt=='bokeh':
        plot_term_bokeh(df,save_path,intx,line=line,n_con=n_con)
    
  
#---------------------------------------------------------------------------------------------------
def plot_coord_plotly(ring_dict, df_comb, det_path, path_out, sfx=''):
    '''
   Plots coordination/split diagram.

   Args:
   - ring_dict (dict): Dictionary containing ring dataframes for Ring 1 and Ring 2.
    - df_comb (DataFrame): Input DataFrame containing traffic event data.
    - det_path (str): Path to detector configuration file.
    - path_out (str): Output path for the plot.
    - sfx (str): Suffix for the output filename.
    '''

    figs = []
    #plot_dts = {1: [53, 54, 61, 62], 2: [49, 50, 57, 58]}

    cl_RY = {'Rc': 'Red', 'Y': 'Yellow'}
    cl_P = {'1': 'DarkOrange', '5': 'LightSalmon', '3': 'Magenta', '7': 'Purple',
            '2': 'Blue', '6': 'Turquoise', '4': 'DarkGreen', '8': 'Lime'}

    #leg = {1: [1, 2, 3, 4, 0], 2: [5, 6, 7, 8, 0]}

    df_dtcfg = pd.concat(read_det_cfg(os.path.join(det_path))[2:])

    for r in [1, 2]:
        df_plt = ring_dict[r]
        df_plt.loc[:, 'y'] = -20

        leg=list(set(int(x.split(' ')[1]) for x in df_plt.columns if len(x.split(' '))>2))+[0]
        
        fig = px.scatter(df_plt, x='Cycle_start', y='y',
                         title=f'Ring {r} Coordination/Split Diagram',
                         hover_data='Coord_plan',
                         height=500)
        fig.update_traces(marker=dict(symbol='line-ew'))

        for c in df_plt.columns[1:-2].to_list():
            gyr, p = c.split()[0:2]

            if gyr == 'G':
                cl = cl_P[p]
            else:
                cl = cl_RY[gyr]

            t = px.bar(df_plt, x='Cycle_start', y=c,
                       color_discrete_sequence=[cl],
                       hover_data=c).data[0]

            t.name = f'Ph {p}'
            fig.add_trace(t)
            if int(p) in leg:
                fig.data[-1].update(showlegend=True)
                leg.remove(int(p))
                
        leg=list(set(int(x.split(' ')[1]) for x in df_plt.columns if len(x.split(' '))>2))+[0]

        fig.update_layout(xaxis_range=[df_plt.iloc[0, 0], df_plt.iloc[0, -1]], barmode='stack')

        #leg = {1: [1, 2, 3, 4, 0], 2: [5, 6, 7, 8, 0]}
        leg=list(set(int(x.split(' ')[1]) for x in df_plt.columns if len(x.split(' '))>2))+[0]

        for idx, dets in df_dtcfg.iterrows():
            if int(idx[1]) in leg:#[r]:
                for det in dets.dropna():
                    df_dt = df_comb[(df_comb.Code == 82) & (df_comb.ID == det)]
                    t = px.scatter(df_dt, x='Cycle_start', y='t_cs',
                                   hover_name='ID',
                                   hover_data='Duration',
                                   color_discrete_sequence=['Black']).data[0]
                    t.legendgroup = det
                    t.name = f'{idx[:5]} ( Det {int(det)})'
                    fig.add_trace(t)
                    fig.data[-1].update(showlegend=True, visible='legendonly')

        fig.layout.xaxis.title = ''
        fig.layout.yaxis.title = ''
        fig.layout.xaxis.autorange = True
        fig.layout.yaxis.autorange = True

        figs.append(fig)

    path_out = os.path.join(path_out, f'Coord_Split{sfx}.html')
    figs[0].write_html(path_out)
    with open(path_out, 'a') as f:
        f.write(figs[1].to_html(full_html=False, include_plotlyjs='cdn'))

#---------------------------------------------------------------------------------------------------

def plot_term_plotly(df, save_path, intx, line=True, n_con=10):
    '''
    Plots termination plot using Plotly.
    df: df_raw
    save_path: path (including filename) to output file
    intx: name of intersection
    line: default=True, plot line showing weighted average of number of max outs
    n_con: default=10, number of consecutive cycles to calculate and plot weighted average
    '''

    fig = make_subplots(rows=1, cols=1, subplot_titles=[f"{intx} Phase Termination"])

    if line:
        df_term = df.copy().loc[(df.Code.isin([4,5,6])), :]
        df_term['Termination'] = (df_term['Code'] - 4).clip(upper=1) * 0.667 + df_term['ID'] - 0.333
        df_term['Termination'] = df_term.groupby('ID')['Termination'].transform(lambda t: t.rolling(n_con, center=True).mean())

        x_range = [df_term['TS_start'].min(), df_term['TS_start'].max()]
        for y in range(2, (2 + 3 * df_term['ID'].max())):
            fig.add_trace(go.Scatter(x=x_range, y=[y / 3] * 2, mode='lines', line=dict(color='gray')))
            
        for ID, df_ID in df_term.groupby('ID'):
            fig.add_trace(go.Scatter(x=df_ID['TS_start'], y=df_ID['Termination'], mode='lines', line=dict(color='black')))

    fig.add_trace(go.Scatter(x=df[df.Code == 4]['TS_start'], y=df[df.Code == 4]['ID'], mode='markers', marker=dict(color='green',symbol='circle')))
    fig.add_trace(go.Scatter(x=df[df.Code == 5]['TS_start'], y=df[df.Code == 5]['ID'], mode='markers', marker=dict(color='red',symbol='square')))
    fig.add_trace(go.Scatter(x=df[df.Code == 6]['TS_start'], y=df[df.Code == 6]['ID'], mode='markers', marker=dict(color='orange',symbol='diamond')))

    fig.update_layout(xaxis=dict(title='Time', type="date"),
                      yaxis=dict(title='Phase'))

    fig.update_layout(showlegend=False)
    #fig.update_layout(title_text=f"{intx} Phase Termination")

    fig.write_html(save_path)

#---------------------------------------------------------------------------------------------------
    
def plot_coord_bokeh(ring_dict, df_comb, det_path, path_out, sfx=''):
    '''
    Plots coordination/split diagram using Bokeh.

    Args:
    - ring_dict (dict): Dictionary containing ring dataframes for Ring 1 and Ring 2.
    - df_comb (DataFrame): Input DataFrame containing traffic event data.
    - det_path (str): Path to detector configuration file.
    - path_out (str): Output path for the plot.
    - sfx (str): Suffix for the output filename.
    '''

    plot_dts = {1: [53, 54, 61, 62], 2: [49, 50, 57, 58]}

    cl_RY = {'Rc': 'red', 'Y': 'yellow'}
    cl_P = {'1': 'orange', '5': 'orange', '3': 'purple', '7': 'purple',
            '2': 'blue', '6': 'blue', '4': 'green', '8': 'green'}

    leg = {1: [1, 2, 3, 4, 0], 2: [5, 6, 7, 8, 0]}

    df_dtcfg = pd.concat(read_det_cfg(os.path.join(det_path))[2:])

    figs = []

    for r in [1, 2]:
        df_plt = ring_dict[r]
        df_plt['y'] = -20

        p=figure(title=f'Ring {r} Coordination/Split Diagram', height=500, toolbar_location=None)
        p.scatter(x='Cycle_start', y='y', source=ColumnDataSource(df_plt), size=5, line_color='black')

        for c in df_plt.columns[1:-2].to_list():
            gyr, ph = c.split()[0:2]

            if gyr == 'G':
                cl = cl_P[ph]
            else:
                cl = cl_RY[gyr]

            p.vbar(x='Cycle_start', top=c, source=ColumnDataSource(df_plt),
                   color=cl, width=0.5, alpha=0.8, legend_label=f'Ph {p}')

        for idx, dets in df_dtcfg.iterrows():
            if int(idx[1]) in leg[r]:
                for det in dets.dropna():
                    df_dt = df_comb[(df_comb.Code == 82) & (df_comb.ID == det)]
                    p.circle(x='Cycle_start', y='t_cs', source=ColumnDataSource(df_dt),
                             size=5, color='black', legend_label=f'{idx[:5]} (Det {int(det)})', alpha=0.8)

        p.xaxis.axis_label = ''
        p.yaxis.axis_label = ''
        p.legend.click_policy = 'hide'
        p.legend.location = 'top_left'

        path_out = os.path.join(path_out, f'Coord_Split{sfx}.html')
        output_file(path_out)
        #show(p)
        figs.append(p)

    return figs

#---------------------------------------------------------------------------------------------------

def plot_term_bokeh(df, save_path, intx, line=True, n_con=10):
    '''
    Plots termination plot.

    Args:
    - df (DataFrame): Input DataFrame containing traffic event data.
    - save_path (str): Path (including filename) to the output file.
    - intx (str): Name of the intersection.
    - line (bool): Default=True, plot line showing weighted average of the number of max outs.
    - n_con (int): Default=10, number of consecutive cycles to calculate and plot the weighted average.
    '''

    p = figure(plot_width=1800, plot_height=500, x_axis_type="datetime",
               title=f"{intx} Phase Termination",
               tools=["pan", "box_zoom", "xwheel_zoom", "ywheel_zoom", "save"])

    if line:
        df_term = df.copy().loc[(df.Code.isin([4, 5])), :]
        df_term.loc[:, 'Termination'] = (df_term.loc[:, 'Code'] - 4) * 0.667 + df_term.loc[:, 'ID'] - 0.333
        df_term.loc[:, 'Termination'] = df_term.groupby('ID')['Termination'].\
            transform(lambda t: t.rolling(n_con, center=True).mean())

        for ID, df_ID in df_term.groupby('ID'):
            p.line(x=df_ID['TS_start'], y=df_ID['Termination'], color='black', alpha=0.25)

        x = [df_term.TS_start.min(), df_term.TS_start.max()]
        for y in range(2, (2 + 3 * df_term.ID.max())):
            p.line(x=x, y=y / 3, color='black', alpha=0.1)

    p.scatter(x=df[df.Code == 4]['TS_start'], y=df[df.Code == 4]['ID'], color='green', alpha=0.5)
    p.scatter(x=df[df.Code == 5]['TS_start'], y=df[df.Code == 5]['ID'], color='red', alpha=0.5)

    output_file(save_path)
    show(p)

#---------------------------------------------------------------------------------------------------

def arrival_on_green(df, dets, ph, tt=0, det_code=82):
    '''
    Calculates arrival on green for a particular phase on a per-cycle basis.

    Args:
    - df (DataFrame): Input DataFrame containing traffic event data.
    - dets (list): List of advanced detectors for the phase.
    - ph (int): Phase.
    - tt (int): Default=0, offset in seconds to account for travel time from arrival detector to intersection.
    - det_code (int): Default=82, detector event code. 82 for actuation, 81 for de-actuation.

    Returns:
    - DataFrame: Result DataFrame containing arrival on green data.
    '''

    df_arr = df[(((df.Code.isin([1, 8, 10, 11])) & (df.ID == ph))) | ((df.Code == det_code) & (df.ID.isin(dets)))]
    df_res = df_arr.loc[:, ['Cycle_start', 'Cycle_len']].drop_duplicates().set_index('Cycle_start')

    if tt:
        df_arr[df_arr.Code == det_code].loc[:, 'TS_start'] = \
            df_arr[df_arr.Code == det_code].loc[:, 'TS_start'] + dt.timedelta(seconds=tt)
        df_arr.sort_values('TS_start', inplace=True)

    df_arr = phase_status(df_arr, [ph])
    df_g = df_arr[df_arr.Code == 1]
    df_arr = df_arr[df_arr.Code == det_code]
    df_aog = df_arr[df_arr[f'Ph {ph} Status'].isin(['G'])]

    df_res.loc[:, 'G'] = df_g.loc[:, ['Cycle_start', 'Duration']].groupby('Cycle_start').sum()['Duration'].fillna(0)
    df_res.loc[:, 'G_pct'] = (df_res.G / df_res.Cycle_len)
    df_res.loc[:, 'Total'] = df_arr.groupby('Cycle_start').count()['ID'].fillna(0)
    df_res.loc[:, 'AoG'] = df_aog.groupby('Cycle_start').count()['ID'].fillna(0)
    df_res.loc[:, 'AoG_pct'] = (df_res.AoG / df_res.Total).fillna(0)
    df_res.loc[:, 'Phase'] = ph

    return df_res

#---------------------------------------------------------------------------------------------------

def bin_AOG(df_res, bin_len=60):
    '''
    Bins the arrival on green data.

    Args:
    - df_res (DataFrame): Arrival on green result DataFrame.
    - bin_len (int): Default=60, length in minutes of bins.

    Returns:
    - DataFrame: Binned arrival on green data.
    '''

    df_binres = pd.DataFrame()
    for ph in df_res.Phase.unique():
        for cp in df_res.Coord_plan.unique():
            df_bin = df_res.loc[(df_res.Phase == ph) & (df_res.Coord_plan == cp),
                                ['Cycle_start', 'Cycle_len', 'G', 'Total', 'AoG']]
            if len(df_bin) > 0:
                df_bin = df_bin.set_index('Cycle_start')
                df_bin = df_bin.resample(f'{bin_len}T').sum()
                df_bin.loc[:, 'AoG_pct'] = df_bin.AoG / df_bin.Total
                df_bin.loc[:, 'Phase'] = ph
                df_bin.loc[:, 'Coord_plan'] = cp
                df_bin.loc[:, 'G_pct'] = df_bin.G / df_bin.Cycle_len
                #df_binres = df_binres.append(df_bin)
                df_binres = pd.concat([df_binres, df_bin])

    df_binres = df_binres.fillna(0).loc[(df_binres.Total != 0),
                                        ['Phase', 'Coord_plan', 'G', 'G_pct', 'Total', 'AoG', 'AoG_pct']]
    df_binres.index.name = 'Time'
    df_binres.reset_index(inplace=True)
    df_binres = df_binres.sort_values(['Phase', 'Time'])

    return df_binres

# ---------------------------------------------------------------------------------------------------

def split_failures(df_raw, ph, det, t_red=5, thresh=0.8):
    '''
    Calculates green occupancy rate, red occupancy rate, and split failure for a particular phase.

    Args:
    - df_raw (DataFrame): Raw traffic data.
    - ph (int): Phase.
    - det (int): Detector.
    - t_red (int): Default=5, time after start of red clearance to calculate red occupancy.
    - thresh (float): Default=0.8, threshold of GOR and ROR to count as split failure.

    Returns:
    - DataFrame: Result DataFrame containing calculated values.
    '''

    # Create detector dataframe: 81 is deactivation, 82 is activation
    df_d = df_raw[(df_raw.Code.isin([81, 82])) & (df_raw.ID == det)].loc[:, ['TS_start', 'Code']].sort_values('TS_start')
    df_d.loc[:, 'Prev'] = df_d.Code.shift(1)
    df_d = df_d[df_d.Code != df_d.Prev]
    df_d.loc[:, 'Next'] = df_d.Code.shift(-1)
    df_d.loc[:, 'TS_end'] = df_d.TS_start.shift(-1)
    df_d = df_d.dropna(axis=0)
    df_d.drop(['Next', 'Prev'], axis=1, inplace=True)

    resdict = {}

    for i in [0, 1]:
        # First time through the loop, set up GOR, second ROR
        if i == 0:
            c_start = 1
            c_end = 10
            idx = ['G', 'G_occ', 'GOR']

            # Create GOR phase event dataframe. 1 is begin green, 10 is begin red clearance
            df_p = df_raw[(df_raw.Code.isin([1, 10])) & (df_raw.ID == ph)].loc[:, ['TS_start', 'Code']]
            df_p.loc[:, 'Next'] = df_p.Code.shift(-1)
            df_p.loc[:, 'TS_end'] = df_p.TS_start.shift(-1)
            df_p = df_p[df_p.Code != df_p.Next].dropna(axis=0)
            df_p.drop('Next', axis=1, inplace=True)

        else:
            c_start = 10
            c_end = 1010
            idx = ['R', 'R_occ', 'ROR']

            # Create ROR phase event dataframe. 10 is begin red clearance, 1010 is red clearance + t_red
            df_p = df_raw[(df_raw.Code == 10) & (df_raw.ID == ph)].loc[:, ['TS_start', 'Code']]
            df_p.loc[:, 'TS_end'] = df_p.loc[:, 'TS_start'] + dt.timedelta(seconds=t_red)
            df_1010 = df_p.copy()
            df_1010.loc[:, 'Code'] = 1010
            df_1010.loc[:, 'TS_start'] = df_1010.loc[:, 'TS_end']
            df_p = pd.concat([df_p, df_1010], ignore_index=True).sort_values('TS_start')

        # Combine detector and phase dataframes
        dfc = pd.concat([df_p, df_d], ignore_index=True).sort_values(['TS_start', 'Code'])

        # Forward fill detector status to phase events
        dfc.loc[(dfc.Code.isin([81, 82])), 'Det'] = dfc.loc[(dfc.Code.isin([81, 82])), 'Code']
        dfc = dfc.ffill()
        dfc.loc[:, 'Next'] = dfc.Code.shift(-1)
        dfc.loc[:, 'TS_next'] = dfc.TS_start.shift(-1)
        dfc.loc[:, 'Prev'] = dfc.Code.shift(1)

        # Change end time for detection events to end of green event
        dfc.loc[((dfc.Code == 82) & (dfc.Next == c_end)), 'TS_end'] = \
            dfc[(dfc.Code == 82) & (dfc.Next == c_end)].loc[:, 'TS_next']

        # Add 82's for all cycles with no detection events but start with detection active
        df_ag = dfc[(dfc.Code == c_start) & (dfc.Next == c_end) & (dfc.Det == 82)]
        df_ag.loc[:, 'Code'] = 82
        dfc = dfc.append(df_ag, ignore_index=True).sort_values(['TS_start', 'Code'])
        dfc.loc[:, 'Next'] = dfc.Code.shift(-1)
        dfc.loc[:, 'TS_next'] = dfc.TS_start.shift(-1)
        dfc.loc[:, 'Prev'] = dfc.Code.shift(1)

        # Add 82's at the beginning of each green if next event is deactivation
        df_gstart = dfc[(dfc.Code == c_start) & (dfc.Next == 81)]
        df_gstart.loc[:, 'TS_end'] = df_gstart.TS_next
        df_gstart.loc[:, 'Code'] = 82
        dfc = dfc.append(df_gstart).sort_values(['TS_start', 'Code'])

        # Add column to group each green by
        dfc.loc[(dfc.Code == c_start), 'Period'] = dfc.loc[dfc.Code == c_start, 'TS_start']
        dfc.loc[(dfc.Code == c_end), 'Period'] = 0
        dfc = dfc.ffill().dropna(axis=0)

        # Drop end green and end detection events
        dfc = dfc[(dfc.Code.isin([c_start, 82])) & (dfc.Period != 0)]

        # Calculate duration
        dfc.loc[:, 'Duration'] = (dfc.TS_end - dfc.TS_start).dt.total_seconds()

        dfres = pd.DataFrame()

        for p, gb in dfc.groupby('Period'):
            if i == 0:
                nm = gb.iloc[0].TS_end
            else:
                nm = p

            srs = pd.Series(name=nm, index=idx)
            d = gb.iloc[0].Duration
            occ = gb[gb.Code == 82].Duration.sum()

            srs.iloc[0] = d
            srs.iloc[1] = min(occ, d)
            srs.iloc[2] = min(occ, d) / d

            dfres = dfres.append(srs)

        resdict[i] = dfres.copy()

    resdict[0].index.name = 'Time'
    resdict[1].index.name = 'Time'
    df_res = pd.merge(resdict[0], resdict[1], on='Time')
    df_res.loc[:, 'Phase'] = ph
    df_res.loc[:, 'Detector'] = det
    df_res.loc[:, 'Split Fail'] = 1 * ((df_res.GOR > thresh) & (df_res.ROR > thresh))
    df_cp = df_raw.loc[:, ['Coord_plan', 'TS_start']].drop_duplicates()
    df_cp = df_cp.set_index('TS_start')
    df_cp.index.name = 'Time'
    df_res = pd.merge(df_res, df_cp, on='Time')

    return df_res.loc[:, ['Phase', 'Detector', 'Coord_plan',
                          'G', 'G_occ', 'GOR',
                          'R', 'R_occ', 'ROR', 'Split Fail']]

# ---------------------------------------------------------------------------------------------------

def bin_SF(df_res, bin_len=60):
    '''f
    Bins the split failures data.

    Args:
    - df_res (DataFrame): Split failure result DataFrame.
    - bin_len (int): Default=60, length in minutes of bins.

    Returns:
    - DataFrame: Binned split failures data.
    '''

    df_binres = pd.DataFrame()
    for ph in df_res.Phase.unique():
        for cp in df_res.Coord_plan.unique():
            df_res.loc[:, 'N_cyc'] = 1
            df_bin = df_res.loc[(df_res.Phase == ph) & (df_res.Coord_plan == cp),
                                ['G', 'G_occ', 'R', 'R_occ', 'N_cyc',
                                 'Split Fail']].resample('%sT' % bin_len).sum()
            if len(df_bin) > 0:
                df_bin.loc[:, 'GOR'] = df_bin.G_occ / df_bin.G
                df_bin.loc[:, 'ROR'] = df_bin.R_occ / df_bin.R
                df_bin.loc[:, 'SF_pct'] = df_bin.loc[:, 'Split Fail'] / df_bin.loc[:, 'N_cyc']
                df_bin.loc[:, 'Phase'] = ph
                df_bin.loc[:, 'Coord_plan'] = cp
                df_binres = df_binres.append(df_bin)

    df_binres = df_binres.loc[:, ['Phase', 'Coord_plan', 'G_occ', 'GOR', 'R_occ', 'ROR',
                                  'N_cyc', 'Split Fail', 'SF_pct']]

    df_binres = df_binres.fillna(0).loc[(df_binres.N_cyc > 0),
                                        ['Phase', 'Coord_plan', 'G_occ', 'GOR', 'R_occ', 'ROR',
                                         'N_cyc', 'Split Fail', 'SF_pct']]
    df_binres.index.name = 'Time'
    df_binres.reset_index(inplace=True)
    df_binres = df_binres.sort_values(['Phase', 'Time']).loc[:, ['Time', 'Phase', 'Coord_plan',
                                                                  'G_occ', 'GOR', 'R_occ', 'ROR',
                                                                  'Split Fail', 'N_cyc', 'SF_pct']]

    return df_binres

# ---------------------------------------------------------------------------------------------------

def combine_aog_sf_tt(intx, bin_len=60, write_to_file=None, sfx='',
                      xl_mode='w', phases=[2, 4, 6, 8], return_by_cp=False,
                      atr=None, sf_sfx='', aog_sfx='', count_sfx=''):
    '''
    Combines AOG, SF, and TT data and performs analysis.

    Args:
    - intx (str): Name of the intersection.
    - bin_len (int): Default=60, length in minutes of bins.
    - write_to_file (str): Default=None, output format ('xl', 'csv', 'pk') or None to skip writing.
    - sfx (str): Default='', suffix for output filenames.
    - xl_mode (str): Default='w', mode for Excel file ('w' for write, 'a' for append).
    - phases (list): Default=[2, 4, 6, 8], list of phases to analyze.
    - return_by_cp (bool): Default=False, return dataframes by coordination plan.
    - atr (dict): Default=None, dictionary specifying additional volume count data.
    - sf_sfx (str): Default='', suffix for SF filenames.
    - aog_sfx (str): Default='', suffix for AOG filenames.
    - count_sfx (str): Default='', suffix for count filenames.

    Returns:
    - dict: Result dataframes.
    '''

    seg_key = pd.read_csv('./inrix/INRIX_key.csv')
    df_tt = pd.read_csv('./inrix/data_%s.csv' % bin_len)
    df_tt.loc[:, 'Date Time'] = pd.to_datetime(df_tt.loc[:, 'Date Time']).apply(
        lambda x: x.replace(tzinfo=None))
    resdict = {}

    ph_dir = {2: 'WB', 4: 'SB', 6: 'EB', 8: 'NB'}

    if bin_len == 60:
        f_end = 'hourly.csv'
    elif bin_len == 15:
        f_end = '15min.csv'

    # Split failure dataframe
    df_sf = pd.DataFrame()
    path_sf = './ACHD_Eagle-{}/Split Failures/'.format(intx)
    for f_sf in [f for f in os.listdir(path_sf) if f.endswith('{}{}'.format(sf_sfx, f_end))]:
        df_sf = df_sf.append(pd.read_csv(os.path.join(path_sf, f_sf)))
    df_sf.loc[:, 'Time'] = pd.to_datetime(df_sf.Time, infer_datetime_format=True)
    df_sfb = df_sf

    # AOG dataframe
    df_aog = pd.DataFrame()
    path_aog = './ACHD_Eagle-{}/AOG/'.format(intx)
    for f_aog in [f for f in os.listdir(path_aog) if f.endswith('{}{}'.format(aog_sfx, f_end))]:
        df_aog = df_aog.append(pd.read_csv(os.path.join(path_aog, f_aog)))
    df_aog.loc[:, 'Time'] = pd.to_datetime(df_aog.Time, infer_datetime_format=True)

    # Count dataframe
    df_ct = pd.DataFrame()
    path_ct = './ACHD_Eagle-{}/Counts/'.format(intx)
    for f_ct in [f for f in os.listdir(path_ct) if f.endswith('{}{}'.format(count_sfx, f_end))]:
        df_ct = df_ct.append(pd.read_csv(os.path.join(path_ct, f_ct)))
    df_ct.loc[:, 'Time'] = pd.to_datetime(df_ct.Time, infer_datetime_format=True)

    df_b = pd.merge(df_aog, df_sf.loc[:, ['Time', 'Phase', 'Coord_plan',
                                          'G_occ', 'GOR', 'R_occ', 'ROR', 'Split Fail', 'N_cyc', 'SF_pct']],
                   'outer', on=['Time', 'Phase', 'Coord_plan'])

    df_b = pd.merge(df_b, df_ct.loc[:, ['Time',
                                        'WBL', 'WBT', 'WBR',
                                        'SBL', 'SBT', 'SBR',
                                        'EBL', 'EBT', 'EBR',
                                        'NBL', 'NBT', 'NBR']])

    for ph in phases:

        seg_row = seg_key.loc[(seg_key.Intersection == intx) & (seg_key.Phase == ph), :]
        if len(seg_row) == 0:
            continue

        if atr and bin_len == 60:
            if ph in list(atr.keys()):
                df_atr = pd.read_csv('./ACHD_Eagle-{}/Counts/{}_long.csv'.format(intx, atr[ph][0]))
                df_atr.loc[:, 'Datetime'] = pd.to_datetime(df_atr.Datetime, infer_datetime_format=True)
                df_atr = df_atr.loc[(df_atr.Dir == atr[ph][1]), ['Datetime', 'Count']]
                df_atr.rename({'Count': 'Vol'}, axis=1, inplace=True)
                df_b = pd.merge(df_b, df_atr, left_on='Time', right_on='Datetime', how='left')

        seg = seg_row['Segment ID'].iloc[0]
        df_tt_int = df_tt.loc[(df_tt.loc[:, 'Segment ID'] == seg),
                              ['Date Time', 'Travel Time(Minutes)']]
        dfbcols = ['Time', 'Coord_plan', 'G', 'G_pct', 'Total',
                   'AoG', 'AoG_pct', 'G_occ', 'GOR', 'R_occ', 'ROR', 'Split Fail', 'SF_pct']
        dfbcols = dfbcols + [c for c in df_b.columns if c.startswith(ph_dir[ph]) or c == 'Vol']
        df_j = pd.merge(df_b.loc[(df_b.Phase == ph), dfbcols],
                        df_tt_int, left_on='Time',
                        right_on='Date Time').drop('Date Time', axis=1)

        if return_by_cp:
            for cp in df_j.Coord_plan.unique():
                resdict['%s_ph%s_cp%s_tt' % (intx, ph, int(cp))] = df_j.loc[(df_j.Coord_plan == cp), :]
        else:
            resdict['{}_ph{}_tt'.format(intx, ph)] = df_j

        if 'Vol' in df_b.columns:
            df_b.drop('Vol', axis=1, inplace=True)

    if write_to_file:
        if write_to_file == 'xl':
            with pd.ExcelWriter('./ACHD_Eagle-%s/Travel_Time%s.xlsx' % (intx, sfx),
                                mode=xl_mode, engine='openpyxl') as writer:
                for nm, df in resdict.items():
                    df.to_excel(writer, sheet_name=nm, index=False)
        elif write_to_file == 'csv':
            if not os.path.exists('./ACHD_Eagle-{}/TT'.format(intx)):
                os.mkdir('./ACHD_Eagle-{}/TT'.format(intx))
            for nm, df in resdict.items():
                df.to_csv('./ACHD_Eagle-{}/TT/{}.csv'.format(intx, nm), index=False)
        elif write_to_file == 'pk':
            if not os.path.exists('./ACHD_Eagle-{}/TT'.format(intx)):
                os.mkdir('./ACHD_Eagle-{}/TT'.format(intx))
            for nm, df in resdiect.items():
                df.to_pickle('./ACHD_Eagle-{}/TT/{}.pk'.format(intx, nm))

    return resdict

# ---------------------------------------------------------------------------------------------------

def tt_vol_anova(rd, t_ba, atr_vol=True):
    ph_dir = {2: 'WB', 4: 'SB', 6: 'EB', 8: 'NB'}
    cp_days = {'1': ['Tuesday', 'Wednesday', 'Thursday'],
               '2': ['Tuesday', 'Wednesday', 'Thursday'],
               '4': ['Saturday'], '6': ['Saturday'],
               '4,6': ['Sunday']}
    intx = list(rd.keys())[0].split('_')[0]
    dfres = pd.DataFrame()
    for ph in ph_dir.keys():
        if '{}_ph{}_tt'.format(intx, ph) not in rd.keys():
            continue
        for cp in cp_days.keys():
            drct = ph_dir[ph]
            df = rd['{}_ph{}_tt'.format(intx, ph)].copy()
            df.rename({'Travel Time(Minutes)': 'TT', 'Split Fail': 'SF'}, axis=1, inplace=True)

            if 'Vol' not in df.columns and atr_vol:
                df.loc[:, 'Vol'] = df.loc[:, '{}T'.format(drct)] + df.loc[:, '{}R'.format(drct)]

            df.loc[:, 'BA'] = 0
            df.loc[(df.Time > t_ba), 'BA'] = 1

            df = filter_day(df, cp_days[cp], 'Time')

            cp_split = [n + '.0' for n in cp.split(',')]
            ttb = df.loc[(df.BA == 0) & (df.Coord_plan.astype(str).isin(cp_split)), 'TT'].describe()
            ttb.index = 'TT_B ' + ttb.index
            tta = df.loc[(df.BA == 1) & (df.Coord_plan.astype(str).isin(cp_split)), 'TT'].describe()
            tta.index = 'TT_A ' + tta.index
            volb = df.loc[(df.BA == 0) & (df.Coord_plan.astype(str).isin(cp_split)), 'Vol'].describe()
            volb.index = 'Vol_B ' + volb.index
            vola = df.loc[(df.BA == 1) & (df.Coord_plan.astype(str).isin(cp_split)), 'Vol'].describe()
            vola.index = 'Vol_A ' + vola.index

            srs = pd.concat([ttb, tta, volb, vola])
            str_days = ''
            for day in cp_days[cp]:
                str_days += day[0:3]
            srs.name = 'Ph{} CP{} {}'.format(ph, cp, str_days)

            for v in ['TT', 'AoG_pct', 'SF_pct']:
                ld = '{} ~ Vol + BA + BA * Vol'.format(v)
                lm = ols(ld, df).fit()
                lm_sum = lm.summary()
                t1 = pd.read_html(lm_sum.tables[1].as_html(), header=0, index_col=0)[0]
                rsq = float(lm_sum.tables[0][0][3].data)
                srs['{} RSQ'.format(v)] = rsq
                for i in ['Vol', 'BA', 'BA:Vol']:
                    for j in ['coef', 'P>|t|']:
                        srs['{} {} {}'.format(v, i, j)] = t1.loc[i, j]

            dfres = pd.concat([dfres, srs], axis=1)
    dfres = dfres.T
    dfres.loc[:, 'Phase'] = dfres.index.map(lambda x: x[2])
    dfres.loc[:, 'Coord_plan'] = dfres.index.map(lambda x: x.split('P')[2])
    dfres.loc[:, 'Delta_50%TT'] = (dfres.loc[:, 'TT_A 50%'] -
                                   dfres.loc[:, 'TT_B 50%']) / dfres.loc[:, 'TT_B 50%']
    dfres.loc[:, 'Delta_50%Vol'] = (dfres.loc[:, 'Vol_A 50%'] -
                                    dfres.loc[:, 'Vol_B 50%']) / dfres.loc[:, 'Vol_B 50%']
    dfres_cols = ['Coord_plan', 'Phase'] + dfres.columns[0:16].to_list() + ['Delta_50%TT'] + \
                 dfres.columns[16:32].to_list() + ['Delta_50%Vol'] + dfres.columns[32:-4].to_list()

    return dfres.loc[:, dfres_cols].sort_values(['Coord_plan', 'Phase']).reset_index(drop=True)

# ---------------------------------------------------------------------------------------------------

