import os
import shutil
import pandas as pd
from pathlib import Path
from spmfunctions.misc_tools import comb_gyr_det, phase_status, detector_status
from datetime import datetime, time

def read_df_raw(f_path):
    '''
    Reads data from csv into dataframe and outputs df_raw.
    
    Args:
    - f_path (str): Path including filename, of "compiled" CSV. 
      If ACHD, should have 4 columns (TS, Code, Description, and ID),
      and if ITD, should have 3. Neither should have headers.
      
    Returns:
    - df_raw (DataFrame): Processed dataframe.
    '''

    # Check if the file path indicates ACHD data
    if 'ACHD' in f_path:
        # Read CSV with specific columns and without headers
        dtype_map = {'Code': 'int16', 'ID': 'int16'}
        df_raw = pd.read_csv(f_path, 
                             usecols=[0, 1, 3], 
                             names=['TS_start', 'Code', 'ID'],
                             dtype=dtype_map,
                             parse_dates=['TS_start'], # Let Pandas handle parsing directly
                             engine='pyarrow',
                             dtype_backend='pyarrow',
                             infer_datetime_format=True)\
                   .drop_duplicates()
        
        # Convert 'TS_start' column to datetime
        df_raw['TS_start'] = pd.to_datetime(df_raw['TS_start'])

        # Sort, drop duplicates, and reset index
        df_raw = df_raw.sort_values(['TS_start', 'Code', 'ID'])\
                       .drop_duplicates()\
                       .reset_index(drop=True)

        # Identify rows of last barrier
        r1=[1,2,5,6]
        r2=[3,4,7,8]        

        dfg = df_raw[df_raw.Code==1].groupby('TS_start').apply(lambda x: list(x.ID))
        idx_bt = dfg[(dfg.apply(lambda x: all(elem in r1 for elem in x)))&
                    (dfg.shift().apply(lambda x: all(elem in r2 for elem in x) if isinstance(x, list) else False))]

        idx_bt = idx_bt.reset_index().drop([0], axis=1)        

    else:
        # Read CSV with specific columns and without headers
        dtype_map = {'Code': 'int16', 'ID': 'int16'}
        df_raw = pd.read_csv(f_path, 
                             names=['TS_start', 'Code', 'ID'],
                             dtype=dtype_map,
                             parse_dates=['TS_start'], # Let Pandas handle parsing directly
                             engine='pyarrow',
                             dtype_backend='pyarrow',
                             infer_datetime_format=True)

        # Convert 'TS_start' column to datetime
        df_raw['TS_start'] = pd.to_datetime(df_raw['TS_start'])

        # Sort, drop duplicates, and reset index
        df_raw = df_raw.sort_values(['TS_start', 'Code', 'ID'])\
                       .drop_duplicates()\
                       .reset_index(drop=True)

        # Identify new cycles
        df_bar = df_raw[df_raw.Code == 31].copy()
        df_bar['next_id'] = df_bar['ID'].shift(-1)
        idx_bt = df_bar.loc[df_bar.next_id < df_bar.ID]
        #idx_bt = df_raw.loc[(df_raw.Code == 31) & 
        #                    (df_raw.ID == df_raw[df_raw.Code == 31].ID.max())]

    # Format Timestamp column and add column for when each cycle starts
    idx_bt.loc[:, 'Cycle_start'] = idx_bt.loc[:, 'TS_start'].copy()
    df_raw = pd.merge(df_raw, idx_bt.loc[:, ['TS_start', 'Cycle_start']].copy(), 
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

def read_int_cfg(path_cfg, f_cfg='int_cfg.csv'):
    '''
    Reads intersection configuration from a CSV file.
    
    Args:
    - path_cfg (str): Path to the directory containing the configuration file.
    - f_det (str): Filename of the intersection configuration CSV file.
    
    Returns:
    - Dictionary: Contains DataFrames for movement, exclusion, arrival, and stopbar configurations.
    '''

    f_path = os.path.join(path_cfg, f_cfg)
    df_cfg = pd.read_csv(f_path, index_col=[0,1])

    # Convert columns to datetime
    df_cfg = df_cfg.dropna(how='all', axis=1)
    df_cfg.columns = pd.to_datetime(df_cfg.columns, errors='ignore')

    # Extract different configurations
    d = {'TM:' : 'Movements', 'Exc:' : 'Exclusions', 'Plt:' : 'Arrivals', 'RB:' : 'Ring-Barrier'}
    cfg_dict = {d[key]: df_cfg.loc[key].T.dropna(how='all') for key in df_cfg.index.levels[0]}

    return cfg_dict

def archive_csv(csv_path: Path) -> None:
    """
    Move a CSV file to an Archive subdirectory within its current directory
    
    Args:
        csv_path (Path): Path to the CSV file to archive
    """
    # Create Archive directory if it doesn't exist
    archive_dir = csv_path.parent / 'Archive'
    archive_dir.mkdir(exist_ok=True)
    
    # Move file to Archive directory
    try:
        shutil.move(str(csv_path), str(archive_dir / csv_path.name))
        print(f"Archived {csv_path.name}")
    except Exception as e:
        print(f"Error archiving {csv_path.name}: {str(e)}")

def extract_timestamps_from_filename(filename: str) -> tuple[datetime, datetime]:
    """
    Extract start and end timestamps from filename pattern: compiled_yyyy_mm_dd_hhmm-yyyy_mm_dd_hhmm.csv
    
    Args:
        filename (str): Filename to parse
        
    Returns:
        tuple[datetime, datetime]: Start and end timestamps
    """
    # Remove 'compiled_' prefix and '.csv' suffix
    timestamp_part = filename.replace('compiled_', '').replace('.csv', '')
    start_str, end_str = timestamp_part.split('-')
    
    # Parse timestamps
    start_time = datetime.strptime(start_str, '%Y_%m_%d_%H%M')
    end_time = datetime.strptime(end_str, '%Y_%m_%d_%H%M')
    
    return start_time, end_time

def generate_pickle_name(date: datetime.date, 
                        file_start_time: datetime, file_end_time: datetime) -> str:
    """
    Generate appropriate pickle filename based on dates
    """
   
    # If this is the first day, use the file's start time
    if date == file_start_time.date():
        start_timestamp = file_start_time.strftime('%H%M')
    else:
        start_timestamp = '0000'
        
    # If this is the last day, use the file's end time
    if date == file_end_time.date():
        end_timestamp = file_end_time.strftime('%H%M')
        if end_timestamp == '2359':
            end_timestamp == '2400'
    else:
        end_timestamp = '2400'
        
    return f"df_raw_{date.strftime('%Y_%m_%d')}_{start_timestamp}-{date.strftime('%Y_%m_%d')}_{end_timestamp}.pkl"

def process_csv_files(input_dir: str, output_dir: str) -> None:
    """
    Process all CSV files in a directory, splitting them by date into pickle files.
    Files are moved to an Archive subdirectory after successful processing.
    
    Args:
        input_dir (str): Path to directory containing CSV files
        output_dir (str): Path to directory where pickle files will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all CSV files in input directory
    csv_files = list(Path(input_dir).glob('*.csv'))
    
    for csv_file in csv_files:
        try:
            print(f"Processing {csv_file.name}")
            
            # Extract file timestamps
            file_start_time, file_end_time = extract_timestamps_from_filename(csv_file.name)
            
            # Read the CSV file using the provided function
            df = read_df_raw(str(csv_file))
            
            # Ensure Cycle_start is datetime
            if not pd.api.types.is_datetime64_any_dtype(df['Cycle_start']):
                df['Cycle_start'] = pd.to_datetime(df['Cycle_start'])
            
            # Get unique dates
            unique_dates = sorted(df['Cycle_start'].dt.date.unique())
            
            print(f"Found {len(unique_dates)} unique dates in {csv_file.name}")
            
            # Process each date
            for date in unique_dates:
                # Filter for current date
                date_mask = df['Cycle_start'].dt.date == date
                date_df = df[date_mask].sort_values('Cycle_start')
                
                # Generate pickle filename
                pickle_name = generate_pickle_name(date, file_start_time, file_end_time)
                pickle_path = os.path.join(output_dir, pickle_name)
                
                print(f"Saving {pickle_name}")
                
                # Save to pickle
                date_df.sort_values('TS_start').reset_index(drop=True).to_pickle(pickle_path, compression="bz2")
            
            # Archive the CSV file after successful processing
            archive_csv(csv_file)
                
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")
            continue

def process_single_csv(csv_path: str, output_dir: str) -> None:
    """
    Process a single CSV file, splitting it by date into pickle files.
    File is moved to an Archive subdirectory after successful processing.
    
    Args:
        csv_path (str): Path to CSV file
        output_dir (str): Path to directory where pickle files will be saved
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        csv_path = Path(csv_path)
        print(f"Processing {csv_path.name}")
        
        # Extract file timestamps
        file_start_time, file_end_time = extract_timestamps_from_filename(csv_path.name)
        
        # Read the CSV file using the provided function
        df = read_df_raw(str(csv_path))
        
        # Ensure Cycle_start is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['Cycle_start']):
            df['Cycle_start'] = pd.to_datetime(df['Cycle_start'])
        
        # Get unique dates
        unique_dates = sorted(df['Cycle_start'].dt.date.unique())
        
        print(f"Found {len(unique_dates)} unique dates")
        
        # Process each date
        for date in unique_dates:
            # Filter for current date
            date_mask = df['Cycle_start'].dt.date == date
            date_df = df[date_mask].sort_values('Cycle_start')
            
            # Generate pickle filename
            pickle_name = generate_pickle_name(date, file_start_time, file_end_time)
            pickle_path = os.path.join(output_dir, pickle_name)
            
            print(f"Saving {pickle_name}")
            
            # Save to pickle
            date_df.to_pickle(pickle_path, compression="bz2")
        
        # Archive the CSV file after successful processing
        archive_csv(csv_path)
            
    except Exception as e:
        print(f"Error processing {csv_path}: {str(e)}")

def read_and_process_pickles(df_directory: str, date_range: str, transforms=None):
    """
    Read and process pickles from the specified directory and date range.
    
    Args:
        df_directory (str): Directory containing the pickles.
        date_range (str): Date or date range in the format "yyyy_mm_dd" or "yyyy_mm_dd-yyyy_mm_dd".
        transforms (list): List of transformation tasks to apply. Each task is a tuple (function, args_dict). Default is None.
        
    Returns:
        pd.DataFrame: Processed dataframe.
    """
    # Parse the date range
    if "-" in date_range:
        start_date_str, end_date_str = date_range.split("-")
        start_date = datetime.strptime(start_date_str.strip(), "%Y_%m_%d")
        end_date = datetime.strptime(end_date_str.strip(), "%Y_%m_%d")
    else:
        start_date = datetime.strptime(date_range.strip(), "%Y_%m_%d")
        end_date = start_date

    # Get list of pickle files in the directory
    pickle_files = list(Path(df_directory).glob("df_raw_*.pkl"))
    
    # Filter pickle files by date range
    filtered_files = []
    for pickle_file in pickle_files:
        # Extract date from filename
        filename = pickle_file.stem
        date_str = filename.split("-")[1]
        file_date = datetime.strptime(date_str[:-5], "%Y_%m_%d")
                                              
        if start_date <= file_date <= end_date:
            filtered_files.append(pickle_file)
    
    # Read and concatenate contiguous dataframes
    dataframes = []
    current_df = None
    for pickle_file in sorted(filtered_files):
        df = pd.read_pickle(pickle_file, compression="bz2")
        
        if current_df is None:
            current_df = df
        else:
            # Check if the dataframes are contiguous
            last_end_time = current_df['Cycle_start'].max()
            first_start_time = df['Cycle_start'].min()
            
            if last_end_time == first_start_time:
                current_df = pd.concat([current_df, df])
            else:
                dataframes.append(current_df)
                current_df = df
    
    if current_df is not None:
        dataframes.append(current_df)
    
    # Apply transformations if specified
    if transforms:
        transformed_dataframes = []
        for df in dataframes:
            for transform, args in transforms:
                df = transform(df, **args)
            transformed_dataframes.append(df)
        dataframes = transformed_dataframes
    
    # Concatenate all dataframes, sort by Cycle_start, and reset index
    final_df = pd.concat(dataframes).sort_values(['Cycle_start','TS_start','Code','ID']).reset_index(drop=True)
    
    return final_df