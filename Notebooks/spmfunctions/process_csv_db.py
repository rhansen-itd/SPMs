import os
import glob
import json
import sqlite3
import pandas as pd
import pytz
from datetime import datetime

# --- Configuration ---
# CSV Input Columns
CSV_COL_TIME = 'Event Time'
CSV_COL_CODE = 'Event Code'
CSV_COL_DESC = 'Event Description'
CSV_COL_PARAM = 'Event Parameter'

# Database Output Columns (Matches process_datz_db.py)
DB_COL_TIMESTAMP = 'timestamp'
DB_COL_EVENT = 'event_type'
DB_COL_PARAM = 'parameter'

def init_db(db_path):
    """Initialize SQLite DB with the same schema as process_datz_db."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            timestamp REAL, 
            event_type INTEGER,
            parameter INTEGER
        )
    ''')
    c.execute('CREATE INDEX IF NOT EXISTS idx_ts ON logs (timestamp)')
    conn.commit()
    return conn

def to_utc_epoch(dt_obj):
    """
    Converts a naive datetime (assumed Mountain Time) to UTC epoch.
    Matches logic in process_datz_db.py.
    """
    mst_tz = pytz.timezone('US/Mountain')
    dt_aware = mst_tz.localize(dt_obj)
    return dt_aware.timestamp()

def parse_file_metadata(filepath):
    """
    Reads the first few lines of the CSV to extract Start and End times.
    Handles headers like: Start time,"Sunday, 30 July 2023 00:00:00",
    """
    meta = {'filepath': filepath, 'start': None, 'end': None}
    try:
        with open(filepath, 'r') as f:
            lines = [f.readline() for _ in range(3)]
            
        # Parse Start Time (Line 2)
        # Split by " to extract the date string safely between quotes
        if '"' in lines[1]:
            start_str = lines[1].split('"')[1]
            meta['start'] = datetime.strptime(start_str, "%A, %d %B %Y %H:%M:%S")
        
        # Parse End Time (Line 3)
        if '"' in lines[2]:
            end_str = lines[2].split('"')[1]
            meta['end'] = datetime.strptime(end_str, "%A, %d %B %Y %H:%M:%S")
        
        return meta
    except Exception as e:
        print(f"Error parsing metadata for {os.path.basename(filepath)}: {e}")
        return None

def filter_files(file_list):
    """
    Removes files whose date ranges are fully contained within other files.
    """
    # Sort by start time, then end time (descending) to prefer longer durations
    sorted_files = sorted(file_list, key=lambda x: (x['start'], -x['end'].timestamp()))
    
    keep_indices = set(range(len(sorted_files)))
    
    for i in range(len(sorted_files)):
        if i not in keep_indices: continue
        current = sorted_files[i]
        
        for j in range(len(sorted_files)):
            if i == j: continue
            if j not in keep_indices: continue
            
            other = sorted_files[j]
            
            # Check if 'other' is fully contained within 'current'
            if current['start'] <= other['start'] and current['end'] >= other['end']:
                keep_indices.discard(j)

    return [sorted_files[i] for i in sorted(list(keep_indices))]

def process_csv_directory(input_dir, output_dir):
    """
    Main workflow: Scan -> Filter -> Analyze Continuity -> Process -> Save DB
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Identify Intersections
    # Pattern: [IntersectionNumber]_Events_[Timestamp].csv
    all_files = glob.glob(os.path.join(input_dir, "*_Events_*.csv"))
    intersection_map = {}
    
    for f in all_files:
        filename = os.path.basename(f)
        try:
            intersection_id = filename.split('_')[0]
            if intersection_id not in intersection_map:
                intersection_map[intersection_id] = []
            intersection_map[intersection_id].append(f)
        except IndexError:
            continue

    print(f"Found {len(intersection_map)} intersections.")

    # 2. Loop through each intersection
    for int_id, file_paths in intersection_map.items():
        print(f"Processing Intersection {int_id}...")
        
        # --- Metadata & Filtering ---
        raw_meta = []
        for fp in file_paths:
            m = parse_file_metadata(fp)
            if m: raw_meta.append(m)
            
        valid_files = filter_files(raw_meta)
        valid_files.sort(key=lambda x: x['start']) # Ensure chronological order
        
        if not valid_files:
            print(f"No valid files for {int_id}")
            continue

        # --- Continuity Analysis & JSON Generation ---
        ranges_report = []
        discontinuities = [] # List of start times where a gap precedes them
        
        prev_end = None
        
        for f in valid_files:
            current_start = f['start']
            current_end = f['end']
            
            is_continuous = True
            if prev_end:
                # Continuous if start is roughly prev_end + 1 second
                # Using 1.5s buffer to allow for standard midnight rollover
                diff = (current_start - prev_end).total_seconds()
                if diff > 1.5: 
                    is_continuous = False
                    discontinuities.append(current_start)
            
            ranges_report.append({
                "file": os.path.basename(f['filepath']),
                "start": current_start.isoformat(),
                "end": current_end.isoformat(),
                "continuous_with_prev": is_continuous
            })
            
            prev_end = current_end

        # Save JSON Report
        json_path = os.path.join(output_dir, f"{int_id}_data_ranges.json")
        with open(json_path, 'w') as jf:
            json.dump(ranges_report, jf, indent=4)

        # --- Process DataFrames & Database ---
        db_name = f"{int_id}_data.db"
        db_path = os.path.join(output_dir, db_name)
        conn = init_db(db_path) # Create DB
        
        for file_info in valid_files:
            fp = file_info['filepath']
            
            try:
                # Read CSV (Header is on line 5)
                df = pd.read_csv(fp, skiprows=4)
                
                # Clean column names (strip spaces)
                df.columns = [c.strip() for c in df.columns]
                
                if CSV_COL_TIME not in df.columns:
                    print(f"Skipping {fp}: Header mismatch. Found {df.columns}")
                    continue

                # Parse Timestamp
                df[DB_COL_TIMESTAMP] = pd.to_datetime(df[CSV_COL_TIME], format='%m/%d/%y %H:%M:%S.%f')
                
                # Convert to UTC Epoch (float)
                df[DB_COL_TIMESTAMP] = df[DB_COL_TIMESTAMP].apply(to_utc_epoch)

                # Rename columns
                df = df.rename(columns={
                    CSV_COL_CODE: DB_COL_EVENT,
                    CSV_COL_PARAM: DB_COL_PARAM
                })
                
                # Keep only DB columns
                df = df[[DB_COL_TIMESTAMP, DB_COL_EVENT, DB_COL_PARAM]]
                
                # Drop Duplicates
                df = df.drop_duplicates()

                # --- Insert Discontinuity Marker ---
                if file_info['start'] in discontinuities:
                    # Calculate gap timestamp (start of this file)
                    gap_ts = to_utc_epoch(file_info['start'])
                    
                    gap_row = pd.DataFrame([{
                        DB_COL_TIMESTAMP: gap_ts, 
                        DB_COL_EVENT: -1, 
                        DB_COL_PARAM: -1
                    }])
                    df = pd.concat([gap_row, df], ignore_index=True)

                # Append to DB
                df.to_sql('logs', conn, if_exists='append', index=False)
                
            except Exception as e:
                print(f"Failed processing data for {fp}: {e}")

        conn.close()
        print(f"Saved {db_name}")


if __name__ == "__main__":
    # Configure paths
    INPUT_DIRECTORY = 'G:\\Python\\SPM_Data_Archive\\ACHD_Data\\Archive'
    OUTPUT_DIRECTORY = '../processed_dbs'
    
    if os.path.exists(INPUT_DIRECTORY):
        process_csv_directory(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
    else:
        print("Please configure INPUT_DIRECTORY path.")