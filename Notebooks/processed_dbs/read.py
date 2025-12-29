import sqlite3
from datetime import datetime
from zoneinfo import ZoneInfo


def analyze_db(db_path):
    # --- Configuration ---
    TABLE_NAME = "logs"  # Change this to your table name
    COLUMN_NAME = "timestamp"      # Change this if your column is named differently
    TARGET_TZ = ZoneInfo("America/Denver") # MST/MDT
    
    try:
        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query for count, min, and max
        query = f"SELECT COUNT(*), MIN({COLUMN_NAME}), MAX({COLUMN_NAME}) FROM {TABLE_NAME}"
        cursor.execute(query)
        
        count, min_ts, max_ts = cursor.fetchone()

        if count == 0:
            print("The table is empty.")
            return

        # Helper to convert Unix UTC to MST Datetime
        def to_mst(unix_ts):
            if unix_ts is None: return "N/A"
            # Create UTC datetime, then convert to Mountain Time
            utc_dt = datetime.fromtimestamp(unix_ts, tz=ZoneInfo("UTC"))
            return utc_dt.astimezone(TARGET_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')

        # Output Results
        print(f"--- Database Analysis: {db_path} ---")
        print(f"Total Records: {count:,}")
        print(f"Earliest (MST): {to_mst(min_ts)}")
        print(f"Latest   (MST): {to_mst(max_ts)}")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Provide the path to your .db or .sqlite file here
    path_to_db = "./Notebooks/processed_dbs/271_data.db"
    #path_to_db = "./Intersections/Franklin & Aviation/Data/spm_data.db"
    analyze_db(path_to_db)