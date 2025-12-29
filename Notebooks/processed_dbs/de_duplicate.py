import os
import glob
import sqlite3

def deduplicate_databases(input_dir):
    """
    Scans a directory for .db files and removes duplicate rows 
    from the 'logs' table based on timestamp, event_type, and parameter.
    """
    # 1. Find all .db files
    db_files = glob.glob(os.path.join(input_dir, "*.db"))
    
    if not db_files:
        print(f"No .db files found in {input_dir}")
        return

    print(f"Found {len(db_files)} databases. Starting deduplication...")

    for db_path in db_files:
        filename = os.path.basename(db_path)
        
        try:
            # 2. Connect to the database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 3. Execute De-duplication SQL
            # We keep the row with the lowest rowid (oldest insert) and delete the rest
            cursor.execute('''
                DELETE FROM logs 
                WHERE rowid NOT IN (
                    SELECT MIN(rowid) 
                    FROM logs 
                    GROUP BY timestamp, event_type, parameter
                );
            ''')
            
            # 4. Check results and commit
            rows_removed = cursor.rowcount
            conn.commit()
            
            if rows_removed > 0:
                print(f"  [FIXED] {filename}: Removed {rows_removed} duplicate rows.")
            else:
                print(f"  [OK]    {filename}: No duplicates found.")
                
        except sqlite3.Error as e:
            print(f"  [ERROR] {filename}: Database error - {e}")
        except Exception as e:
            print(f"  [ERROR] {filename}: Unexpected error - {e}")
        finally:
            if conn:
                conn.close()

if __name__ == "__main__":
    # --- Configuration ---
    # Change this to the folder containing your .db files
    TARGET_DIRECTORY = './Notebooks/processed_dbs'
    
    if os.path.exists(TARGET_DIRECTORY):
        deduplicate_databases(TARGET_DIRECTORY)
    else:
        print(f"Directory not found: {TARGET_DIRECTORY}")