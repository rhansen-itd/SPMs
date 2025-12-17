import os
import subprocess
import re
import shutil
from datetime import datetime, timedelta

def parse_filename_timestamp(filename):
    match = re.search(r"(\d{4}_\d{2}_\d{2}_\d{4})", filename)
    if match:
        return datetime.strptime(match.group(1), "%Y_%m_%d_%H%M")
    return None

def round_up_to_increment(time, increment):
    """Rounds up time to the next multiple of increment after the start of the hour."""
    minutes_past_hour = time.minute + time.second / 60
    next_increment = (minutes_past_hour // increment + 1) * increment
    return time.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=next_increment)

def find_contiguous_segments(filenames, time_increment):
    filenames = sorted(filenames, key=lambda f: parse_filename_timestamp(f))
    segments = []
    current_segment = []
    expected_time = None
    
    for file in filenames:
        file_time = parse_filename_timestamp(file)
        
        if expected_time is None or file_time == expected_time:
            current_segment.append(file)
        else:
            segments.append(current_segment)
            current_segment = [file]

        expected_time = round_up_to_increment(file_time, time_increment)
    
    if current_segment:
        segments.append(current_segment)
    
    return segments

def concat_csv_files(csv_files, output_file, skip=8):
    """Concatenate CSV files using basic file operations"""
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for csv_file in csv_files:
            if not os.path.exists(csv_file):
                print(f"Warning: CSV file {csv_file} not found.")
                continue
                
            with open(csv_file, 'r', encoding='utf-8', errors='replace') as infile:
                # Skip the first 8 lines of the input file
                for _ in range(skip):
                    next(infile)
                
                # Write all lines from the input file to the output file
                for line in infile:
                    outfile.write(line)
    
    return True

def archive_processed_files(datz_dir, successful_datz_files, failed_files):
    """
    Move processed datZ files to an Archive subfolder.
    Creates the Archive folder if it doesn't exist.
    """
    archive_dir = os.path.join(datz_dir, "Archive")
    
    # Create Archive directory if it doesn't exist
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
        print(f"Created archive directory: {archive_dir}")
    
    # Move successful files
    for datz_file in successful_datz_files:
        source_path = os.path.join(datz_dir, datz_file)
        dest_path = os.path.join(archive_dir, datz_file)
        
        try:
            shutil.move(source_path, dest_path)
        except Exception as e:
            print(f"Warning: Could not move {datz_file} to archive: {str(e)}")
    
    # Move failed files
    for datz_file in failed_files:
        source_path = os.path.join(datz_dir, datz_file)
        dest_path = os.path.join(archive_dir, f"FAILED_{datz_file}")
        
        try:
            shutil.move(source_path, dest_path)
        except Exception as e:
            print(f"Warning: Could not move failed file {datz_file} to archive: {str(e)}")
    
    print(f"Moved {len(successful_datz_files)} successful and {len(failed_files)} failed files to archive")

def move_datz_from_spm_subdirs(datz_dir):
    """
    Moves all files (including .datZ) from subdirectories starting with 'SPM_'
    into the root datz_dir, then deletes those subdirectories.

    Args:
        datz_dir (str): Path to the main directory containing .datZ files and subfolders.
    """
    for root, dirs, files in os.walk(datz_dir, topdown=True):
        for d in dirs:
            if d.startswith("SPM_"):
                spm_dir = os.path.join(root, d)
                print(f"Found SPM subdirectory: {spm_dir}")

                for item in os.listdir(spm_dir):
                    src_path = os.path.join(spm_dir, item)
                    dest_path = os.path.join(datz_dir, item)

                    # Handle duplicate filenames
                    if os.path.exists(dest_path):
                        base, ext = os.path.splitext(item)
                        counter = 1
                        while os.path.exists(dest_path):
                            dest_path = os.path.join(datz_dir, f"{base}_{counter}{ext}")
                            counter += 1

                    shutil.move(src_path, dest_path)

                shutil.rmtree(spm_dir)
                print(f"Moved contents and removed: {spm_dir}")

        break  # only handle top-level subdirectories

def process_datz_files(datz_dir, exe_dir, output_dir, time_increment):
    all_files = os.listdir(datz_dir)
    csv_files = [f for f in all_files if f.endswith(".csv")]
    datz_files = [f for f in all_files if f.endswith(".datZ") and f.replace(".datZ", '.csv') not in csv_files]
    
    if not datz_files:
        print("No .datZ files found in the directory.")
        return
    
    exe_files = [f for f in os.listdir(exe_dir) if f.endswith(".exe") or f.endswith(".dll")]
    
    for exe_file in exe_files:
        shutil.copy(os.path.join(exe_dir, exe_file), datz_dir)
    
    exe_path = os.path.join(datz_dir, "highrestranslator.exe")
    
    # Track successful and failed conversions
    successful_csvs = [(f.replace('.csv', '.datZ'), os.path.join(datz_dir, f)) for f in csv_files]
    failed_files = []
    
    # First, attempt to process all datZ files
    for datz_file in datz_files:
        datz_path = os.path.join(datz_dir, datz_file)
        csv_file = os.path.join(datz_dir, datz_file.replace(".datZ", ".csv"))
        
        try:
            subprocess.run([exe_path, datz_path], check=True, cwd=datz_dir)
            
            if os.path.exists(csv_file):
                # Store the original datZ filename along with the CSV path for timestamp extraction
                successful_csvs.append((datz_file, csv_file))
            else:
                failed_files.append(datz_file)
                print(f"Warning: CSV file {csv_file} was not created.")
        except subprocess.CalledProcessError:
            failed_files.append(datz_file)
            print(f"Error: Failed to process {datz_file}")
    
    # Now find contiguous segments based on successfully processed files
    if successful_csvs:
        # Extract original datZ filenames for segment determination
        successful_datz_files = [item[0] for item in successful_csvs]
        segments = find_contiguous_segments(successful_datz_files, time_increment)
        
        # Create a mapping from datZ filename to CSV path
        csv_map = {datz: csv for datz, csv in successful_csvs}
        
        # Process each segment
        for segment in segments:
            if not segment:
                continue
            
            start_time = parse_filename_timestamp(segment[0]).strftime("%Y_%m_%d_%H%M")
            end_time = (parse_filename_timestamp(segment[-1]) + timedelta(minutes=time_increment)).strftime("%Y_%m_%d_%H%M")
            output_csv = os.path.join(output_dir, f"compiled_{start_time}-{end_time}.csv")
            
            # Get CSV files corresponding to this segment
            segment_csv_files = [csv_map[datz] for datz in segment]
            
            success = concat_csv_files(segment_csv_files, output_csv)
            if success:
                print(f"Saved: {output_csv}")
        
        # Clean up temporary CSV files
        for _, csv_file in successful_csvs:
            os.remove(csv_file)
    
    # Remove copied executable files
    for exe_file in exe_files:
        os.remove(os.path.join(datz_dir, exe_file))
    
    # Archive processed files
    successful_datz_files = [item[0] for item in successful_csvs]
    archive_processed_files(datz_dir, successful_datz_files, failed_files)
    
    # Report failed files
    if failed_files:
        print("\nThe following files could not be processed:")
        for file in failed_files:
            print(f"  - {file}")
    
    print("Processing complete. Executable and DLL files removed from datZ directory.")