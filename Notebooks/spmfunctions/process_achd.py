import os
import csv
from datetime import datetime, timedelta
import re
import shutil
from spmfunctions.process_datz import concat_csv_files

def read_achd_int_key(key_file):
    int_key = {}
    with open(key_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            int_key[row[0]] = row[1]
    return int_key

def round_to_nearest_hour(dt, direction='down'):
    if direction == 'down':
        return dt.replace(minute=0, second=0, microsecond=0)
    elif direction == 'up':
        return (dt + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)

def process_achd_files(key_dir, intersection_dir):
    key_file = os.path.join(key_dir, '_ACHD_int_key.csv')
    int_key = read_achd_int_key(key_file)
    
    for int_num, int_name in int_key.items():
        # Create a list of CSV files that start with the intersection number
        csv_files = [f for f in os.listdir(key_dir) if f.startswith(int_num) and f.endswith('.csv')]
        
        # Determine contiguity of the segments
        segments = find_contiguous_segments_achd(csv_files)
        
        # Process each segment
        for segment in segments:
            if not segment:
                continue
            
            # Determine the start and end times from the first and last files in the segment
            with open(os.path.join(key_dir, segment[0]), 'r', encoding='utf-8') as file:
                for _ in range(5):
                    file.readline()  # Skip the first 5 lines
                first_line = file.readline().strip()
                start_time = datetime.strptime(first_line.split(',')[0], '%m/%d/%y %H:%M:%S.%f')
                start_time = round_to_nearest_hour(start_time, 'down')
            
            with open(os.path.join(key_dir, segment[-1]), 'r', encoding='utf-8') as file:
                last_line = file.readlines()[-1].strip()
                end_time = datetime.strptime(last_line.split(',')[0], '%m/%d/%y %H:%M:%S.%f')
                end_time = round_to_nearest_hour(end_time, 'up')
                if end_time.hour == 0:
                    end_time -= timedelta(minutes=1)
            
            # Create the output directory
            output_dir = os.path.join(intersection_dir, f"ACHD_{int_name}", "Data", "CSV")
            os.makedirs(output_dir, exist_ok=True)
            
            # Create the output file name
            if end_time.hour==23 and end_time.minute==59:
                end_time_str = end_time.strftime('%Y_%m_%d_%H%M'.replace('2359','2400'))
            else:
                end_time_str = end_time.strftime('%Y_%m_%d_%H%M')
            output_file = os.path.join(output_dir, f"compiled_{start_time.strftime('%Y_%m_%d_%H%M')}-{end_time_str}.csv")
            
            # Concatenate the CSV files
            segment_csv_files = [os.path.join(key_dir, f) for f in segment]
            concat_csv_files(segment_csv_files, output_file, skip=5)
            print(f"Saved: {output_file}")
        
        # Archive the original CSV files
        archive_dir = os.path.join(key_dir, "Archive")
        os.makedirs(archive_dir, exist_ok=True)
        for csv_file in csv_files:
            shutil.move(os.path.join(key_dir, csv_file), os.path.join(archive_dir, csv_file))
        print(f"Archived {len(csv_files)} files to {archive_dir}")

def parse_achd_filename_timestamp(filename):
    match = re.search(r"_(\d{8})T0000\.csv$", filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d")
    return None

def find_contiguous_segments_achd(filenames):
    filenames = sorted(filenames, key=lambda f: parse_achd_filename_timestamp(f))
    segments = []
    current_segment = []
    expected_date = None
    
    for file in filenames:
        file_date = parse_achd_filename_timestamp(file)
        
        if expected_date is None or file_date == expected_date:
            current_segment.append(file)
        else:
            segments.append(current_segment)
            current_segment = [file]

        # Determine the next expected date
        if file_date.day == 1:
            expected_date = file_date.replace(day=16)
        elif file_date.day == 16:
            next_month = file_date.replace(day=1) + timedelta(days=32)
            expected_date = next_month.replace(day=1)
    
    if current_segment:
        segments.append(current_segment)
    
    return segments