import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from tkcalendar import Calendar
import os
import sys
import configparser
import threading
from datetime import datetime
from spmfunctions.process_datz import process_datz_files, move_datz_from_spm_subdirs
from spmfunctions.read_data import process_csv_files, read_int_cfg, read_and_process_pickles
from spmfunctions.misc_tools import create_int_directory, comb_gyr_det, get_nearest_date
from spmfunctions.plotting import plot_term, plot_coord
from spmfunctions.counts import det_counts
import spmfunctions.video_processing as video_processing
import warnings
import glob
import cv2
import importlib
warnings.filterwarnings("ignore")

CONFIG_FILE = "settings.ini"

def load_settings():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    return config

def save_settings(config):
    with open(CONFIG_FILE, 'w') as configfile:
        config.write(configfile)

class SPMGui:
    def __init__(self, root):
        self.root = root
        self.root.title("Signal Performance Measures")
        self.config = load_settings()
        # Intersection Folder Selection
        frame = ttk.Frame(root)
        frame.pack(pady=5)
        ttk.Label(frame, text="Intersection:").pack(side=tk.LEFT, padx=5)
        self.intersection_var = tk.StringVar()
        self.intersection_entry = ttk.Entry(frame, textvariable=self.intersection_var, width=40, state='readonly')
        self.intersection_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Browse", command=self.select_intersection_dir).pack(side=tk.LEFT)
        # Date Range Selector
        self.create_date_range_selector(root)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(expand=True, fill='both')
        self.create_data_processing_tab()
        self.create_plotting_tab()
        self.create_counts_tab()
        self.create_configuration_tab()
        self.create_video_processing_tab()

        self.load_saved_settings()

    def select_intersection_dir(self):
        folder_selected = filedialog.askdirectory(initialdir=os.path.join(self.spm_directory_var.get(), "Intersections"))
        if folder_selected:
            self.intersection_var.set(os.path.basename(folder_selected))
            self.datz_dir = os.path.join(folder_selected, "data", "datz")
            self.csv_dir = os.path.join(folder_selected, "data", "csv")
            self.pickle_dir = os.path.join(folder_selected, "data", "DataFrames")
            self.plt_dir = os.path.join(folder_selected, "Plotting")
            self.cfg_dir = os.path.join(folder_selected, "Configuration")
            self.cnt_dir = os.path.join(folder_selected, "counts")
            self.vid_dir = os.path.join(folder_selected, "Video Processing")
            try:
                self.int_cfg = read_int_cfg(self.cfg_dir, 'int_cfg.csv')
            except FileNotFoundError: 
                self.create_output_window()
                print(f"Configuration file not found in {self.cfg_dir}.")

    def load_saved_settings(self):
        if 'Settings' in self.config:
            if 'SPM_Directory' in self.config['Settings']:
                self.spm_directory_var.set(self.config['Settings']['SPM_Directory'])
            if 'DATZ_Translator_Directory' in self.config['Settings']:
                self.exe_dir.set(self.config['Settings']['DATZ_Translator_Directory'])

    def create_date_range_selector(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(pady=5)
        ttk.Label(frame, text="Date Range (yyyy_mm_dd or yyyy_mm_dd-yyyy_mm_dd):").pack(side=tk.LEFT, padx=5)
        self.date_range = ttk.Entry(frame, width=22)
        self.date_range.pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="Select", command=self.select_date_range).pack(side=tk.LEFT)

    def create_data_processing_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Data Processing")
        ttk.Label(tab, text="Time Increment (1, 15, 60):").grid(row=0, column=0, sticky='w')
        self.time_increment = ttk.Combobox(tab, values=[1, 15, 60], width=5)
        self.time_increment.set(15)
        self.time_increment.grid(row=0, column=1)
        ttk.Button(tab, text="Process Intersection DATZ Files", command=self.run_process_datz).grid(row=0, column=2, padx=20)
        ttk.Button(tab, text="Process Intersection CSV Files", command=self.run_process_csv).grid(row=1, column=2, padx=20)
        ttk.Button(tab, text="Batch Process DATZ Files", command=self.check_datz_warning).grid(row=0, column=3)
        ttk.Button(tab, text="Batch Process CSV Files", command=self.run_batch_csv).grid(row=1, column=3)

    def create_plotting_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Plotting")
        self.term_var = tk.BooleanVar()
        ttk.Checkbutton(tab, text="Phase Termination", variable=self.term_var).grid(row=1, column=0, sticky='w')
        self.term_line_var = tk.BooleanVar()
        ttk.Checkbutton(tab, text="Line", variable=self.term_line_var).grid(row=1, column=2, sticky='w')
        self.term_n_var = 10
        self.coord_var = tk.BooleanVar()
        ttk.Checkbutton(tab, text="Coordination / Split Diagram", variable=self.coord_var).grid(row=3, column=0, sticky='w')
        ttk.Button(tab, text="Generate Plots", command=self.run_generate_plots).grid(row=5, column=1)

    def create_counts_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Counts")
        ttk.Label(tab, text="Bin Length (Minutes):").grid(row=0, column=0, sticky='w')
        self.bin_length = ttk.Combobox(tab, values=["Cycle", 5, 10, 15, 30, 60], width=5)
        self.bin_length.set(60)
        self.bin_length.grid(row=0, column=1, sticky="w")
        self.pedestrian_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tab, text="Pedestrian: Include pedestrian counts.", variable=self.pedestrian_var).grid(row=1, column=0, columnspan=2, sticky='w')
        self.hourly_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab, text="Hourly: Report vehicle counts in hourly volume (For example, when Bin Length=15 Minutes, multiplies vehicle count by 4.)", variable=self.hourly_var).grid(row=2, column=0, columnspan=2, sticky='w')
        ttk.Button(tab, text="Calculate Counts", command=self.calculate_counts).grid(row=3, column=1)

    def create_configuration_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Configuration")
        ttk.Label(tab, text="New Intersection:").grid(row=0, column=0, sticky='w')
        self.new_intersection_entry = ttk.Entry(tab, width=40)
        self.new_intersection_entry.grid(row=0, column=1)
        ttk.Button(tab, text="Create", command=self.create_intersection).grid(row=0, column=2)
        ttk.Label(tab, text="SPM Directory:").grid(row=1, column=0, sticky='w')
        self.spm_directory_var = tk.StringVar()
        self.spm_directory_entry = ttk.Entry(tab, textvariable=self.spm_directory_var, width=80, state='readonly')
        self.spm_directory_entry.grid(row=1, column=1)
        ttk.Button(tab, text="Browse", command=self.select_spm_directory).grid(row=1, column=2)
        ttk.Label(tab, text="DATZ Translator Directory:").grid(row=2, column=0, sticky='w')
        self.exe_dir = tk.StringVar()
        self.exe_dir_entry = ttk.Entry(tab, textvariable=self.exe_dir, width=80, state='readonly')
        self.exe_dir_entry.grid(row=2, column=1)
        ttk.Button(tab, text="Browse", command=self.select_translator_directory).grid(row=2, column=2)

    def create_video_processing_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Video Processing")

        # --- Variables ---
        self.video_path_var = tk.StringVar()
        self.output_path_var = tk.StringVar()
        self.shape_csv_var = tk.StringVar()
        self.start_time_var = tk.StringVar(value="00:00:00")

        # Video File
        row = 0
        ttk.Button(tab, text="Browse Video File", command=self.browse_video).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        ttk.Entry(tab, textvariable=self.video_path_var, width=50, state='readonly').grid(row=row, column=1, columnspan=2, padx=10)

        # Output Video
        row += 1
        ttk.Button(tab, text="Browse Output Video", command=self.browse_output).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        ttk.Entry(tab, textvariable=self.output_path_var, width=50, state='readonly').grid(row=row, column=1, columnspan=2, padx=10)

        # Shape CSV (Video Cfg)
        row += 1
        ttk.Button(tab, text="Browse/Create Video Cfg", command=self.browse_shape_csv).grid(row=row, column=0, sticky='w', padx=10, pady=5)
        ttk.Entry(tab, textvariable=self.shape_csv_var, width=50, state='readonly').grid(row=row, column=1, columnspan=2, padx=10)

        # Start Time
        row += 1
        ttk.Label(tab, text="Start Time (HH:MM:SS):").grid(row=row, column=0, sticky='w', padx=10, pady=5)
        ttk.Entry(tab, textvariable=self.start_time_var).grid(row=row, column=1, sticky='w', padx=10)

        # Buttons
        row += 1
        ttk.Button(tab, text="Create Video Cfg", command=self.create_shape_config).grid(row=row, column=0, pady=15, padx=10)
        ttk.Button(tab, text="Process Video", command=self.process_video_full).grid(row=row, column=1, pady=15)

        # Add Turning Movement Button
        row += 1
        ttk.Button(tab, text="Process Turning Movements", command=self.process_turning_movements).grid(row=row, column=1, pady=5)

        # Detection Mode
        row += 1
        ttk.Label(tab, text="Detection Mode:").grid(row=row, column=0, sticky='e', padx=5)
        self.detection_mode_var = tk.StringVar(value="bg_sub")
        detection_frame = ttk.Frame(tab)
        detection_frame.grid(row=row, column=1, sticky='w', pady=5)
        ttk.Radiobutton(detection_frame, text="Background Subtraction", variable=self.detection_mode_var, value="bg_sub").pack(side='left')
        ttk.Radiobutton(detection_frame, text="YOLOv8", variable=self.detection_mode_var, value="yolo").pack(side='left')

        # Status Label
        self.video_status_label = ttk.Label(tab, text="", foreground="blue")
        self.video_status_label.grid(row=row+1, column=0, columnspan=3, pady=5)

    def create_output_window(self):
        self.output_window = tk.Toplevel(self.root)
        self.output_window.title("Output Log")
        self.output_text = tk.Text(self.output_window, wrap='word', state='disabled')
        self.output_text.pack(expand=True, fill='both')
        sys.stdout = TextRedirector(self.output_text, "stdout")
        sys.stderr = TextRedirector(self.output_text, "stderr")

    def select_date_range(self):
        top = tk.Toplevel(self.root)
        top.title("Select Date Range")
        self.use_range = tk.BooleanVar(value=False)
        def toggle_end_date():
            if self.use_range.get():
                end_cal.pack(pady=5)
                end_label.pack()
            else:
                end_cal.pack_forget()
                end_label.pack_forget()
        ttk.Checkbutton(top, text="Select Date Range", variable=self.use_range, command=toggle_end_date).pack()
        ttk.Label(top, text="Start Date:").pack()
        today = datetime.today()
        start_cal = Calendar(top, selectmode='day', year=today.year, month=today.month, day=today.day)
        start_cal.pack(pady=5)
        end_label = ttk.Label(top, text="End Date:")
        end_cal = Calendar(top, selectmode='day', year=today.year, month=today.month, day=today.day)
        def set_date_range():
            if self.use_range.get():
                start_date = start_cal.get_date()
                end_date = end_cal.get_date()
                formatted_start_date = datetime.strptime(start_date, "%m/%d/%y").strftime("%Y_%m_%d")
                formatted_end_date = datetime.strptime(end_date, "%m/%d/%y").strftime("%Y_%m_%d")
                self.date_range.delete(0, tk.END)
                self.date_range.insert(0, f"{formatted_start_date}-{formatted_end_date}")
            else:
                start_date = start_cal.get_date()
                formatted_start_date = datetime.strptime(start_date, "%m/%d/%y").strftime("%Y_%m_%d")
                self.date_range.delete(0, tk.END)
                self.date_range.insert(0, formatted_start_date)
            top.destroy()
        ttk.Button(top, text="OK", command=set_date_range).pack(pady=10)

    def check_datz_warning(self):
        response = messagebox.askyesno(
            "Warning",
            "Warning, if the DATZ files, including for different intersections, are different time increments, then this processing will not work correctly. Proceed?"
        )
        if response:
            print("Proceeding with DATZ processing...")
            self.run_batch_datz()
        else:
            print("Processing terminated.")

    def run_process_datz(self):
        self.create_output_window()
        threading.Thread(target=self.process_datz).start()

    def run_process_csv(self):
        self.create_output_window()
        threading.Thread(target=self.process_csv).start()

    def run_batch_datz(self):
        self.create_output_window()
        threading.Thread(target=self.batch_datz).start()

    def run_batch_csv(self):
        self.create_output_window()
        threading.Thread(target=self.batch_csv).start()

    def run_generate_plots(self):
        self.create_output_window()
        threading.Thread(target=self.generate_plots).start()

    def process_datz(self):
        print("Processing DATZ files...")
        move_datz_from_spm_subdirs(self.datz_dir)
        process_datz_files(self.datz_dir, self.exe_dir.get(), self.csv_dir, int(self.time_increment.get()))
        print("DATZ files processed successfully.")

    def process_csv(self):
        print("Processing CSV files...")
        process_csv_files(self.csv_dir, self.pickle_dir)
        print("CSV files processed successfully.")

    def batch_datz(self):
        self.int_dir = os.path.join(self.spm_directory_var.get(), 'Intersections')
        for intersection in os.listdir(self.int_dir):
            print(f"Processing {intersection}")
            self.batch_csv_dir = os.path.join(self.int_dir, intersection, 'Data', 'CSV')
            self.batch_datz_dir = os.path.join(self.int_dir, intersection, 'Data', 'DATZ')
            if not os.path.exists(self.batch_datz_dir):
                print(f"{self.batch_datz_dir} does not exist.")
                continue
            move_datz_from_spm_subdirs(self.batch_datz_dir)
            process_datz_files(self.batch_datz_dir, self.exe_dir.get(), self.batch_csv_dir, int(self.time_increment.get()))
        print("DATZ files processed successfully.")

    def batch_csv(self):
        self.int_dir = os.path.join(self.spm_directory_var.get(), 'Intersections')
        for intersection in os.listdir(self.int_dir):
            print(f'Processing {intersection}')
            self.batch_csv_dir = os.path.join(self.int_dir, intersection, 'Data', 'CSV')
            self.batch_pickle_dir = os.path.join(self.int_dir, intersection, 'Data', 'Dataframes')
            process_csv_files(self.batch_csv_dir, self.batch_pickle_dir)
        print("CSV files processed successfully.")    

    def generate_plots(self):
        if self.term_var.get():
            df = read_and_process_pickles(self.pickle_dir, self.date_range.get())
            print("Generating Phase Termination plot...")
            plot_term(df, self.plt_dir, self.intersection_var.get(), 
                      self.term_line_var.get(), 
                      sfx=self.date_range.get())
            print(f"Phase Termination plots saved at {self.plt_dir}.")
        if self.coord_var.get():    
            print("Reading and transforming data...")
            transforms = [(comb_gyr_det, {})]
            df = read_and_process_pickles(self.pickle_dir, self.date_range.get(), transforms)
            print("Generating Coordination / Split Diagram plots...")
            for date in df.Cycle_start.dt.date.unique():
                formatted_date = date.strftime('%Y_%m_%d')
                nearest_date = get_nearest_date(self.int_cfg['Ring-Barrier'].index, date)
                both_str = self.int_cfg['Ring-Barrier'].loc[nearest_date, 'B']
                both = list(map(int, both_str.split(','))) if pd.notna(both_str) else []
                det_srs = self.int_cfg['Arrivals'].T.dropna(how='any')[nearest_date]
                plot_coord(df[df.Cycle_start.dt.date == date], det_srs, self.plt_dir, both=both, sfx=formatted_date)
                print(f"Saved Coordination / Split Diagram plot for {formatted_date}.")

    def calculate_counts(self):
        bin_length_value = 'cycle' if self.bin_length.get().lower() == "cycle" else int(self.bin_length.get())
        df = read_and_process_pickles(self.pickle_dir, self.date_range.get())
        det_counts(df, self.int_cfg, bin_length_value, self.hourly_var.get(), self.pedestrian_var.get(), self.cnt_dir)

    def create_intersection(self):
        self.create_output_window()
        create_int_directory(self.new_intersection_entry.get(), self.spm_directory_var.get())
        print(f"{self.new_intersection_entry.get()} created successfully.")

    def select_spm_directory(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.spm_directory_var.set(folder_selected)
            if 'Settings' not in self.config:
                self.config['Settings'] = {}
            self.config['Settings']['SPM_Directory'] = folder_selected
            save_settings(self.config)

    def select_translator_directory(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.exe_dir.set(folder_selected)
            if 'Settings' not in self.config:
                self.config['Settings'] = {}
            self.config['Settings']['DATZ_Translator_Directory'] = folder_selected
            save_settings(self.config)

    def browse_video(self):
        default_dir = self.vid_dir if hasattr(self, 'vid_dir') and self.vid_dir else ""
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")],
            initialdir=default_dir
        )
        if path:
            self.video_path_var.set(path)
            dirname = os.path.dirname(path)
            basename = os.path.splitext(os.path.basename(path))[0]
            self.output_path_var.set(os.path.join(dirname, f"{basename}_overlayed.mp4"))

    def browse_output(self):
        default_dir = self.vid_dir if hasattr(self, 'vid_dir') and self.vid_dir else ""
        path = filedialog.asksaveasfilename(
            title="Save Output Video As",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4"), ("AVI", "*.avi")],
            initialdir=default_dir
        )
        if path:
            self.output_path_var.set(path)

    def browse_shape_csv(self):
        default_dir = self.cfg_dir if hasattr(self, 'cfg_dir') and self.cfg_dir else ""
        path = filedialog.askopenfilename(
            title="Select or Create Video Cfg File",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialdir=default_dir
        )
        if path:
            self.shape_csv_var.set(path)

    def get_date_range(self):
        date_str = self.date_range.get().strip()
        if '-' in date_str:
            start_date_str = date_str.split('-')[0]
        else:
            start_date_str = date_str
        try:
            start_date = datetime.strptime(start_date_str, "%Y_%m_%d").date()
            return start_date
        except ValueError:
            messagebox.showerror("Invalid Date", f"Cannot parse date: {start_date_str}")
            return None

    def make_timestamp(self, date, time_str):
        try:
            time_part = datetime.strptime(time_str, "%H:%M:%S").time()
            dt = datetime.combine(date, time_part)
            return pd.to_datetime(dt)
        except Exception as e:
            messagebox.showerror("Invalid Time", f"Error parsing time '{time_str}': {e}")
            return None

    def find_pickle_file(self, start_dt, end_dt):
        data_dir = self.pickle_dir
        if not os.path.exists(data_dir):
            messagebox.showerror("Path Error", f"DataFrames directory not found: {data_dir}")
            return None
        pattern = os.path.join(data_dir, "df_raw_*.pkl")
        candidates = []
        for pkl_file in glob.glob(pattern):
            base = os.path.basename(pkl_file)
            if base.startswith("df_raw_") and base.endswith(".pkl"):
                name = base[7:-4]
                parts = name.split('-')
                if len(parts) != 2:
                    continue
                try:
                    pkl_start_str = parts[0].replace('_', ':').replace('-', ' ')
                    pkl_end_str = parts[1].replace('_', ':').replace('-', ' ')
                    # Handle 2400 case for end time
                    if pkl_end_str[-4:-2] == "24":
                        # Replace hour with 23 and minute with 59
                        pkl_end_str = pkl_end_str[:-4] + "23" + "59"
                    pkl_start = datetime.strptime(pkl_start_str, "%Y:%m:%d:%H%M")
                    pkl_end = datetime.strptime(pkl_end_str, "%Y:%m:%d:%H%M")
                    pkl_start = pd.to_datetime(pkl_start)
                    pkl_end = pd.to_datetime(pkl_end)
                    if pkl_start <= end_dt and pkl_end >= start_dt:
                        candidates.append((pkl_start, pkl_file))
                except Exception as e:
                    continue
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
        return None

    def create_shape_config(self):
        if not self.video_path_var.get():
            messagebox.showwarning("Missing Input", "Please select a video file first.")
            return
        if not self.shape_csv_var.get():
            messagebox.showwarning("Missing Input", "Please select a Video Cfg (CSV) file.")
            return
        self.create_output_window()
        print("Starting shape configuration interface...")
        processor = video_processing.VideoProcessor()
        try:
            processor.read_video(self.video_path_var.get())
            if os.path.exists(self.shape_csv_var.get()):
                processor.load_shapes_from_csv(self.shape_csv_var.get())
                print(f"Loaded existing shapes from {self.shape_csv_var.get()}")
            processor.draw_shapes_interface()
            processor.save_shapes_to_csv(self.shape_csv_var.get())
            print(f"Shapes saved to {self.shape_csv_var.get()}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create shape config: {e}")

    def process_video_full(self):
        if not all([self.video_path_var.get(), self.output_path_var.get(), 
                    self.shape_csv_var.get(), self.intersection_var.get()]):
            messagebox.showwarning("Missing Input", "Please fill all required fields.")
            return

        start_time_str = self.start_time_var.get()
        date = self.get_date_range()
        if not date:
            return

        start_dt = self.make_timestamp(date, start_time_str)
        if not start_dt:
            return

        cap = cv2.VideoCapture(self.video_path_var.get())
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open video to read duration.")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        if fps <= 0 or frame_count <= 0:
            messagebox.showerror("Error", "Could not read video FPS or frame count.")
            return

        video_duration = frame_count / fps
        end_dt = start_dt + pd.Timedelta(seconds=video_duration)

        pickle_path = self.find_pickle_file(start_dt, end_dt)
        if not pickle_path:
            messagebox.showerror("Not Found", f"No pickle file found covering {start_dt} to {end_dt}")
            return

        print(f"Using pickle file: {pickle_path}")
        print(f"Processing video from {start_dt} to {end_dt}")

        self.create_output_window()
        threading.Thread(
            target=self.run_video_processing,
            args=(pickle_path, start_dt),
            daemon=True
        ).start()

    def run_video_processing(self, pickle_path, start_dt):
        try:
            processor = video_processing.VideoProcessor()
            processor.process_video(
                video_path=self.video_path_var.get(),
                pickle_path=pickle_path,
                output_path=self.output_path_var.get(),
                shape_csv=self.shape_csv_var.get(),
                start_time_str=start_dt.strftime("%Y-%m-%d %H:%M:%S")
            )
            self.video_status_label.config(text="✅ Video processing completed.", foreground="green")
        except Exception as e:
            messagebox.showerror("Processing Failed", f"Error: {e}")
            self.video_status_label.config(text="❌ Failed", foreground="red")

    def process_turning_movements(self):

        importlib.reload(video_processing)

        if not all([self.video_path_var.get(), self.shape_csv_var.get(), self.intersection_var.get()]):
            messagebox.showwarning("Missing Input", "Please select video, shape CSV, and intersection.")
            return
        start_time_str = self.start_time_var.get()
        date = self.get_date_range()
        if not date:
            return
        video_start_dt = self.make_timestamp(date, start_time_str)
        if not video_start_dt:
            return
        output_csv = filedialog.asksaveasfilename(
            title="Save Turning Movement Counts As",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialdir=self.cnt_dir if hasattr(self, 'cnt_dir') else ""
        )
        if not output_csv:
            return
        self.create_output_window()
        threading.Thread(
            target=self.run_turning_count,
            args=(video_start_dt, output_csv),
            daemon=True
        ).start()

    def run_turning_count(self, video_start_dt, output_csv):
        try:
            processor = video_processing.VideoProcessor()
            processor.count_turning_movements(
                video_path=self.video_path_var.get(),
                shape_csv=self.shape_csv_var.get(),
                output_csv=output_csv,
                start_time_str=video_start_dt,
                detection_mode=self.detection_mode_var.get()
            )
            self.video_status_label.config(text="✅ Turning counts completed.", foreground="green")
        except Exception as e:
            messagebox.showerror("Failed", f"Error in turning count: {e}")
            self.video_status_label.config(text="❌ Failed", foreground="red")

class TextRedirector:
    def __init__(self, widget, tag):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state='normal')
        self.widget.insert('end', str, (self.tag,))
        self.widget.configure(state='disabled')
        self.widget.see('end')

    def flush(self):
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = SPMGui(root)
    root.mainloop()