# spmfunctions/video_processing.py
import cv2
import pandas as pd
import numpy as np
import csv
from datetime import datetime, timedelta
import os
from spmfunctions.misc_tools import detector_status, phase_status, overlap_status, comb_gyr_det
import tkinter as tk
from tkinter import simpledialog, messagebox
from shapely.geometry import Point, Polygon
from PIL import Image

def create_video_from_image(image_path, output_path, duration=300, fps=10):
    """
    Create an MP4 video from a still image.
    
    Args:
        image_path (str): Path to input JPG image
        output_path (str): Path for output MP4 video
        duration (int): Video duration in seconds (default: 300)
        fps (int): Frames per second (default: 10)
    """
    # Load and convert image
    img = Image.open(image_path).convert('RGB')
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Calculate total frames
    total_frames = duration * fps
    
    # Initialize video writer
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for _ in range(total_frames):
        out.write(frame)
    
    # Cleanup
    out.release()
    print(f"Video saved: {output_path}")


class EnhancedIOUTracker:
    """
    Enhanced IOU-based tracker for vehicles.
    Improvements:
    - Uses trajectory (velocity) and size history for better continuity.
    - Attempts to predict positions of disappeared objects for reacquisition.
    - Uses intersection area to prevent dropping tracks entering it and
      to aid in reacquiring objects that appear inside it.
    - Tuned for fast-moving objects by prioritizing trajectory prediction.
    """
    def __init__(self, max_disappeared=15, iou_threshold=0.05, # Lowered IOU threshold
                 history_length=5, max_reacquire_frames=30,
                 max_distance_factor=3.0): # New parameter for trajectory tolerance
        self.next_id = 0
        # obj_id -> {'bbox': (x1,y1,x2,y2), 'history': [...], 'velocity': (vx,vy),
        #            'disappeared': int, 'in_intersection': bool}
        self.objects = {}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.history_length = history_length
        self.max_reacquire_frames = max_reacquire_frames
        self.max_distance_factor = max_distance_factor # How many "bbox widths" an object can move

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        union = float(boxAArea + boxBArea - interArea)
        return interArea / union if union > 0 else 0.0

    def get_centroid(self, bbox):
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def get_size(self, bbox):
        """Estimate object size (e.g., average of width and height)"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return (w + h) / 2.0 # Or use max(w,h) or sqrt(w*h) if preferred

    def add_history(self, obj_data, new_bbox):
        if 'history' not in obj_data:
            obj_data['history'] = []
        obj_data['history'].append(new_bbox)
        if len(obj_data['history']) > self.history_length:
            obj_data['history'].pop(0)

    def calculate_velocity(self, obj_data):
        history = obj_data.get('history', [])
        if len(history) < 2:
            return (0.0, 0.0)
        # Velocity based on last two history points
        c1 = self.get_centroid(history[-2])
        c2 = self.get_centroid(history[-1])
        return (c2[0] - c1[0], c2[1] - c1[1])

    def predict_position(self, obj_data):
        if obj_data['disappeared'] > self.max_reacquire_frames:
             # Don't predict if disappeared for too long
            return None

        bbox = obj_data['bbox']
        vx, vy = obj_data['velocity']

        # Simple prediction: current pos + velocity
        pred_cx = (bbox[0] + bbox[2]) / 2.0 + vx
        pred_cy = (bbox[1] + bbox[3]) / 2.0 + vy

        # Estimate size for distance threshold
        size = self.get_size(bbox)

        # Estimate size change (simple: assume same size)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        pred_x1 = pred_cx - w / 2.0
        pred_y1 = pred_cy - h / 2.0
        pred_x2 = pred_cx + w / 2.0
        pred_y2 = pred_cy + h / 2.0

        return (pred_x1, pred_y1, pred_x2, pred_y2), size

    def point_in_polygon(self, point, polygon_points):
        """Check if a point is inside a polygon."""
        polygon = Polygon(polygon_points)
        point_obj = Point(point)
        return polygon.contains(point_obj)


    def is_plausible_reacquisition(self, predicted_data, new_bbox):
        """Check if a new detection is a plausible match for a predicted disappeared object."""
        if predicted_data is None:
            return False
        predicted_bbox, predicted_size = predicted_data

        # --- Focus on distance for plausibility ---
        pred_centroid = self.get_centroid(predicted_bbox)
        new_centroid = self.get_centroid(new_bbox)

        # Calculate distance between centroids
        distance = np.linalg.norm(np.array(pred_centroid) - np.array(new_centroid))

        # Allowable distance based on object size and velocity factor
        # You might adjust this logic or use a fixed threshold if size is unreliable
        max_allowed_distance = self.max_distance_factor * predicted_size

        # Check if within distance threshold
        is_close = distance <= max_allowed_distance

        # Optional: Still check for a minimal IOU to ensure it's not matching a completely different object
        # This can prevent false positives if two objects pass close to each other's paths
        # iou = self.compute_iou(predicted_bbox, new_bbox)
        # has_min_iou = iou > (self.iou_threshold / 5.0) # Very relaxed IOU check

        # return is_close and has_min_iou # Combine distance and relaxed IOU
        return is_close # Prioritize distance

    def associate_detections_to_tracks(self, new_boxes, intersection_polygon):
        """Core association logic."""
        updated_objects = {}
        used_boxes = set()

        # --- 1. Try to match active and recently disappeared tracks to new detections ---
        tracks_to_match = [obj_id for obj_id, data in self.objects.items()
                           if data['disappeared'] <= self.max_reacquire_frames]

        # Sort tracks: prioritize active ones, then recently disappeared
        tracks_to_match.sort(key=lambda oid: self.objects[oid]['disappeared'])

        for obj_id in tracks_to_match:
            obj_data = self.objects[obj_id]
            best_score = -1 # Using a combined score now
            best_idx = -1

            for i, new_box in enumerate(new_boxes):
                if i in used_boxes:
                    continue

                score = 0
                iou = self.compute_iou(obj_data['bbox'], new_box)

                if obj_data['disappeared'] == 0:
                    # Active track: prioritize IOU (but with lowered threshold)
                    if iou > self.iou_threshold:
                        score = iou
                else:
                    # Disappeared track: prioritize prediction distance
                    predicted_data = self.predict_position(obj_data)
                    if self.is_plausible_reacquisition(predicted_data, new_box):
                        predicted_bbox, _ = predicted_data
                        pred_centroid = self.get_centroid(predicted_bbox)
                        new_centroid = self.get_centroid(new_box)
                        pred_dist = np.linalg.norm(np.array(pred_centroid) - np.array(new_centroid))
                        # Inverse of distance as a component of score (closer is better)
                        # Avoid division by zero
                        dist_score = 1.0 / (1.0 + pred_dist)

                        # Combine IOU and prediction closeness, but weight distance more heavily
                        # Adjust weights (0.3, 0.7) as needed. Higher weight for dist_score favors trajectory.
                        score = 0.3 * iou + 0.7 * dist_score

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx != -1 and best_score > 0:
                # Match found
                matched_box = new_boxes[best_idx]
                updated_objects[obj_id] = {
                    'bbox': matched_box,
                    'disappeared': 0,
                    'in_intersection': (intersection_polygon is not None and
                                        self.point_in_polygon(self.get_centroid(matched_box),
                                                              intersection_polygon))
                }
                self.add_history(updated_objects[obj_id], matched_box)
                updated_objects[obj_id]['velocity'] = self.calculate_velocity(updated_objects[obj_id])
                used_boxes.add(best_idx)
            else:
                # No match found, increment disappeared
                obj_data['disappeared'] += 1
                # Check if it moved into the intersection before disappearing
                was_in_intersection = obj_data.get('in_intersection', False)
                is_now_in_intersection = (intersection_polygon is not None and
                                          self.point_in_polygon(self.get_centroid(obj_data['bbox']),
                                                                intersection_polygon))
                # If it just entered the intersection, don't drop it immediately
                # even if disappeared count exceeds max_disappeared
                if was_in_intersection or is_now_in_intersection:
                    obj_data['in_intersection'] = True
                    # Keep it for reacquisition longer if in intersection
                    if obj_data['disappeared'] <= self.max_reacquire_frames:
                         updated_objects[obj_id] = obj_data
                elif obj_data['disappeared'] <= self.max_disappeared:
                     updated_objects[obj_id] = obj_data


        # --- 2. Handle new detections (unmatched boxes) ---
        for i, box in enumerate(new_boxes):
            if i not in used_boxes:
                box_centroid = self.get_centroid(box)
                new_obj_id = self.next_id
                new_obj_data = {
                    'bbox': box,
                    'disappeared': 0,
                    'in_intersection': (intersection_polygon is not None and
                                        self.point_in_polygon(box_centroid, intersection_polygon))
                }

                # Special handling if detection appears inside intersection
                if new_obj_data['in_intersection']:
                    # Search recently disappeared tracks for a plausible origin
                    best_candidate_id = None
                    best_candidate_score = -1
                    for obj_id, obj_data in self.objects.items():
                        # Consider tracks that were recently active or in/near intersection
                        if (obj_data['disappeared'] > 0 and
                            obj_data['disappeared'] <= self.max_reacquire_frames):

                            predicted_data = self.predict_position(obj_data)
                            if self.is_plausible_reacquisition(predicted_data, box):
                                predicted_bbox, _ = predicted_data
                                pred_centroid = self.get_centroid(predicted_bbox)
                                pred_dist = np.linalg.norm(np.array(pred_centroid) - np.array(box_centroid))
                                dist_score = 1.0 / (1.0 + pred_dist)

                                # Size similarity (area ratio)
                                pred_area = (predicted_bbox[2] - predicted_bbox[0]) * (predicted_bbox[3] - predicted_bbox[1])
                                new_area = (box[2] - box[0]) * (box[3] - box[1])
                                size_ratio = min(pred_area, new_area) / max(pred_area, new_area)
                                size_score = size_ratio if size_ratio > 0 else 0

                                # Prioritize distance for score in intersection reacq too
                                candidate_score = 0.7 * dist_score + 0.3 * size_score

                                if candidate_score > best_candidate_score:
                                    best_candidate_score = candidate_score
                                    best_candidate_id = obj_id

                    if best_candidate_id is not None:
                        # Reassociate with existing ID
                        new_obj_id = best_candidate_id
                        # print(f"Reassociated new detection inside intersection (ID {new_obj_id})") # Optional debug

                # Add new object (or update reassociated one)
                self.add_history(new_obj_data, box)
                new_obj_data['velocity'] = self.calculate_velocity(new_obj_data)
                updated_objects[new_obj_id] = new_obj_data

                # Only increment next_id if it was truly a new object
                if new_obj_id == self.next_id:
                    self.next_id += 1

        return updated_objects

    def update(self, new_boxes, intersection_polygon=None):
        """
        Update the tracker with new detections.
        :param new_boxes: List of tuples (x1, y1, x2, y2)
        :param intersection_polygon: List of (x, y) tuples defining the intersection area.
        :return: Dictionary of tracked objects {obj_id: (x1, y1, x2, y2)}
        """
        if len(new_boxes) == 0:
            # All objects are now disappeared
            for obj_data in self.objects.values():
                obj_data['disappeared'] += 1
            # Keep disappeared tracks for potential reacquisition
            # (logic handled in association step, but cleanup here too)
            self.objects = {k: v for k, v in self.objects.items()
                            if v['disappeared'] <= max(self.max_disappeared, self.max_reacquire_frames)}
            # Return current state
            return {obj_id: data['bbox'] for obj_id, data in self.objects.items()}

        # Perform association
        updated_objects = self.associate_detections_to_tracks(new_boxes, intersection_polygon)

        # Update internal state
        self.objects = updated_objects

        # Return simple dict for compatibility
        return {obj_id: data['bbox'] for obj_id, data in self.objects.items()}


class VideoProcessor:
    def __init__(self):
        self.cap = None
        self.fps = 0
        self.width = 0
        self.height = 0
        self.shapes = []
        self.current_shape = []
        self.mode = 'loop'
        self.color = (0, 255, 0)
        self.input_val = 1
        self.phase = 1

    def read_video(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return self.cap, self.fps, self.width, self.height

    def draw_shape(self, img, shape):
        dir_colors = {'N': (255, 0, 0), 'E': (0, 255, 0), 'S': (0, 0, 255), 'W': (255, 255, 0)}
        if shape['type'] == 'loop':
            pts = np.array(shape['points'], dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=shape['color'], thickness=2)
        elif shape['type'] == 'stopbar':
            pt1, pt2 = shape['points']
            cv2.line(img, pt1, pt2, color=(0, 0, 255), thickness=2)
        elif shape['type'] == 'approach':
            pt1, pt2 = shape['points']
            color = dir_colors.get(shape['direction'], (128, 128, 128))
            cv2.line(img, pt1, pt2, color=color, thickness=3)
            mid_x = (pt1[0] + pt2[0]) // 2
            mid_y = (pt1[1] + pt2[1]) // 2
            cv2.putText(img, shape['direction'], (mid_x, mid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_shape.append((x, y))
            if len(self.current_shape) == 4 and self.mode == 'loop':
                self.shapes.append({
                    'type': 'loop',
                    'points': list(self.current_shape),
                    'color': self.color,
                    'input': self.input_val
                })
                self.current_shape = []
            elif len(self.current_shape) == 2 and self.mode == 'stopbar':
                self.shapes.append({
                    'type': 'stopbar',
                    'points': list(self.current_shape),
                    'phase': self.phase
                })
                self.current_shape = []
            elif len(self.current_shape) == 2 and self.mode == 'approach':
                root = tk.Tk()
                root.withdraw()
                direction = simpledialog.askstring("Direction", "Enter direction (N, E, S, W):", parent=root)
                root.destroy()
                if direction and direction.upper() in ['N', 'E', 'S', 'W']:
                    self.shapes.append({
                        'type': 'approach',
                        'points': list(self.current_shape),
                        'direction': direction.upper()
                    })
                else:
                    print("Invalid direction. Use N, E, S, or W.")
                self.current_shape = []

    def draw_shapes_interface(self):
        """Interactive interface for drawing shapes with Tkinter input dialogs and edit mode."""
        if not self.cap:
            raise ValueError("Video not loaded")
        ret, first_frame = self.cap.read()
        if not ret:
            raise ValueError("Cannot read first frame")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        image = first_frame.copy()
        cv2.namedWindow("Draw Shapes")
        cv2.setMouseCallback("Draw Shapes", self.mouse_callback)

        root = tk.Tk()
        root.withdraw()
        instruction_window = tk.Toplevel()
        instruction_window.title("Draw Shapes Instructions")
        instruction_window.attributes('-topmost', True)
        instruction_window.resizable(False, False)
        instruction_text = ("""Instructions:
    - Press 'l' to switch to loop mode (4 points)
    - Press 's' to switch to stop bar mode (2 points)
    - Press 'a' to switch to approach mode (2 points, then enter N/E/S/W)
    - Press 'c' to change color (for loops)
    - Press 'i' to set input value (for loops)
    - Press 'p' to set phase value (for stop bars)
    - Press 'u' to undo last action
    - Press 'e' to enter/exit edit mode
    - In edit mode: 'n'/'p' to cycle next and previous shape, click near point to drag, 'i'/'p'/'d' to edit values
    - Press 'q' when finished"""
            )
        label = tk.Label(instruction_window, text=instruction_text, justify='left', padx=10, pady=10)
        label.pack()
        instruction_window.update_idletasks()
        screen_width = instruction_window.winfo_screenwidth()
        window_width = instruction_window.winfo_width()
        instruction_window.geometry(f"+{screen_width - window_width - 40}+40")

        colors = {
            'Green': (0, 255, 0),
            'Blue': (255, 0, 0),
            'Red': (0, 0, 255),
            'Yellow': (255, 255, 0),
            'Magenta': (255, 0, 255),
            'Cyan': (0, 255, 255),
            'Black': (0, 0, 0)
        }
        color_index = 0

        # Edit mode state
        edit_mode = False
        current_edit_index = -1
        edit_shape_type = None  # Will be set to 'loop', 'stopbar', or 'approach'
        dragging_point = None  # (shape_index, point_index)
        dot_radius = 5  # Matches drawing radius

        def dist(p1, p2):
            return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

        def mouse_callback_edit(event, x, y, flags, param):
            nonlocal dragging_point
            if not edit_mode:
                return
            if event == cv2.EVENT_LBUTTONDOWN:
                # Find if clicking near any vertex of currently selected shape
                if current_edit_index == -1:
                    return
                shape = self.shapes[current_edit_index]
                for i, pt in enumerate(shape['points']):
                    if dist((x, y), pt) <= dot_radius * 2:  # Buffer = 2x dot radius
                        dragging_point = (current_edit_index, i)
                        print(f"Dragging point {i} of shape {current_edit_index}")
                        break

            elif event == cv2.EVENT_LBUTTONUP:
                if dragging_point:
                    shape_idx, pt_idx = dragging_point
                    self.shapes[shape_idx]['points'][pt_idx] = (x, y)
                    print(f"Moved point {pt_idx} to ({x}, {y})")
                    dragging_point = None

            elif event == cv2.EVENT_MOUSEMOVE:
                if dragging_point:
                    # Optional: visualize drag in real-time (not necessary since we redraw every frame)
                    pass

        cv2.setMouseCallback("Draw Shapes", lambda e, x, y, f, p: self.mouse_callback(e, x, y, f, p) if not edit_mode else mouse_callback_edit(e, x, y, f, p))

        while True:
            img_copy = image.copy()

            # Draw all shapes
            for idx, shape in enumerate(self.shapes):
                is_selected = edit_mode and idx == current_edit_index
                color = shape.get('color', (0, 255, 0))
                if is_selected:
                    # Highlight selected shape: thicker lines and dots
                    if shape['type'] == 'loop':
                        pts = np.array(shape['points'], dtype=np.int32)
                        cv2.polylines(img_copy, [pts], isClosed=True, color=(255, 255, 255), thickness=4)
                        for pt in shape['points']:
                            cv2.circle(img_copy, pt, dot_radius + 2, (255, 255, 255), -1)
                    elif shape['type'] == 'stopbar':
                        pt1, pt2 = shape['points']
                        cv2.line(img_copy, pt1, pt2, color=(255, 255, 255), thickness=4)
                        cv2.circle(img_copy, pt1, dot_radius + 2, (255, 255, 255), -1)
                        cv2.circle(img_copy, pt2, dot_radius + 2, (255, 255, 255), -1)
                    elif shape['type'] == 'approach':
                        pt1, pt2 = shape['points']
                        cv2.line(img_copy, pt1, pt2, color=(255, 255, 255), thickness=4)
                        cv2.circle(img_copy, pt1, dot_radius + 2, (255, 255, 255), -1)
                        cv2.circle(img_copy, pt2, dot_radius + 2, (255, 255, 255), -1)
                        mid_x, mid_y = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
                        cv2.putText(img_copy, shape['direction'], (mid_x, mid_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    self.draw_shape(img_copy, shape)

            # Draw current shape being created
            if not edit_mode and len(self.current_shape) > 0:
                for pt in self.current_shape:
                    cv2.circle(img_copy, pt, 5, (0, 0, 0), -1)
                if len(self.current_shape) == 2:
                    cv2.line(img_copy, self.current_shape[0], self.current_shape[1], (0, 0, 0), 2)
                elif len(self.current_shape) >= 3:
                    pts = np.array(self.current_shape, dtype=np.int32)
                    cv2.polylines(img_copy, [pts], isClosed=(len(self.current_shape) == 4 and self.mode == 'loop'),
                                color=(0, 0, 0), thickness=1)

            # Display mode and values
            mode_text = f"Mode: {self.mode}"
            if edit_mode:
                mode_text += " | EDIT MODE"
            if self.mode == 'loop' and not edit_mode:
                color_name = next((name for name, rgb in colors.items() if rgb == self.color), str(self.color))
                mode_text += f" | Color: {color_name} | Input: {self.input_val}"
            elif not edit_mode:
                mode_text += f" | Phase: {self.phase}"
            cv2.putText(img_copy, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Draw Shapes", img_copy)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('l'):
                self.mode = 'loop'
                edit_shape_type = None
                edit_mode = False
                current_edit_index = -1
                dragging_point = None

            elif key == ord('s'):
                self.mode = 'stopbar'
                edit_shape_type = None
                edit_mode = False
                current_edit_index = -1
                dragging_point = None

            elif key == ord('a'):
                self.mode = 'approach'
                edit_shape_type = None
                edit_mode = False
                current_edit_index = -1
                dragging_point = None

            elif key == ord('c') and self.mode == 'loop' and not edit_mode:
                color_names = list(colors.keys())
                color_index = (color_index + 1) % len(color_names)
                color_name = color_names[color_index]
                self.color = colors[color_name]

            elif key == ord('i') and not edit_mode:
                inp = simpledialog.askinteger("Input Value", "Enter input value (1-64):", minvalue=1, maxvalue=64)
                if inp is not None:
                    self.input_val = inp

            elif key == ord('p') and not edit_mode:
                phase_input = simpledialog.askstring("Phase Value", "Enter phase value (1-16 or A-P):")
                if phase_input:
                    phase_input = phase_input.strip().upper()
                    if phase_input.isdigit():
                        phase = int(phase_input)
                        if 1 <= phase <= 16:
                            self.phase = phase
                        else:
                            messagebox.showerror("Phase", "Phase must be between 1-16")
                    elif len(phase_input) == 1 and 'A' <= phase_input <= 'P':
                        self.phase = f"OL{phase_input}"
                    else:
                        messagebox.showerror("Phase", "Phase must be between 1-16 or A-P")

            elif key == ord('e'):
                if not edit_mode:
                    # Enter edit mode: remember last mode, filter shapes
                    edit_mode = True
                    edit_shape_type = self.mode
                    # Find first shape of current type
                    for i, s in enumerate(self.shapes):
                        if s['type'] == edit_shape_type:
                            current_edit_index = i
                            break
                    else:
                        current_edit_index = -1
                    print(f"Entered edit mode for type: {edit_shape_type}. Selected shape {current_edit_index}")
                    cv2.setMouseCallback("Draw Shapes", mouse_callback_edit)
                else:
                    # Exit edit mode
                    edit_mode = False
                    current_edit_index = -1
                    edit_shape_type = None
                    dragging_point = None
                    cv2.setMouseCallback("Draw Shapes", self.mouse_callback)
                    print("Exited edit mode.")

            elif edit_mode:
                if key == ord('i') and current_edit_index != -1:
                    shape = self.shapes[current_edit_index]
                    if shape['type'] == 'loop':
                        inp = simpledialog.askinteger("Input Value", "Edit input value (1-64):",
                                                    initialvalue=shape.get('input', 1), minvalue=1, maxvalue=64)
                        if inp is not None:
                            shape['input'] = inp
                elif key == ord('p') and current_edit_index != -1:
                    shape = self.shapes[current_edit_index]
                    if shape['type'] == 'stopbar':
                        phase_input = simpledialog.askstring("Phase Value", "Edit phase value (1-16 or A-P):",
                                                            initialvalue=str(shape.get('phase', '')))
                        if phase_input:
                            phase_input = phase_input.strip().upper()
                            if phase_input.isdigit():
                                phase = int(phase_input)
                                if 1 <= phase <= 16:
                                    shape['phase'] = phase
                                else:
                                    messagebox.showerror("Phase", "Phase must be between 1-16")
                            elif len(phase_input) == 1 and 'A' <= phase_input <= 'P':
                                shape['phase'] = f"OL{phase_input}"
                            else:
                                messagebox.showerror("Phase", "Phase must be between 1-16 or A-P")
                elif key == ord('d') and current_edit_index != -1:
                    shape = self.shapes[current_edit_index]
                    if shape['type'] == 'approach':
                        direction = simpledialog.askstring("Direction", "Edit direction (N/E/S/W):",
                                                        initialvalue=shape.get('direction', 'N'))
                        if direction and direction.upper() in ['N', 'E', 'S', 'W']:
                            shape['direction'] = direction.upper()
                        else:
                            messagebox.showerror("Direction", "Must be N, E, S, or W")

                # Cycle shapes using 'n' (next) and 'p' (previous)
                elif key == ord('n'):  # Next shape
                    if not self.shapes:
                        continue
                    candidates = [i for i, s in enumerate(self.shapes) if s['type'] == edit_shape_type]
                    if not candidates:
                        continue
                    try:
                        current_idx = candidates.index(current_edit_index)
                        next_idx = (current_idx + 1) % len(candidates)
                        current_edit_index = candidates[next_idx]
                        print(f"[n] Cycled to next shape: #{current_edit_index} ({self.shapes[current_edit_index]['type']})")
                    except ValueError:
                        current_edit_index = candidates[0]

                elif key == ord('p'):  # Previous shape
                    if not self.shapes:
                        continue
                    candidates = [i for i, s in enumerate(self.shapes) if s['type'] == edit_shape_type]
                    if not candidates:
                        continue
                    try:
                        current_idx = candidates.index(current_edit_index)
                        prev_idx = (current_idx - 1) % len(candidates)
                        current_edit_index = candidates[prev_idx]
                        print(f"[p] Cycled to previous shape: #{current_edit_index} ({self.shapes[current_edit_index]['type']})")
                    except ValueError:
                        current_edit_index = candidates[0]

            elif key == ord('u'):  # Undo
                if self.current_shape:
                    self.current_shape.pop()
                    print("Undone: Removed last point.")
                elif self.shapes:
                    removed_shape = self.shapes.pop()
                    shape_type = removed_shape['type'].title()
                    print(f"Undone: Removed last {shape_type} shape.")
                else:
                    print("Nothing to undo.")

        cv2.destroyAllWindows()
        root.destroy()
        return self.shapes

    def save_shapes_to_csv(self, filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['type', 'points', 'color', 'input', 'phase', 'direction', 'video_width', 'video_height'])
            for shape in self.shapes:
                points_str = ';'.join([f"{pt[0]},{pt[1]}" for pt in shape['points']])
                color_str = f"{shape.get('color', (0,0,0))[0]},{shape.get('color', (0,0,0))[1]},{shape.get('color', (0,0,0))[2]}"
                writer.writerow([
                    shape['type'],
                    points_str,
                    color_str,
                    shape.get('input', ''),
                    shape.get('phase', ''),
                    shape.get('direction', ''),
                    self.width,
                    self.height
                ])

    def load_shapes_from_csv(self, filename):
        shapes = []
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    points = []
                    for pt_str in row['points'].split(';'):
                        x, y = map(int, pt_str.split(','))
                        points.append((x, y))
                    color = tuple(map(int, row['color'].split(','))) if row['color'] else (0, 255, 0)
                    shape = {
                        'type': row['type'],
                        'points': points,
                        'color': color,
                        'input': int(row['input']) if row['input'] else None,
                        'phase': row['phase'] if row['phase'] else None,
                        'direction': row['direction'] if row['direction'] else None
                    }
                    shapes.append(shape)
        self.shapes = shapes
        return shapes

    def load_and_process_data(self, pickle_path, start_time_str, end_time_str, time_step=0.1):
        """Load and process data from pickle file with datetime filtering"""
        df = pd.read_pickle(pickle_path)
        
        # Convert start and end times to datetime
        start_dt = pd.to_datetime(start_time_str)
        end_dt = pd.to_datetime(end_time_str)
        df_start = start_dt - pd.Timedelta(hours=1)  # 1 hour before start_dt
        df_end = end_dt + pd.Timedelta(hours=1)      # 1 hour after end_dt
        
        # Filter data to relevant time range
        df = df[(df['TS_start'] >= df_start) & (df['TS_start'] <= df_end)]
        
        if df.empty:
            print("Warning: No data found in the specified time range")
            return df
        
        # Get relevant phases and detectors from shapes
        relevant_phases = list(set([s['phase'] for s in self.shapes if s['type'] == 'stopbar' and s['phase'] is not None and "OL" not in str(s['phase'])]))
        
        # Map overlap names like "OLA", "OLB", etc. to numbers: "OLA"->1, "OLB"->2, ...
        overlap_map = {f"OL{chr(ord('A') + i)}": i + 1 for i in range(26)}
        relevant_overlaps = []
        for s in self.shapes:
            if s['type'] == 'stopbar' and s['phase'] is not None and "OL" in str(s['phase']):
                phase_str = str(s['phase'])
                mapped_val = overlap_map.get(phase_str)
                if mapped_val is not None:
                    relevant_overlaps.append(mapped_val)
        relevant_overlaps = list(set(relevant_overlaps))
        
        relevant_detectors = list(set([s['input'] for s in self.shapes if s['type'] == 'loop' and s['input'] is not None]))


        df = comb_gyr_det(df)
        df = detector_status(df, relevant_detectors)
        df = phase_status(df, relevant_phases)
        df = overlap_status(df, relevant_overlaps)
        
        # Select relevant columns
        cols_to_keep = ['TS_start']
        for ph in relevant_phases:
            col_name = f'Ph {ph} Status'
            if col_name in df.columns:
                cols_to_keep.append(col_name)
        for ol in relevant_overlaps:
            col_name = f'OL{chr(ord("A") + ol - 1)} Status' if ol <= 26 else f'OL{ol} Status'
            if col_name in df.columns:
                cols_to_keep.append(col_name)
        for det in relevant_detectors:
            col_name = f'Det {det} Status'
            if col_name in df.columns:
                cols_to_keep.append(col_name)

        
        df = df[cols_to_keep]
        df.drop_duplicates(subset=['TS_start'], inplace=True, keep='last')
        df.sort_values('TS_start', inplace=True)

        # Expand timestamps
        if not df.empty:
            expanded_times = pd.date_range(start=start_dt, end=end_dt, freq=f'{int(time_step*1000)}ms')
            expanded_df = pd.DataFrame({'TS_start': expanded_times})

            # Merge and forward-fill
            merged_df = pd.merge_asof(expanded_df, df, on='TS_start', direction='forward')
            return merged_df
        
        df.to_csv('expanded_data.csv', index=False)
        return df

    def extract_frames_at_intervals(self, start_dt, end_dt, interval=0.1):
        """Extract frames at specified datetime intervals"""
        frames = []
        timestamps = []

        # Convert datetime to seconds from video start
        video_start_time = 0.0  # Assuming video starts at 0 seconds
        
        current_dt = start_dt
        while current_dt <= end_dt:
            # Calculate time in seconds from video start
            time_diff = (current_dt - start_dt).total_seconds()
            current_time = video_start_time + time_diff
            
            frame_idx = int(current_time * self.fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if ret:
                frames.append(frame.copy())
                timestamps.append(current_dt)
            current_dt += pd.Timedelta(seconds=interval)

        return frames, timestamps

    def overlay_shapes(self, frame, row_data):
        """Overlay shapes on frame based on data"""
        for shape in self.shapes:
            if shape['type'] == 'loop' and shape['input'] is not None:
                det_col = f"Det {shape['input']} Status"
                status = row_data.get(det_col, 'na') if det_col in row_data else 'na'

                if pd.isna(status) or status == 'na':
                    outline_color = (0,0,0)
                    fill_color = None
                    alpha = 0
                elif status == 'Off':
                    outline_color = shape['color']
                    fill_color = None
                    alpha = 0
                elif status == 'On':
                    outline_color = (255, 255, 255)
                    fill_color = shape['color']
                    alpha = 0.2

                pts = np.array(shape['points'], dtype=np.int32)
                if alpha > 0:
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [pts], fill_color)
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                cv2.polylines(frame, [pts], isClosed=True, color=outline_color, thickness=1)

            elif shape['type'] == 'stopbar' and shape['phase'] is not None:
                # Determine if phase is overlap (e.g., 'OLA', 'OLB', ...) or integer
                phase_val = shape['phase']
                # Check if phase is integer-like (int or string of digits)
                if isinstance(phase_val, int) or (isinstance(phase_val, str) and phase_val.isdigit()):
                    ph_col = f"Ph {phase_val} Status"
                else:
                    ph_col = f"{phase_val} Status"
                status = row_data.get(ph_col, 'na') if ph_col in row_data else 'na'
                color_map = {'R': (0, 0, 255), 'Y': (0, 255, 255), 'G': (0, 255, 0), 'Rc': (0, 0, 255), 'na': (128, 128, 128)}
                color = color_map.get(status, (0, 0, 0))  # Default to black if status not recognized
                pt1, pt2 = shape['points']
                cv2.line(frame, pt1, pt2, color, thickness=3)

    def write_video(self, frames, output_path, fps):
        """Write frames to output video"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (self.width, self.height))
        for frame in frames:
            out.write(frame)
        out.release()

    def process_video(self, video_path, pickle_path, output_path, shape_csv, start_time_str):
        """
        Main processing function that:
        - Aligns video to real-world time using start_time_str
        - Loads and processes data from pickle (0.1s granularity)
        - Overlays phase/loop status on each frame based on nearest prior data
        - Writes synchronized output video
        """
        # Read video
        self.read_video(video_path)
        
        # Load or create shapes
        if os.path.exists(shape_csv):
            self.load_shapes_from_csv(shape_csv)
            print(f"Loaded {len(self.shapes)} shapes from {shape_csv}")
        else:
            print("No shape file found. Starting drawing interface...")
            self.draw_shapes_interface()
            self.save_shapes_to_csv(shape_csv)
            print(f"Saved {len(self.shapes)} shapes to {shape_csv}")

        # Parse start time
        try:
            start_dt = pd.to_datetime(start_time_str)
        except Exception as e:
            raise ValueError(f"Invalid datetime format: {e}")

        # --- Load and Process Data ---
        df = pd.read_pickle(pickle_path)

        # Convert start time and add buffer
        df_start = start_dt - pd.Timedelta(hours=1)
        df_end = start_dt + pd.Timedelta(hours=2)  # Adjust if video is longer

        # Filter data to expanded time window
        df = df[(df['TS_start'] >= df_start) & (df['TS_start'] <= df_end)]
        if df.empty:
            print("Warning: No data found in the extended time range (±1 hour)")
            return

        # Get relevant phases, overlaps, and detectors from shapes
        relevant_phases = list(set(
            s['phase'] for s in self.shapes
            if s['type'] == 'stopbar' and s['phase'] is not None and "OL" not in str(s['phase'])
        ))

        # Map overlap names: "OLA" -> 1, "OLB" -> 2, etc.
        overlap_map = {f"OL{chr(ord('A') + i)}": i + 1 for i in range(26)}
        relevant_overlaps = []
        for s in self.shapes:
            if s['type'] == 'stopbar' and s['phase'] is not None and "OL" in str(s['phase']):
                phase_str = str(s['phase'])
                mapped_val = overlap_map.get(phase_str)
                if mapped_val is not None:
                    relevant_overlaps.append(mapped_val)
        relevant_overlaps = list(set(relevant_overlaps))

        relevant_detectors = list(set(
            s['input'] for s in self.shapes
            if s['type'] == 'loop' and s['input'] is not None
        ))

        # Apply data processing functions (from misc_tools)
        df = comb_gyr_det(df)
        df = detector_status(df, relevant_detectors)
        df = phase_status(df, relevant_phases)
        df = overlap_status(df, relevant_overlaps)

        # Select relevant columns
        cols_to_keep = ['TS_start']
        for ph in relevant_phases:
            col_name = f'Ph {ph} Status'
            if col_name in df.columns:
                cols_to_keep.append(col_name)
        for ol in relevant_overlaps:
            col_name = f'OL{chr(ord("A") + ol - 1)} Status' if ol <= 26 else f'OL{ol} Status'
            if col_name in df.columns:
                cols_to_keep.append(col_name)
        for det in relevant_detectors:
            col_name = f'Det {det} Status'
            if col_name in df.columns:
                cols_to_keep.append(col_name)

        df = df[cols_to_keep]

        # ✅ Drop duplicates, keep last (as you wanted)
        df.drop_duplicates(subset=['TS_start'], inplace=True, keep='last')
        df.sort_values('TS_start', inplace=True)

        # --- Prepare Video Output ---
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        if not out.isOpened():
            raise RuntimeError(f"Cannot create output video: {output_path}")

        # --- Process Each Frame Sequentially ---
        frame_idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Compute real-world timestamp for this frame
            timestamp = start_dt + pd.Timedelta(seconds=frame_idx / self.fps)

            # Reset canvas
            display_frame = frame.copy()

            # Overlay shapes based on data
            if not df.empty:
                # Use pd.merge_asof to get the last known data at or before this timestamp
                # This ensures no future data is used (causal alignment)
                data_row = df[df['TS_start'] <= timestamp]
                if not data_row.empty:
                    row_data = data_row.iloc[-1].to_dict()
                    self.overlay_shapes(display_frame, row_data)

            # Add timestamp text
            ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]  # 10ms precision
            cv2.putText(display_frame, ts_str, (10, self.height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Write frame
            out.write(display_frame)
            frame_idx += 1

        # Cleanup
        self.cap.release()
        out.release()

        print(f"Output video saved to {output_path}")
        print(f"Processed {frame_idx} frames from {start_dt} at {self.fps:.2f} FPS")
    
    def count_turning_movements(self, video_path, shape_csv, output_csv, start_time_str, detection_mode="bg_sub", yolo_model="yolov8s.pt"):
        """
        Count turning movements using either background subtraction ('bg_sub') or YOLOv8 ('yolo').
        Gracefully handles interruptions to ensure debug video is saved.
        Args:
            video_path (str): Path to input video.
            shape_csv (str): Path to shapes CSV with approach lines.
            output_csv (str): Output CSV to save turning movements.
            start_time_str (str): Start timestamp for video (e.g., '2023-01-01 12:00:00').
            detection_mode (str): 'bg_sub' or 'yolo'.
            yolo_model (str): YOLOv8 model to use (e.g., 'yolov8n.pt', 'yolov8s.pt').
        """
        stop_tracking_after_count = True
        print("count_turning_movements: Method started", flush=True)
        print(f"Video path: {video_path}", flush=True)
        print(f"Detection mode: {detection_mode}", flush=True)
        cap = None
        out = None
        debug_video_path = video_path.rsplit('.', 1)[0] + f"_TMC_debug_{detection_mode}.mp4"
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Video properties: {frame_width}x{frame_height}@{fps}fps", flush=True)
            self.load_shapes_from_csv(shape_csv)
            approaches = [s for s in self.shapes if s['type'] == 'approach']
            if len(approaches) < 2:
                raise ValueError("At least two approach lines are required.")
            print(f"Loaded {len(approaches)} approach lines: {[a['direction'] for a in approaches]}", flush=True)
            direction_to_line = {app['direction']: app['points'] for app in approaches}
            dir_colors = {'N': (255, 0, 0), 'E': (0, 255, 0), 'S': (0, 0, 255), 'W': (255, 255, 0)}

            # --- Define Intersection Polygon ---
            intersection_polygon_points = None
            if len(approaches) >= 2:
                # Find the innermost points of the approach lines to define the intersection
                inner_points = []
                for app in approaches:
                    p1, p2 = app['points']
                    # Assume direction indicates which end is the "approach" end
                    # We want the point closer to the *center* of the image as the "inner" point
                    # This is a simplification. A more robust method might be needed.
                    center_x, center_y = frame_width / 2, frame_height / 2
                    dist_p1 = (p1[0] - center_x)**2 + (p1[1] - center_y)**2
                    dist_p2 = (p2[0] - center_x)**2 + (p2[1] - center_y)**2
                    if dist_p1 < dist_p2:
                        inner_points.append(p1)
                    else:
                        inner_points.append(p2)

                if len(inner_points) >= 3:
                     # Order points to form a polygon (e.g., sort by angle from centroid)
                     # Simple clockwise sort based on centroid
                     cx = sum(p[0] for p in inner_points) / len(inner_points)
                     cy = sum(p[1] for p in inner_points) / len(inner_points)
                     def clockwise_angle(point):
                         return np.arctan2(point[1] - cy, point[0] - cx)
                     intersection_polygon_points = sorted(inner_points, key=clockwise_angle)
                     print(f"Defined intersection polygon points: {intersection_polygon_points}")

            # Initialize detector
            if detection_mode == "bg_sub":
                bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=16, detectShadows=False)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            elif detection_mode == "yolo":
                from ultralytics import YOLO
                print(f"Loading YOLOv8 model: {yolo_model}", flush=True)
                yolo = YOLO(yolo_model)
                vehicle_classes = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
            else:
                raise ValueError("detection_mode must be 'bg_sub' or 'yolo'")

            # --- Use the Enhanced Tracker ---
            tracker = EnhancedIOUTracker(
                max_disappeared=15,        # Keep as is or adjust if needed
                iou_threshold=0.05,        # Lowered from 0.15
                history_length=5,          # Keep as is
                max_reacquire_frames=30,   # Keep as is or adjust if needed
                max_distance_factor=3.0    # Increased tolerance for movement based on size
            )

            turn_map = {
                ('N', 'E'): 'SBL', ('N', 'W'): 'SBR', ('N', 'S'): 'SBT',
                ('E', 'W'): 'WBT', ('E', 'N'): 'WBR', ('E', 'S'): 'WBL',
                ('S', 'N'): 'NBT', ('S', 'W'): 'NBL', ('S', 'E'): 'NBR',
                ('W', 'E'): 'EBT', ('W', 'N'): 'EBL', ('W', 'S'): 'EBR'
            }
            results = []
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(debug_video_path, fourcc, max(5, fps), (frame_width, frame_height))
            if not out.isOpened():
                raise RuntimeError(f"Cannot create debug video: {debug_video_path}")
            print(f"Debug video will be saved to: {debug_video_path}", flush=True)
            frame_idx = 0
            start_dt = pd.to_datetime(start_time_str)
            if not hasattr(self, 'entry_log'):
                self.entry_log = {}
            if not hasattr(self, 'last_centroids'):
                self.last_centroids = {}
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    if frame is None:
                        print(f"Failed to read frame {frame_idx}: returned None", flush=True)
                    else:
                        print(f"End of video at frame {frame_idx}", flush=True)
                    break
                timestamp = start_dt + pd.Timedelta(seconds=frame_idx / fps)
                debug_frame = frame.copy()
                # Draw approach lines
                for direction, (p1, p2) in direction_to_line.items():
                    color = dir_colors.get(direction, (128, 128, 128))
                    cv2.line(debug_frame, p1, p2, color, thickness=3)
                    mid_x, mid_y = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
                    cv2.putText(debug_frame, direction, (mid_x, mid_y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # === Object Detection ===
                boxes = []
                if detection_mode == "bg_sub":
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    fg_mask = bg_subtractor.apply(gray)
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
                    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    min_area = frame_width * frame_height * 0.0005
                    max_area = frame_width * frame_height * 0.05
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        x, y, w, h = cv2.boundingRect(cnt)
                        if min_area < area < max_area and 0.3 < w/h < 3:
                            boxes.append((x, y, x + w, y + h))
                elif detection_mode == "yolo":
                    results_yolo = yolo(frame, verbose=False, conf=0.5, imgsz=max(frame_height, frame_width))
                    for result in results_yolo:
                        for box in result.boxes:
                            cls_id = int(box.cls.cpu().numpy())
                            if cls_id not in vehicle_classes:
                                continue
                            b = box.xyxy.cpu().numpy()[0]
                            x1, y1, x2, y2 = map(int, b)
                            boxes.append((x1, y1, x2, y2))

                # --- Update tracker with intersection polygon ---
                # tracked_objects = tracker.update(boxes)
                tracked_objects = tracker.update(boxes, intersection_polygon=intersection_polygon_points)

                to_remove = []
                for obj_id, (x1, y1, x2, y2) in tracked_objects.items():
                    try:
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        if obj_id not in self.entry_log:
                            self.entry_log[obj_id] = {
                                'entry_dir': None,
                                'confirmed_frame': None,
                                'history': [],
                                'show_id': False,
                                'turn_logged': False
                            }
                        entry = self.entry_log[obj_id]
                        prev_cx, prev_cy = self.last_centroids.get(obj_id, (cx, cy))
                        self.last_centroids[obj_id] = (cx, cy)
                        # Check line crossings
                        for direction, (p1, p2) in direction_to_line.items():
                            if self.crosses_approach_line((prev_cx, prev_cy), (cx, cy), p1, p2, threshold=30):
                                entry['history'].append(direction)
                                if len(entry['history']) > 5:
                                    entry['history'].pop(0)
                                entry['show_id'] = True
                                if entry['entry_dir'] is None and len(entry['history']) >= 1:
                                    entry['entry_dir'] = entry['history'][-1]
                                    entry['confirmed_frame'] = frame_idx
                                if not entry.get('turn_logged', False):
                                    if (entry['entry_dir'] is not None and
                                        entry['entry_dir'] != direction and
                                        frame_idx - entry.get('confirmed_frame', -100) > 5):
                                        turn_key = (entry['entry_dir'], direction)
                                        movement = turn_map.get(turn_key)
                                        if movement:
                                            results.append({
                                                'Timestamp': timestamp,
                                                'Code': 82,
                                                'ID': movement
                                            })
                                            cv2.putText(debug_frame, f"TURN: {movement}", (cx, cy - 30),
                                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                            entry['turn_logged'] = True
                                            if stop_tracking_after_count:
                                                to_remove.append(obj_id)
                        # Draw tracking info
                        if True: #entry['show_id']:
                            color = (0, 0, 255) if entry.get('turn_logged', False) else (0, 255, 0)
                            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(debug_frame, f"ID{obj_id}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    except Exception as e:
                        print(f"Error processing object {obj_id}: {e}", flush=True)
                # Remove objects after logging
                for obj_id in to_remove:
                    if obj_id in tracker.objects: # Access internal dict
                        del tracker.objects[obj_id]
                    if obj_id in self.last_centroids:
                        del self.last_centroids[obj_id]
                # Add timestamp
                ts_text = timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]
                cv2.putText(debug_frame, ts_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # Write frame
                out.write(debug_frame)
                frame_idx += 1
        except KeyboardInterrupt:
            print("\nInterrupted by user. Shutting down gracefully...", flush=True)
        except Exception as e:
            print(f"Error during processing: {e}", flush=True)
            import traceback
            traceback.print_exc()
        finally:
            # Ensure resources are released
            if cap is not None:
                cap.release()
                print("Video capture released.", flush=True)
            if out is not None:
                out.release()
                print(f"Debug video saved to: {debug_video_path}", flush=True)
            else:
                print("Debug video was not created.", flush=True)
        # Save results
        df = pd.DataFrame(results)
        if not df.empty:
            df.to_csv(output_csv, index=False)
            print(f"Turning movement counts saved to {output_csv}", flush=True)
        else:
            print("WARNING: No turning movements detected. Check debug video and logs.", flush=True)
        return df

    
    
    @staticmethod
    def crosses_approach_line(prev_point, curr_point, line_p1, line_p2, threshold=25):
        """
        Check if movement crosses the finite approach line segment.
        Uses segment-segment intersection with tolerance.
        """
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        def segment_intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

        # Expand line perpendicularly by threshold to create a "gate"
        x1, y1 = line_p1
        x2, y2 = line_p2

        # Line vector
        dx = x2 - x1
        dy = y2 - y1
        length = (dx**2 + dy**2)**0.5
        if length == 0:
            return False

        # Unit normal vector (perpendicular)
        nx = -dy / length
        ny = dx / length

        # Expand line into a "gate" by offsetting perpendicularly
        offset = threshold
        C = (x1 + nx * offset, y1 + ny * offset)
        D = (x2 + nx * offset, y2 + ny * offset)
        E = (x1 - nx * offset, y1 - ny * offset)
        F = (x2 - nx * offset, y2 - ny * offset)

        # Create expanded gate as a parallelogram
        # Check if movement segment intersects any of the gate edges
        A, B = prev_point, curr_point

        if segment_intersect(A, B, C, D) or segment_intersect(A, B, E, F) or segment_intersect(A, B, (x1, y1), (x2, y2)):
            return True

        return False
    
    '''
    @staticmethod
    def crosses_approach_line(prev_point, curr_point, line_p1, line_p2, threshold=30):
        """
        Check if movement from prev_point to curr_point crosses the approach line segment.
        Uses finite line segment intersection with a small perpendicular tolerance.
        """
        x1, y1 = line_p1
        x2, y2 = line_p2
        px, py = prev_point
        cx, cy = curr_point

        # Vector for the approach line
        line_vec = (x2 - x1, y2 - y1)
        line_length = (line_vec[0]**2 + line_vec[1]**2)**0.5
        if line_length == 0:
            return False  # Degenerate line

        # Unit normal vector (perpendicular to line)
        nx = -line_vec[1] / line_length
        ny = line_vec[0] / line_length

        # Distance from point to infinite line
        def dist_to_line(x, y):
            return nx * (x - x1) + ny * (y - y1)

        d_prev = dist_to_line(px, py)
        d_curr = dist_to_line(cx, cy)

        # Check if the movement crosses the infinite line
        if d_prev * d_curr >= 0:
            return False  # No crossing of infinite line

        # Check if crossing is within bounds of the segment
        # Project the midpoint of the move onto the line
        mid_x = (px + cx) / 2
        mid_y = (py + cy) / 2

        # Vector from line start to midpoint
        to_mid = (mid_x - x1, mid_y - y1)

        # Dot product to get projection scalar
        proj = (to_mid[0] * line_vec[0] + to_mid[1] * line_vec[1]) / line_length

        # Check if projection is within [0, line_length] and within threshold
        if 0 <= proj <= line_length and abs(d_curr) < threshold * 15:
            return True

        return False
        '''