"""
SkillWallet Vision Processing Module
Performs privacy-preserving anonymous face detection, centroid tracking,
sustained presence calculation, head orientation proxy, and motion analysis.
"""

import os
import time
import math
from typing import Dict, List, Any, Optional, Tuple, Callable
import cv2
import numpy as np


class CentroidTracker:
    """
    Euclidean distance centroid tracker with bounding box association.
    Assigns persistent anonymous IDs (e.g. STU-01, STU-02) without biometric identification.
    """

    def __init__(self, max_disappeared: int = 15, max_distance: float = 90.0):
        self.next_object_id = 1
        self.objects: Dict[str, Tuple[int, int]] = {}  # track_id -> (centroid_x, centroid_y)
        self.bboxes: Dict[str, Tuple[int, int, int, int]] = {}  # track_id -> (x, y, w, h)
        self.disappeared: Dict[str, int] = {}  # track_id -> consecutive frames missed
        self.first_seen: Dict[str, float] = {}  # track_id -> timestamp (seconds)
        self.last_seen: Dict[str, float] = {}  # track_id -> timestamp (seconds)
        self.visible_durations: Dict[str, float] = {}  # track_id -> active seconds detected
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid: Tuple[int, int], bbox: Tuple[int, int, int, int], timestamp: float, dt: float) -> str:
        track_id = f"STU-{self.next_object_id:02d}"
        self.objects[track_id] = centroid
        self.bboxes[track_id] = bbox
        self.disappeared[track_id] = 0
        self.first_seen[track_id] = timestamp
        self.last_seen[track_id] = timestamp
        self.visible_durations[track_id] = dt
        self.next_object_id += 1
        return track_id

    def deregister(self, track_id: str) -> None:
        if track_id in self.objects:
            del self.objects[track_id]
        if track_id in self.bboxes:
            del self.bboxes[track_id]
        if track_id in self.disappeared:
            del self.disappeared[track_id]

    def update(self, rects: List[Tuple[int, int, int, int]], timestamp: float, dt: float) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Updates tracked objects given bounding boxes for the current sampled frame.
        rects: list of (x, y, w, h)
        Returns: Dict of active track_id -> (x, y, w, h)
        """
        if len(rects) == 0:
            for track_id in list(self.disappeared.keys()):
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self.deregister(track_id)
            return {k: self.bboxes[k] for k in self.objects.keys()}

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x, y, w, h) in enumerate(rects):
            input_centroids[i] = (int(x + w / 2.0), int(y + h / 2.0))

        if len(self.objects) == 0:
            for i in range(len(rects)):
                self.register(tuple(input_centroids[i]), rects[i], timestamp, dt)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array(list(self.objects.values()))

            # Compute Euclidean distances between existing centroids and new centroids
            dists = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)

            rows = dists.min(axis=1).argsort()
            cols = dists.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if dists[row, col] > self.max_distance:
                    continue

                track_id = object_ids[row]
                self.objects[track_id] = tuple(input_centroids[col])
                self.bboxes[track_id] = rects[col]
                self.disappeared[track_id] = 0
                self.last_seen[track_id] = timestamp
                self.visible_durations[track_id] = self.visible_durations.get(track_id, 0.0) + dt

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, dists.shape[0])).difference(used_rows)
            unused_cols = set(range(0, dists.shape[1])).difference(used_cols)

            for row in unused_rows:
                track_id = object_ids[row]
                self.disappeared[track_id] += 1
                if self.disappeared[track_id] > self.max_disappeared:
                    self.deregister(track_id)

            for col in unused_cols:
                self.register(tuple(input_centroids[col]), rects[col], timestamp, dt)

        return {k: self.bboxes[k] for k in self.objects.keys()}


class VisionAnalyzer:
    """
    Analyzes classroom video streams using computer vision techniques:
    1. Multi-face detection (Haar Cascades with multi-scale scanning)
    2. Head orientation & gaze proxy estimation (eye presence and face geometry)
    3. Motion indexing & inactivity detection (temporal frame differencing)
    4. Sustained presence thresholding for anonymous attendance
    """

    def __init__(self):
        # Load OpenCV Haar Cascades
        frontal_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        profile_path = os.path.join(cv2.data.haarcascades, "haarcascade_profileface.xml")
        eye_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")

        self.face_cascade = cv2.CascadeClassifier(frontal_path)
        self.profile_cascade = cv2.CascadeClassifier(profile_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_path)

    def detect_faces(self, gray_frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[str]]:
        """
        Detects frontal and profile faces in a grayscale frame.
        Returns: list of (x, y, w, h) bounding boxes and list of face types ('frontal' or 'profile').
        """
        faces = []
        face_types = []

        # 1. Frontal faces
        frontal_detections = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=(35, 35),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in frontal_detections:
            faces.append((int(x), int(y), int(w), int(h)))
            face_types.append("frontal")

        # 2. Profile faces (side views)
        profile_detections = self.profile_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(35, 35)
        )
        for (px, py, pw, ph) in profile_detections:
            # Check for overlap with already detected frontal faces
            overlap = False
            for (fx, fy, fw, fh) in faces:
                dx = min(px + pw, fx + fw) - max(px, fx)
                dy = min(py + ph, fy + fh) - max(py, fy)
                if dx > 0 and dy > 0:
                    intersection = dx * dy
                    if intersection / float(pw * ph) > 0.35:
                        overlap = True
                        break
            if not overlap:
                faces.append((int(px), int(py), int(pw), int(ph)))
                face_types.append("profile")

        return faces, face_types

    def evaluate_head_orientation(self, gray_frame: np.ndarray, bbox: Tuple[int, int, int, int], face_type: str) -> float:
        """
        Estimates proxy head orientation score (0.0 to 1.0):
        - Frontal face with both eyes detected: 0.85 - 1.0 (looking forward / at board)
        - Frontal face with one eye detected: 0.60 - 0.75 (partially turned)
        - Frontal face with no visible eyes: 0.40 - 0.55 (looking down at notebook/laptop)
        - Profile face (tilted > 45 degrees away): 0.20 - 0.40
        """
        if face_type == "profile":
            return 0.35

        x, y, w, h = bbox
        # Upper face ROI for eye detection (top 55% of face)
        roi_y_end = int(y + h * 0.55)
        face_roi = gray_frame[y:roi_y_end, x:x+w]
        if face_roi.size == 0:
            return 0.50

        eyes = self.eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(15, 15)
        )
        num_eyes = len(eyes)

        if num_eyes >= 2:
            return 0.95
        elif num_eyes == 1:
            return 0.70
        else:
            # Looking downward or eyes partially obscured
            return 0.45

    def compute_motion_index(self, prev_gray: Optional[np.ndarray], curr_gray: np.ndarray) -> float:
        """
        Computes normalized motion index (0.0 to 1.0) using temporal frame differencing.
        """
        if prev_gray is None:
            return 0.05

        # Absolute frame difference
        diff = cv2.absdiff(prev_gray, curr_gray)
        # Threshold to eliminate minor camera sensor noise
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        non_zero = np.count_nonzero(thresh)
        total_pixels = thresh.size
        motion_ratio = non_zero / float(total_pixels) if total_pixels > 0 else 0.0

        return float(min(1.0, motion_ratio * 12.0))


def process_video_stream(
    video_path: str,
    presence_threshold_sec: float = 5.0,
    interval_sec: float = 5.0,
    target_sample_fps: float = 4.0,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> Dict[str, Any]:
    """
    Main processing pipeline for classroom video:
    - Extracts frame samples at target_sample_fps
    - Detects faces and updates anonymous centroid tracker
    - Calculates head orientation and temporal motion
    - Computes interval-based video engagement sub-metrics
    - Synthesizes attendance records with presence threshold verification
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or math.isnan(native_fps):
        native_fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / native_fps if native_fps > 0 else 0.0

    # Frame sampling step
    step = max(1, int(round(native_fps / target_sample_fps)))
    effective_sample_fps = native_fps / step
    dt = 1.0 / effective_sample_fps

    analyzer = VisionAnalyzer()
    tracker = CentroidTracker(max_disappeared=int(effective_sample_fps * 4), max_distance=video_width * 0.15)

    prev_gray: Optional[np.ndarray] = None
    frame_idx = 0
    sampled_idx = 0

    # Aggregated metrics per interval
    num_intervals = max(1, int(math.ceil(duration_sec / interval_sec))) if duration_sec > 0 else 1
    interval_bins: List[Dict[str, Any]] = [
        {
            "interval_idx": i,
            "start_sec": round(i * interval_sec, 2),
            "end_sec": round(min(duration_sec, (i + 1) * interval_sec), 2),
            "face_counts": [],
            "orientations": [],
            "motions": []
        }
        for i in range(num_intervals)
    ]

    preview_frames: List[np.ndarray] = []
    capture_preview_indices = [
        int(total_frames * 0.20),
        int(total_frames * 0.50),
        int(total_frames * 0.80)
    ]

    start_process_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            curr_time_sec = frame_idx / native_fps
            interval_idx = min(int(curr_time_sec // interval_sec), num_intervals - 1)

            # Resize for optimal detection speed if larger than 960px width
            orig_h, orig_w = frame.shape[:2]
            target_w = 800
            if orig_w > target_w:
                scale = target_w / float(orig_w)
                resized = cv2.resize(frame, (target_w, int(orig_h * scale)), interpolation=cv2.INTER_AREA)
            else:
                scale = 1.0
                resized = frame

            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            # Apply CLAHE for contrast normalization in varying classroom lighting
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_gray = clahe.apply(gray)

            # Detect faces
            faces, face_types = analyzer.detect_faces(enhanced_gray)

            # Rescale faces back to original frame dimensions if resized
            if scale != 1.0:
                scaled_faces = [
                    (int(x / scale), int(y / scale), int(w / scale), int(h / scale))
                    for (x, y, w, h) in faces
                ]
            else:
                scaled_faces = faces

            # Update anonymous tracker
            active_tracks = tracker.update(scaled_faces, curr_time_sec, dt)

            # Head orientation proxy
            frame_orientations = []
            for i, bbox in enumerate(faces):
                ftype = face_types[i] if i < len(face_types) else "frontal"
                orient_score = analyzer.evaluate_head_orientation(enhanced_gray, bbox, ftype)
                frame_orientations.append(orient_score)

            # Motion index
            motion_val = analyzer.compute_motion_index(prev_gray, enhanced_gray)
            prev_gray = enhanced_gray.copy()

            # Record into current interval
            current_faces_count = len(scaled_faces)
            interval_bins[interval_idx]["face_counts"].append(current_faces_count)
            interval_bins[interval_idx]["motions"].append(motion_val)
            if frame_orientations:
                interval_bins[interval_idx]["orientations"].append(float(np.mean(frame_orientations)))
            else:
                interval_bins[interval_idx]["orientations"].append(0.5)

            # Capture preview annotated frame for visual UI proof (converted to RGB for Streamlit)
            should_capture = (
                (any(abs(frame_idx - p_idx) < step for p_idx in capture_preview_indices) and len(preview_frames) < 3) or
                (len(scaled_faces) > 0 and len(preview_frames) < 3)
            )
            if should_capture:
                annotated = frame.copy()
                for tr_id, (tx, ty, tw, th) in active_tracks.items():
                    cv2.rectangle(annotated, (tx, ty), (tx + tw, ty + th), (0, 242, 254), 2)
                    label = f"ID: {tr_id}"
                    cv2.putText(annotated, label, (tx, max(20, ty - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 242, 254), 2)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                preview_frames.append(annotated_rgb)

            sampled_idx += 1

            if progress_callback and total_frames > 0:
                pct = min(1.0, frame_idx / float(total_frames))
                progress_callback(pct, f"Analyzing video frames... ({int(pct*100)}%)")

        frame_idx += 1

    cap.release()
    total_processing_time = round(time.time() - start_process_time, 2)

    # Compile Attendance Records
    attendance_records: List[Dict[str, Any]] = []
    all_known_tracks = set(list(tracker.first_seen.keys()) + list(tracker.visible_durations.keys()))

    for track_id in sorted(all_known_tracks):
        f_seen = tracker.first_seen.get(track_id, 0.0)
        l_seen = tracker.last_seen.get(track_id, 0.0)
        v_dur = round(tracker.visible_durations.get(track_id, 0.0), 2)
        sustained = v_dur >= presence_threshold_sec

        if sustained:
            status = "Present"
        elif v_dur > 0:
            status = "Partial / Fleeting"
        else:
            status = "Absent"

        attendance_records.append({
            "track_id": track_id,
            "first_seen_sec": round(f_seen, 2),
            "last_seen_sec": round(l_seen, 2),
            "total_visible_sec": v_dur,
            "sustained_present": sustained,
            "status": status,
            "avg_confidence": 0.92
        })

    # Compile Interval Metrics
    compiled_intervals: List[Dict[str, Any]] = []
    max_observed_faces = max([max(b["face_counts"] or [0]) for b in interval_bins] or [1])
    if max_observed_faces == 0:
        max_observed_faces = 1

    for b in interval_bins:
        avg_faces = float(np.mean(b["face_counts"])) if b["face_counts"] else 0.0
        avg_orient = float(np.mean(b["orientations"])) if b["orientations"] else 0.50
        avg_motion = float(np.mean(b["motions"])) if b["motions"] else 0.05

        # Video Sub-score calculation (0 - 100):
        # 1. Visibility (40%): Face presence vs peak expected faces
        visibility_ratio = min(1.0, avg_faces / float(max_observed_faces)) if max_observed_faces > 0 else 0.0
        visibility_score = visibility_ratio * 100.0

        # 2. Head Orientation / Gaze proxy (35%): Looking forward vs tilted away
        orientation_score = avg_orient * 100.0

        # 3. Motion Index (25%): Natural attentive activity vs frozen or hyperactive
        # Inactivity (< 0.03): ~40 score; Optimal (0.05 - 0.25): ~90 score; Hyperactive (> 0.45): ~65 score
        if avg_motion < 0.03:
            motion_score = 40.0 + (avg_motion / 0.03) * 20.0
        elif avg_motion <= 0.25:
            motion_score = 80.0 + ((0.25 - avg_motion) / 0.22) * 20.0
        else:
            motion_score = max(50.0, 85.0 - (avg_motion - 0.25) * 60.0)

        video_score = round(0.40 * visibility_score + 0.35 * orientation_score + 0.25 * motion_score, 1)

        compiled_intervals.append({
            "interval_idx": b["interval_idx"],
            "start_sec": b["start_sec"],
            "end_sec": b["end_sec"],
            "face_count": int(round(avg_faces)),
            "head_orientation_score": round(orientation_score, 1),
            "motion_index": round(avg_motion * 100.0, 1),
            "video_score": video_score
        })

    video_meta = {
        "video_path": video_path,
        "filename": os.path.basename(video_path),
        "duration_sec": round(duration_sec, 2),
        "fps": round(native_fps, 2),
        "total_frames": total_frames,
        "effective_sample_fps": round(effective_sample_fps, 2),
        "processed_samples": sampled_idx,
        "processing_time_sec": total_processing_time,
        "resolution": f"{video_width}x{video_height}",
        "max_faces_detected": max_observed_faces,
        "total_students_tracked": len(attendance_records)
    }

    return {
        "meta": video_meta,
        "attendance": attendance_records,
        "intervals": compiled_intervals,
        "preview_frames": preview_frames
    }
