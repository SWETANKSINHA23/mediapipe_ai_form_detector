import cv2
import mediapipe as mp
import numpy as np
import json
import os
import sys
from pathlib import Path

"""Automated exercise extraction from long video streams."""

class ExerciseExtractor:
    """Automatically detect and extract exercise clips from MM-Fit workout videos."""

    def __init__(self, video_path, output_dir="data/processed"):
        self.video_path = video_path
        self.output_dir = output_dir
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.exercise_timestamps = {}
        self.pose_data_history = [] 

    def calculate_angle(self, p1, p2, p3):
        """Calculate angle at point2 formed by point1-point2-point3"""
        v1 = np.array([p1.x - p2.x, p1.y - p2.y])
        v2 = np.array([p3.x - p2.x, p3.y - p2.y])
        
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
            
        cosine_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def analyze_frame(self, frame):
        """Extract pose keypoints and calculate relevant measurements"""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)
        
        if not results.pose_landmarks:
            return None
            
        landmarks = results.pose_landmarks.landmark
        h, w, c = frame.shape
        
        r_shoulder = landmarks[12]
        r_elbow = landmarks[14]
        r_wrist = landmarks[16]
        
        elbow_angle = self.calculate_angle(r_shoulder, r_elbow, r_wrist)
        
        wrist_x = r_wrist.x * w
        wrist_y = r_wrist.y * h
        
        return {
            "elbow_angle": elbow_angle,
            "wrist_x": wrist_x,
            "wrist_y": wrist_y,
            "landmarks": landmarks
        }

    def detect_bicep_curl(self, pose_data_sequence):
        """Detects bicep curl pattern based on elbow angle variance."""
        angles = [d['elbow_angle'] for d in pose_data_sequence]
        wrist_xs = [d['wrist_x'] for d in pose_data_sequence]
        
        if not angles: return 0.0, 0
        
        angle_variance = np.ptp(angles)
        
        x_displacement = np.ptp(wrist_xs)
        
        if angle_variance > 80 and x_displacement < 100:
            return min(1.0, angle_variance / 120.0), pose_data_sequence[0]['timestamp']
            
        return 0.0, 0

    def detect_lateral_raise(self, pose_data_sequence):
        """Detects lateral raise pattern based on wrist vertical displacement."""
        angles = [d['elbow_angle'] for d in pose_data_sequence]
        wrist_ys = [d['wrist_y'] for d in pose_data_sequence]
        
        if not angles: return 0.0, 0
        
        y_displacement = np.ptp(wrist_ys)
        mean_angle = np.mean(angles)
        
        if y_displacement > 150 and mean_angle > 140:
            return min(1.0, y_displacement / 300.0), pose_data_sequence[0]['timestamp']
            
        return 0.0, 0

    def analyze_video(self):
        """Scan through video and detect exercises."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Error opening video: {self.video_path}")
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        sample_interval = 30
        frame_idx = 0
        
        print(f"Duration: {int(duration)}s. Frames: {total_frames}. Sampling every {sample_interval}.")
        
        buffer = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % sample_interval == 0:
                if frame_idx % (sample_interval * 10) == 0:
                    print(f"Scanning... {(frame_idx/total_frames)*100:.1f}%", end='\r')
                
                timestamp = frame_idx / fps
                data = self.analyze_frame(frame)
                
                if data:
                    data['timestamp'] = timestamp
                    data['frame_idx'] = frame_idx
                    self.pose_data_history.append(data)
            
            frame_idx += 1
            
        cap.release()
        print("\nAnalysis complete. Detecting sequences...")
        
        
        window_size = 4
        
        best_curl = {'conf': 0, 'time': None}
        best_lat = {'conf': 0, 'time': None}
        
        for i in range(len(self.pose_data_history) - window_size):
            window = self.pose_data_history[i:i+window_size]
            
            
            conf, ts = self.detect_bicep_curl(window)
            if conf > best_curl['conf']:
                best_curl = {'conf': conf, 'time': ts}
                
            
            conf, ts = self.detect_lateral_raise(window)
            if conf > best_lat['conf']:
                best_lat = {'conf': conf, 'time': ts}
                
        
        if best_curl['conf'] > 0.5:
             t_str = f"{int(best_curl['time']//60):02d}:{int(best_curl['time']%60):02d}"
             self.exercise_timestamps['bicep_curl'] = {
                 'time': t_str,
                 'timestamp_seconds': best_curl['time'],
                 'confidence': best_curl['conf'],
                 'output_file': 'bicep_curl_1.mp4'
             }
             
        if best_lat['conf'] > 0.5:
             t_str = f"{int(best_lat['time']//60):02d}:{int(best_lat['time']%60):02d}"
             self.exercise_timestamps['lateral_raise'] = {
                 'time': t_str,
                 'timestamp_seconds': best_lat['time'],
                 'confidence': best_lat['conf'],
                 'output_file': 'lateral_raise_1.mp4'
             }

    def extract_clip(self, start_time, duration, output_path):
        """Extracts a video clip at the specified timestamp."""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        start_frame = int(start_time * fps)
        end_frame = start_frame + int(duration * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        curr = start_frame
        while curr < end_frame:
            ret, frame = cap.read()
            if not ret: break
            out.write(frame)
            curr += 1
            
        out.release()
        cap.release()
        print(f"Saved clip: {output_path}")

    def save_metadata(self):
        """Save extraction metadata to JSON"""
        metadata = {
            "source_video": "w06_rgb.mp4",
            "source_dataset": "MM-Fit Dataset (Zenodo)",
            "exercises_detected": []
        }
        
        for exercise, data in self.exercise_timestamps.items():
            metadata["exercises_detected"].append({
                "exercise_name": exercise.replace('_', ' ').title(),
                "timestamp_start": data['time'],
                "timestamp_seconds": data['timestamp_seconds'],
                "confidence": float(f"{data['confidence']:.2f}"),
                "output_file": data['output_file']
            })
        
        log_path = os.path.join(self.output_dir, 'extraction_log.json')
        with open(log_path, 'w') as f:
            json.dump(metadata, indent=2, fp=f)
        
        print("Metadata saved to extraction_log.json")

    def run_extraction(self):
        """Main execution: analyze video and extract detected clips."""
        print("="*60)
        print("EXERCISE EXTRACTION SYSTEM - MM-Fit Dataset")
        print("="*60)
        print(f"Source: {self.video_path}")
        print("Analyzing video for bicep curls and lateral raises...")
        print("This may take 2-3 minutes for 1.7 GB video...\n")
        
        self.analyze_video()
        
        
        if 'bicep_curl' in self.exercise_timestamps:
            print(f"Bicep curl detected at {self.exercise_timestamps['bicep_curl']['time']}")
            self.extract_clip(
                self.exercise_timestamps['bicep_curl']['timestamp_seconds'],
                duration=4,
                output_path='data/processed/bicep_curl_1.mp4'
            )
        else:
            print("\n! No valid Bicep Curl detected.")
        
        if 'lateral_raise' in self.exercise_timestamps:
            print(f"Lateral raise detected at {self.exercise_timestamps['lateral_raise']['time']}")
            self.extract_clip(
                self.exercise_timestamps['lateral_raise']['timestamp_seconds'],
                duration=4,
                output_path='data/processed/lateral_raise_1.mp4'
            )
        else:
            print("\n! No valid Lateral Raise detected.")
        
        
        self.save_metadata()
        
        print("\n" + "="*60)
        print("EXTRACTION COMPLETE!")
        print("="*60)
        print("Output files:")
        if 'bicep_curl' in self.exercise_timestamps:
            print("  - data/processed/bicep_curl_1.mp4")
        if 'lateral_raise' in self.exercise_timestamps:
            print("  - data/processed/lateral_raise_1.mp4")
        print("  - data/processed/extraction_log.json")
        print("="*60)

def main():
    """Main entry point for exercise extraction."""
    video_path = 'data/raw/w06_rgb.mp4'
    if not os.path.exists(video_path):
        print("ERROR: w06_rgb.mp4 not found!")
        print(f"Please place the video file at: {video_path}")
        print("\nDownload from: https://zenodo.org/records/7672767")
        return

    
    extractor = ExerciseExtractor(video_path)
    extractor.run_extraction()

if __name__ == "__main__":
    main()
