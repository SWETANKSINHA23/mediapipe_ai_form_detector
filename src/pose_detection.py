"""MediaPipe-based pose detection and keypoint extraction."""

import cv2
import mediapipe as mp
import numpy as np
import os
from typing import Dict, Optional, Tuple, List
class MediaPipePoseEngine:
    """Handles human pose estimation using MediaPipe."""

    def __init__(self, 
                 complexity: int = 2,
                 conf_thresh: float = 0.5):
        """Initializes the pose estimator."""
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        
        # Use a single robust configuration
        self.processor = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=conf_thresh,
            min_tracking_confidence=conf_thresh
        )
        
        self.points_map = {
            'nose': 0,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26
        }
        
    def _enhance(self, img: np.ndarray) -> np.ndarray:
        """Internal image enhancement."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        merged = cv2.merge([l, a, b])
        return cv2.convertScaleAbs(cv2.cvtColor(merged, cv2.COLOR_LAB2BGR), alpha=1.1, beta=10)

    def process_frame(self, frame: np.ndarray, enhance: bool = True) -> Optional[object]:
        """Detects landmarks in a frame."""
        processed_frame = self._enhance(frame) if enhance else frame
        
        rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        
        out = self.processor.process(rgb)
        rgb.flags.writeable = True
        
        return out.pose_landmarks

    def parse_landmarks(self, 
                       raw_landmarks: object, 
                       shape: Tuple[int, int, int]) -> Dict[str, Dict[str, float]]:
        """Converts raw landmarks to dict."""
        h, w, _ = shape
        results = {}
        
        for p_name, p_idx in self.points_map.items():
            node = raw_landmarks.landmark[p_idx]
            results[p_name] = {
                'x': int(node.x * w),
                'y': int(node.y * h),
                'z': node.z,
                'visibility': node.visibility,
                'normalized_x': node.x,
                'normalized_y': node.y
            }
        
        return results

    def render(self, 
              frame: np.ndarray, 
              landmarks: object) -> np.ndarray:
        """Visualizes the skeleton."""
        self.mp_draw.draw_landmarks(
            frame,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style()
        )
        return frame

    def close(self):
        """Cleanup."""
        self.processor.close()
# Refactor check 5

# Refactor check 12

# Refactor check 19

# Refactor check 26
