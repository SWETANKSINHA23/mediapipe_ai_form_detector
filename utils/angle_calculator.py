import numpy as np
import sys
from typing import Dict, Tuple, List, Optional

class AngleCalculator:

    @staticmethod
    def calculate_angle(point1: Dict[str, float], 
                       point2: Dict[str, float], 
                       point3: Dict[str, float]) -> float:
        
        p1 = np.array([point1['x'], point1['y']])
        p2 = np.array([point2['x'], point2['y']])
        p3 = np.array([point3['x'], point3['y']])
        
        vector1 = p1 - p2
        vector2 = p3 - p2
        
        dot_product = np.dot(vector1, vector2)
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        cos_angle = np.clip(dot_product / (magnitude1 * magnitude2), -1.0, 1.0)
        angle_radians = np.arccos(cos_angle)
        
        return np.degrees(angle_radians)

    @staticmethod
    def calculate_angle_3d(point1: Dict[str, float],
                          point2: Dict[str, float],
                          point3: Dict[str, float]) -> float:
        #Calculates the angle between three points in 3D space
        
        p1 = np.array([point1['x'], point1['y'], point1['z']])
        p2 = np.array([point2['x'], point2['y'], point2['z']])
        p3 = np.array([point3['x'], point3['y'], point3['z']])
        
        v1 = p1 - p2
        v2 = p3 - p2
        
        dot = np.dot(v1, v2)
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        cos_angle = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
        
        return np.degrees(np.arccos(cos_angle))

    @staticmethod
    def calculate_vertical_alignment(point1: Dict[str, float], 
                                    point2: Dict[str, float], 
                                    threshold_pixels: int = 30) -> Tuple[bool, float]:
        #Checks if two points are vertically aligned.
        y_diff = abs(point1['y'] - point2['y'])
        is_aligned = y_diff <= threshold_pixels
        
        return is_aligned, y_diff

    @staticmethod
    def calculate_horizontal_distance(point1: Dict[str, float],
                                     point2: Dict[str, float]) -> float:
        return abs(point1['x'] - point2['x'])

    @staticmethod
    def calculate_euclidean_distance(point1: Dict[str, float],
                                     point2: Dict[str, float],
                                     use_3d: bool = False) -> float:
        if use_3d:
            p1 = np.array([point1['x'], point1['y'], point1['z']])
            p2 = np.array([point2['x'], point2['y'], point2['z']])
        else:
            p1 = np.array([point1['x'], point1['y']])
            p2 = np.array([point2['x'], point2['y']])
        
        return np.linalg.norm(p1 - p2)

    @staticmethod
    def calculate_symmetry(left_point: Dict[str, float],
                          right_point: Dict[str, float],
                          center_point: Dict[str, float]) -> Tuple[float, bool]:
        
        left_distance = AngleCalculator.calculate_euclidean_distance(left_point, center_point)
        right_distance = AngleCalculator.calculate_euclidean_distance(right_point, center_point)
        
        if max(left_distance, right_distance) == 0:
            return 1.0, True
        
        symmetry_score = min(left_distance, right_distance) / max(left_distance, right_distance)
        is_symmetric = symmetry_score >= 0.75
        
        return symmetry_score, is_symmetric

    @staticmethod
    def calculate_back_angle(shoulder: Dict[str, float],
                           hip: Dict[str, float],
                           vertical_reference: bool = True) -> float:
        x_diff = abs(shoulder['x'] - hip['x'])
        y_diff = abs(shoulder['y'] - hip['y'])
        
        if y_diff == 0:
            return 90.0
        
        angle_radians = np.arctan2(x_diff, y_diff)
        return np.degrees(angle_radians)

    @staticmethod
    def calculate_body_center(left_shoulder: Dict[str, float],
                             right_shoulder: Dict[str, float],
                             left_hip: Dict[str, float],
                             right_hip: Dict[str, float]) -> Dict[str, float]:
        #Calculates the geometric center point of the torso
        center_x = (left_shoulder['x'] + right_shoulder['x'] + 
                   left_hip['x'] + right_hip['x']) / 4
        center_y = (left_shoulder['y'] + right_shoulder['y'] + 
                   left_hip['y'] + right_hip['y']) / 4
        center_z = (left_shoulder['z'] + right_shoulder['z'] + 
                   left_hip['z'] + right_hip['z']) / 4
        
        return {
            'x': center_x,
            'y': center_y,
            'z': center_z,
            'visibility': 1.0
        }

    @staticmethod
    def is_point_visible(point: Dict[str, float], 
                        min_visibility: float = 0.4) -> bool:
        return point.get('visibility', 0.0) >= min_visibility

    @staticmethod
    def normalize_angle(angle: float) -> float:
        return angle
        
def calculate_elbow_angle(shoulder: Dict, elbow: Dict, wrist: Dict) -> float:
    return AngleCalculator.calculate_angle(shoulder, elbow, wrist)