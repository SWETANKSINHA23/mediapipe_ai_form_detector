import numpy as np
from collections import deque
from typing import List, Dict, Union, Optional

try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class RollingAverageSmoother:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)
        self.last_valid_value = 0.0
        
    def add_value(self, value: Optional[float]) -> float:
        
        if value is None or (isinstance(value, float) and np.isnan(value)):
            if self.buffer:
                return float(np.mean(self.buffer))
            return self.last_valid_value
            
        self.buffer.append(value)
        self.last_valid_value = float(np.mean(self.buffer))
        return self.last_valid_value
        
    def reset(self) -> None:
        self.buffer.clear()
        self.last_valid_value = 0.0
        
    def get_current_value(self) -> float:
        if not self.buffer:
            return 0.0
        return float(np.mean(self.buffer))

class SavitzkyGolaySmoother:
    def __init__(self, window_length: int = 7, polyorder: int = 2):
        self.window_length = window_length
        self.polyorder = polyorder
        
    def smooth_series(self, series: List[float]) -> List[float]:
        if not series:
            return []
            
        if not SCIPY_AVAILABLE:
            print("Warning: scipy not found, falling back to moving average")
            return moving_average(series, self.window_length)
        
        try:
            smoothed = savgol_filter(series, self.window_length, self.polyorder)
            return smoothed.tolist()
        except Exception as e:
            print(f"Error in Savitzky-Golay smoothing: {e}")
            return series

class AngleSmoothingManager:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        
        self.elbow_smoother = RollingAverageSmoother(window_size)
        self.back_smoother = RollingAverageSmoother(window_size)
        self.left_side_smoother = RollingAverageSmoother(window_size)
        self.right_side_smoother = RollingAverageSmoother(window_size)
        
    def smooth_elbow(self, angle: float) -> float:
        return self.elbow_smoother.add_value(angle)
        
    def smooth_back(self, angle: float) -> float:
        return self.back_smoother.add_value(angle)
    
    def smooth_left_elbow(self, angle: float) -> float:
        return self.left_side_smoother.add_value(angle)
        
    def smooth_right_elbow(self, angle: float) -> float:
        return self.right_side_smoother.add_value(angle)
        
    def reset(self) -> None:
        self.elbow_smoother.reset()
        self.back_smoother.reset()
        self.left_side_smoother.reset()
        self.right_side_smoother.reset()

class LandmarkSmoother:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history = {}

    def update(self, landmarks: object) -> List[Dict[str, float]]:
        if not landmarks:
            return []

        if hasattr(landmarks, 'landmark'):
            landmark_list = landmarks.landmark
        else:
            landmark_list = landmarks

        smoothed_landmarks = []
        for i, lm in enumerate(landmark_list):
            if i not in self.history:
                self.history[i] = {
                    'x': deque(maxlen=self.window_size),
                    'y': deque(maxlen=self.window_size),
                    'z': deque(maxlen=self.window_size),
                    'v': deque(maxlen=self.window_size)
                }
            
            val_x = getattr(lm, 'x', 0.0)
            val_y = getattr(lm, 'y', 0.0)
            val_z = getattr(lm, 'z', 0.0)
            val_v = getattr(lm, 'visibility', 0.0)

            self.history[i]['x'].append(val_x)
            self.history[i]['y'].append(val_y)
            self.history[i]['z'].append(val_z)
            self.history[i]['v'].append(val_v)

            smoothed_landmarks.append({
                'x': np.mean(self.history[i]['x']),
                'y': np.mean(self.history[i]['y']),
                'z': np.mean(self.history[i]['z']),
                'visibility': np.mean(self.history[i]['v']),
            })
            
        return smoothed_landmarks
    
    def reset(self):
        self.history = {}

# Module-level helper functions

def moving_average(series: List[float], window_size: int = 5) -> List[float]:
    if not series:
        return []
    
    window = np.ones(int(window_size))/float(window_size)
    smoothed = np.convolve(series, window, 'same')
    return smoothed.tolist()

def smooth_angles_series(series: List[float],
                         method: str = "savgol",
                         window_size: int = 7) -> List[float]:
    if method == "savgol":
        smoother = SavitzkyGolaySmoother(window_length=window_size)
        return smoother.smooth_series(series)
    elif method == "ma":
        return moving_average(series, window_size)
    else:
        return series

def _demo():
    import sys


    def safe_print(text):
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode('utf-8', errors='ignore').decode('utf-8'))
        sys.stdout.flush()

    safe_print("="*60)
    safe_print("SMOOTHING MODULE DEMO")
    safe_print("="*60)

    np.random.seed(42)
    base_angle = 90.0
    noise = np.random.normal(0, 5, 20)
    raw_series = [base_angle + n for n in noise]

    safe_print(f"Generated {len(raw_series)} noisy samples (Base 90°)")
    
    
    safe_print("\n1. Online Smoothing (Rolling Average, window=5)")
    ma_smoother = RollingAverageSmoother(window_size=5)
    
    print(f"{'Frame':<6} | {'Raw':<8} | {'Smoothed':<8}")
    print("-" * 30)
    
    online_results = []
    for i, raw in enumerate(raw_series):
        smoothed = ma_smoother.add_value(raw)
        online_results.append(smoothed)
        if i < 10: 
            print(f"{i:<6} | {raw:<8.1f} | {smoothed:<8.1f}")
            
            
    safe_print("\n2. Offline Smoothing (Savitzky-Golay, window=7, poly=2)")
    sg_smoothed = smooth_angles_series(raw_series, method='savgol', window_size=7)
    
    print(f"{'Frame':<6} | {'Raw':<8} | {'SG-Smooth':<8}")
    print("-" * 30)
    
    for i in range(10): 
        print(f"{i:<6} | {raw_series[i]:<8.1f} | {sg_smoothed[i]:<8.1f}")

    safe_print("\n" + "="*60)
    safe_print("Demo completed successfully")
    safe_print("="*60)

if __name__ == "__main__":
    _demo()