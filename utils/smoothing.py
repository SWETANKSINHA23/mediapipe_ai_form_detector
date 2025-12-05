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