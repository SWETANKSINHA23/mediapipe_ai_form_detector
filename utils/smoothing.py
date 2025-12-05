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