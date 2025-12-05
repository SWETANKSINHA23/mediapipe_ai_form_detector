import numpy as np
from collections import deque
from typing import List, Dict, Union, Optional

try:
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False