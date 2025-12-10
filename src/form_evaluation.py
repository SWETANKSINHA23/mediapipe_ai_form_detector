import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.angle_calculator import AngleCalculator
from utils.smoothing import AngleSmoothingManager
from typing import Dict, List, Tuple
class FormAnalysis:
    """Evaluates exercise form using biomechanical parameters."""

    def __init__(self, window: int = 5):
        """Initializes the FormAnalysis engine."""
        self.smoother = AngleSmoothingManager(window_size=window)
        self.calc = AngleCalculator()
        self.history = []
        
        print(f"Form Analysis online (smooth={window})")

    def _analyze_curl(self, points: Dict[str, Dict[str, float]]) -> Dict:
        """Internal logic for Bicep Curl."""
        res = {
            'exercise': 'Bicep Curl',
            'is_correct': True,
            'issues': [],
            'angles': {},
            'score': 100,
            'rules_passed': 0,
            'rules_total': 4
        }
        
        # 1. Validation
        required = ['left_shoulder', 'left_elbow', 'left_wrist', 'right_shoulder', 'left_hip']
        for p in required:
            if p not in points:
                res['is_correct'] = False
                res['issues'].append(f"Missing: {p}")
                res['score'] = 0
                return res
            
            if not self.calc.is_point_visible(points[p], min_visibility=0.4):
                res['is_correct'] = False
                res['issues'].append(f"Not visible: {p}")
                res['score'] = 0
                return res
        
        l_sh = points['left_shoulder']
        l_elb = points['left_elbow']
        l_wri = points['left_wrist']
        
        # 2. Elbow Angle Analysis
        raw_angle = self.calc.calculate_angle(l_sh, l_elb, l_wri)
        smooth_angle = self.smoother.smooth_elbow(raw_angle)
        res['angles']['elbow'] = round(smooth_angle, 1)
        
        if 20 <= smooth_angle <= 170:
            res['rules_passed'] += 1
        else:
            res['is_correct'] = False
            msg = f"Elbow too bent: {int(smooth_angle)}°" if smooth_angle < 20 else f"Elbow too straight: {int(smooth_angle)}°"
            res['issues'].append(msg)
            res['score'] -= 30
        
        # 3. Stability Check (Elbow Sway)
        sway = self.calc.calculate_horizontal_distance(l_elb, l_sh)
        res['angles']['elbow_displacement'] = round(sway, 1)
        
        if sway <= 150:
            res['rules_passed'] += 1
        else:
            res['is_correct'] = False
            res['issues'].append(f"Elbow swinging: {int(sway)}px")
            res['score'] -= 25
        
        # 4. Wrist Alignment
        ref_pt = {'x': l_elb['x'], 'y': l_elb['y'] - 50, 'z': l_elb['z'], 'visibility': 1.0}
        wrist_ang = self.calc.calculate_angle(ref_pt, l_elb, l_wri)
        res['angles']['wrist_alignment'] = round(wrist_ang, 1)
        
        if wrist_ang > 150:
            res['rules_passed'] += 1
        else:
            res['is_correct'] = False
            res['issues'].append(f"Wrist bent: {int(wrist_ang)}°")
            res['score'] -= 20
        
        # 5. Posture Check
        back_ang = self.calc.calculate_back_angle(l_sh, points['left_hip'])
        smooth_back = self.smoother.smooth_back(back_ang)
        res['angles']['back_posture'] = round(smooth_back, 1)
        
        lying_down = smooth_back > 60
        if lying_down:
            res['rules_passed'] += 1 # Auto-pass vertical check if lying
        else:
            if smooth_back <= 25:
                res['rules_passed'] += 1
            else:
                res['is_correct'] = False
                res['issues'].append(f"Leaning: {int(smooth_back)}°")
                res['score'] -= 15
        
        res['score'] = max(0, res['score'])
        return res

    def _analyze_raise(self, points: Dict[str, Dict[str, float]]) -> Dict:
        """Internal logic for Lateral Raise."""
        res = {
            'exercise': 'Lateral Raise',
            'is_correct': True,
            'issues': [],
            'angles': {},
            'score': 100,
            'rules_passed': 0,
            'rules_total': 4
        }
        
        # Validation
        targets = ['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 
                  'left_wrist', 'right_wrist', 'left_hip', 'nose']
                  
        for p in targets:
            if p not in points:
                res['is_correct'] = False
                res['issues'].append(f"Missing: {p}")
                res['score'] = 0
                return res
            if not self.calc.is_point_visible(points[p], min_visibility=0.4):
                res['is_correct'] = False
                res['issues'].append(f"Low visibility: {p}")
                res['score'] = 0
                return res
        
        # 1. Height Alignment
        l_ok, l_diff = self.calc.calculate_vertical_alignment(points['left_wrist'], points['left_shoulder'], 60)
        r_ok, r_diff = self.calc.calculate_vertical_alignment(points['right_wrist'], points['right_shoulder'], 60)
        
        res['angles']['left_wrist_alignment'] = round(l_diff, 1)
        res['angles']['right_wrist_alignment'] = round(r_diff, 1)
        
        if l_ok or r_ok:
            res['rules_passed'] += 1
        else:
            res['is_correct'] = False
            res['issues'].append(f"Lift higher (L:{int(l_diff)}px R:{int(r_diff)}px)")
            res['score'] -= 35
        
        # 2. Symmetry
        sym_val, _ = self.calc.calculate_symmetry(points['left_wrist'], points['right_wrist'], points['nose'])
        res['angles']['symmetry'] = round(sym_val, 3)
        
        if sym_val >= 0.75:
            res['rules_passed'] += 1
        else:
            res['is_correct'] = False
            res['issues'].append(f"Uneven lift ({int(sym_val*100)}% symmetric)")
            res['score'] -= 25
        
        # 3. Elbow Bend
        l_ang = self.calc.calculate_angle(points['left_shoulder'], points['left_elbow'], points['left_wrist'])
        l_smooth = self.smoother.smooth_left_elbow(l_ang)
        res['angles']['left_elbow'] = round(l_smooth, 1)
        
        if 155 <= l_smooth <= 180:
            res['rules_passed'] += 1
        else:
            res['is_correct'] = False
            msg = f"Elbow too bent: {int(l_smooth)}°" if l_smooth < 155 else f"Elbow locked: {int(l_smooth)}°"
            res['issues'].append(msg)
            res['score'] -= 20
        
        # 4. Torso Stability
        back_val = self.calc.calculate_back_angle(points['left_shoulder'], points['left_hip'])
        back_smooth = self.smoother.smooth_back(back_val)
        res['angles']['back_posture'] = round(back_smooth, 1)
        
        if back_smooth > 60: # Lying down
            res['rules_passed'] += 1
        else:
            if back_smooth <= 25:
                res['rules_passed'] += 1
            else:
                res['is_correct'] = False
                res['issues'].append(f"Leaning torso: {int(back_smooth)}°")
                res['score'] -= 15
        
        res['score'] = max(0, res['score'])
        return res

    def analyze(self, keypoints: Dict[str, Dict[str, float]], 
               mode: str) -> Dict:
        """Route to specific exercise logic."""
        if mode == 'bicep_curl':
            return self._analyze_curl(keypoints)
        elif mode == 'lateral_raise':
            return self._analyze_raise(keypoints)
        else:
            return {
                'exercise': 'Unknown',
                'is_correct': False,
                'issues': [f'Unknown mode: {mode}'],
                'angles': {},
                'score': 0,
                'rules_passed': 0,
                'rules_total': 0
            }

    def reset(self):
        """Reset state."""
        self.smoother.reset()
        self.history.clear()

# Refactor check 2

# Refactor check 9

# Refactor check 16

# Refactor check 23
