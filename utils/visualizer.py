import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional

"""Visualization library for exercise feedback."""

class FeedbackVisualizer:
    """Renders real-time feedback overlays for exercise analysis."""
    
    
    COLOR_CORRECT = (0, 255, 0)      # Green
    COLOR_INCORRECT = (0, 0, 255)    # Red
    COLOR_WARNING = (0, 165, 255)    # Orange
    COLOR_TEXT = (255, 255, 255)     # White
    COLOR_BG = (40, 40, 40)          # Dark Gray for panel backgrounds
    
    
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    
    def __init__(self):
        """Initializes the visualizer configuration."""
        pass

    def draw_skeleton(self, frame: np.ndarray, keypoints: Dict[str, Dict[str, float]]) -> np.ndarray:
        """Draws the pose skeleton connecting specific keypoints."""
        if not keypoints:
            return frame
            
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip')
        ]
        
        h, w = frame.shape[:2]
        
        
        for start_key, end_key in connections:
            if start_key in keypoints and end_key in keypoints:
                pt1 = keypoints[start_key]
                pt2 = keypoints[end_key]
                
                
                if pt1.get('visibility', 0) > 0.4 and pt2.get('visibility', 0) > 0.4:
                    p1_x, p1_y = int(pt1['x']), int(pt1['y'])
                    p2_x, p2_y = int(pt2['x']), int(pt2['y'])
                    
                    if pt1['x'] <= 1.0: p1_x = int(pt1['x'] * w)
                    if pt1['y'] <= 1.0: p1_y = int(pt1['y'] * h)
                    if pt2['x'] <= 1.0: p2_x = int(pt2['x'] * w)
                    if pt2['y'] <= 1.0: p2_y = int(pt2['y'] * h)

                    cv2.line(frame, (p1_x, p1_y), (p2_x, p2_y), (255, 255, 255), 2)

        
        for name, pt in keypoints.items():
            if pt.get('visibility', 0) > 0.4:
                px, py = int(pt['x']), int(pt['y'])
                if pt['x'] <= 1.0: px = int(pt['x'] * w)
                if pt['y'] <= 1.0: py = int(pt['y'] * h)
                
                cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)
                
        return frame

    def draw_complete_overlay(self, frame: np.ndarray, feedback: Dict, 
                            frame_number: int = 0, total_frames: int = 0, fps: float = 30.0) -> np.ndarray:
        """Composes all visualization components onto the frame."""
        output = frame.copy()
        h, w = frame.shape[:2]
        
        
        self._draw_status_bar(output, feedback, w)
        
        
        self._draw_angle_panel(output, feedback, h)
        
        
        self._draw_issues_panel(output, feedback, h)
        
        
        self._draw_frame_info(output, frame_number, total_frames, fps, h, w)
        
        return output

    def _draw_status_bar(self, frame: np.ndarray, feedback: Dict, width: int):
        """Renders the top status bar indicating pass/fail state."""
        is_correct = feedback.get('is_correct', False)
        score = feedback.get('score', 0)
        exercise = feedback.get('exercise', 'Unknown')
        rules_passed = feedback.get('rules_passed', 0)
        rules_total = feedback.get('rules_total', 0)
        
        color = self.COLOR_CORRECT if is_correct else self.COLOR_INCORRECT
        status_text = "CORRECT FORM" if is_correct else "INCORRECT FORM"
        
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 60), color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        
        cv2.putText(frame, f"{exercise}: {status_text}", (20, 40), 
                   self.FONT, 1.0, self.COLOR_TEXT, 2)
        
        
        score_text = f"Score: {score}/100"
        rules_text = f"Rules: {rules_passed}/{rules_total}"
        
        score_size = cv2.getTextSize(score_text, self.FONT, 0.8, 2)[0]
        rules_size = cv2.getTextSize(rules_text, self.FONT, 0.6, 1)[0]
        
        cv2.putText(frame, score_text, (width - score_size[0] - 20, 30), 
                   self.FONT, 0.8, self.COLOR_TEXT, 2)
        cv2.putText(frame, rules_text, (width - rules_size[0] - 20, 50), 
                   self.FONT, 0.6, self.COLOR_TEXT, 1)

    def _draw_angle_panel(self, frame: np.ndarray, feedback: Dict, height: int):
        """Renders the side panel showing real-time angle metrics."""
        angles = feedback.get('angles', {})
        if not angles:
            return
            
        # Panel settings
        panel_w = 280
        start_y = 80
        padding = 15
        item_height = 30
        
        # Calculate height needed
        panel_h = len(angles) * item_height + padding * 2
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, start_y), (panel_w, start_y + panel_h), self.COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        
        y = start_y + padding + 20
        for name, value in angles.items():
            display_name = name.replace('_', ' ').title()
            # Truncate if too long
            if len(display_name) > 18:
                display_name = display_name[:17] + "."
                
            text = f"{display_name}: {value:.1f}"
            cv2.putText(frame, text, (15, y), self.FONT, 0.6, self.COLOR_TEXT, 1)
            y += item_height

    def _draw_issues_panel(self, frame: np.ndarray, feedback: Dict, height: int):
        """Renders the bottom alerts panel if issues are detected."""
        issues = feedback.get('issues', [])
        if not issues:
            return
            
        panel_w = 400
        padding = 15
        line_height = 25
        
        wrapped_lines = []
        for issue in issues:
            if len(issue) > 40:
                wrapped_lines.append(f"- {issue[:37]}...")
            else:
                wrapped_lines.append(f"- {issue}")
                
        panel_h = len(wrapped_lines) * line_height + 40 # Header + padding
        start_y = height - panel_h - 60 # Above bottom frame info
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, start_y), (panel_w, height - 50), self.COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        
        cv2.putText(frame, "ISSUES DETECTED:", (15, start_y + 25), 
                   self.FONT, 0.6, (0, 0, 255), 2)
        
        
        y = start_y + 55
        for line in wrapped_lines:
            cv2.putText(frame, line, (15, y), self.FONT, 0.5, self.COLOR_TEXT, 1)
            y += line_height

    def _draw_frame_info(self, frame: np.ndarray, frame_num: int, total: int, 
                        fps: float, height: int, width: int):
        """Renders playback statistics (time, frame count)."""
        if total <= 0:
            return
            
        time_sec = frame_num / fps if fps > 0 else 0
        total_time = total / fps if fps > 0 else 0
        
        text = f"Frame: {frame_num}/{total} | Time: {time_sec:.1f}s / {total_time:.1f}s"
        
        overlay = frame.copy()
        panel_h = 40
        start_y = height - panel_h
        cv2.rectangle(overlay, (0, start_y), (width, height), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        
        text_size = cv2.getTextSize(text, self.FONT, 0.6, 1)[0]
        text_x = width - text_size[0] - 20
        cv2.putText(frame, text, (text_x, height - 12), self.FONT, 0.6, (200, 200, 200), 1)

def test_visualizer():
    """Generates synthetic frames to verify visualization layout."""
    print("="*70)
    print("TESTING FEEDBACK VISUALIZER")
    print("="*70)

    
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)  # Dark gray background

    
    viz = FeedbackVisualizer()

    
    print("\n### TEST 1: Correct Form Visualization ###")
    correct_feedback = {
        'exercise': 'Bicep Curl',
        'is_correct': True,
        'score': 95,
        'rules_passed': 4,
        'rules_total': 4,
        'angles': {
            'elbow': 85.5,
            'elbow_displacement': 45.2,
            'wrist_alignment': 172.8,
            'back_posture': 8.3
        },
        'issues': []
    }

    frame_correct = frame.copy()
    frame_correct = viz.draw_complete_overlay(
        frame_correct, correct_feedback,
        frame_number=45, total_frames=120, fps=30
    )

    
    cv2.imwrite('test_correct_form.jpg', frame_correct)
    print("Saved 'test_correct_form.jpg'")
    

    
    print("\n### TEST 2: Incorrect Form with Issues ###")
    incorrect_feedback = {
        'exercise': 'Lateral Raise',
        'is_correct': False,
        'score': 60,
        'rules_passed': 2,
        'rules_total': 4,
        'angles': {
            'left_wrist_alignment': 45.7,
            'right_wrist_alignment': 48.2,
            'symmetry': 0.78,
            'left_elbow': 158.3,
            'back_posture': 12.1
        },
        'issues': [
            "Arms not raised fully: L:46px R:48px from shoulder (max 30px)",
            "Uneven arms: 78% symmetry (need >85%)",
            "Elbow too bent: 158° (keep 165-175°)"
        ]
    }

    frame_incorrect = frame.copy()
    frame_incorrect = viz.draw_complete_overlay(
        frame_incorrect, incorrect_feedback,
        frame_number=75, total_frames=120, fps=30
    )

    cv2.imwrite('test_incorrect_form.jpg', frame_incorrect)
    print("Saved 'test_incorrect_form.jpg'")
    

    print("\n" + "="*70)
    print("Visualization tests completed")
    print("="*70)

if __name__ == "__main__":
    test_visualizer()
# Refactor check 6

# Refactor check 13

# Refactor check 20
