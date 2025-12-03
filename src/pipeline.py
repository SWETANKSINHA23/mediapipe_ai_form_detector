import cv2
import os
import sys
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pose_detection import MediaPipePoseEngine
from src.form_evaluation import FormAnalysis
from utils.visualizer import FeedbackVisualizer

class ExerciseFormPipeline:
    """End-to-end pipeline for exercise form analysis."""
    
    def __init__(self):
        """Initializes pipeline components."""
        print("="*70)
        print("EXERCISE ANALYSIS PIPELINE")
        print("="*70)
        print("\nLoading modules...")
        
        self.pose_engine = MediaPipePoseEngine(
            complexity=2,
            conf_thresh=0.3
        )
        
        self.form_evaluator = FormAnalysis(window=5)
        self.visualizer = FeedbackVisualizer()
        
        print("System ready.")
        print("="*70)


    def process_video(self, 
                     input_path: str, 
                     output_path: str, 
                     exercise_type: str,
                     draw_skeleton: bool = True,
                     draw_feedback: bool = True) -> dict:
        """Processes a single video through the analysis pipeline."""
        print(f"\n{'='*70}")
        print(f"File: {os.path.basename(input_path)}")
        print(f"{'='*70}")
        print(f"Exercise type: {exercise_type.replace('_', ' ').title()}")
        print(f"Output: {output_path}")
        
        if not os.path.exists(input_path):
            print(f"\nError: File not found: {input_path}")
            return None
        
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            print(f"\nError: Cannot open video input")
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"\nMeta: {width}x{height} @ {fps}fps")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {duration:.1f} seconds")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"\nError: Cannot open output stream")
            cap.release()
            return None
        
        stats = {
            'total_frames': total_frames,
            'frames_processed': 0,
            'frames_with_pose': 0,
            'frames_correct_form': 0,
            'frames_incorrect_form': 0,
            'total_score': 0,
            'avg_score': 0,
            'detection_rate': 0,
            'form_accuracy': 0
        }

        self.form_evaluator.reset()

        print(f"\nAnalyzing...")
        print("-" * 70)

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    print(f"\nReached end of video at frame {frame_count}")
                    break

                frame_count += 1
                stats['frames_processed'] = frame_count

                landmarks = self.pose_engine.process_frame(frame)

                if landmarks:
                    stats['frames_with_pose'] += 1
                    
                    keypoints = self.pose_engine.parse_landmarks(landmarks, frame.shape)
                    
                    feedback = self.form_evaluator.analyze(keypoints, exercise_type)
                    
                    stats['total_score'] += feedback['score']
                    
                    if feedback['is_correct']:
                        stats['frames_correct_form'] += 1
                    else:
                        stats['frames_incorrect_form'] += 1
                    
                    if draw_skeleton:
                        frame = self.pose_engine.render(frame, landmarks)
                    
                    if draw_feedback:
                        frame = self.visualizer.draw_complete_overlay(
                            frame, feedback, 
                            frame_number=frame_count,
                            total_frames=total_frames,
                            fps=fps
                        )
                else:
                    cv2.putText(frame, "No pose detected in this frame", 
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (0, 0, 255), 2)

                out.write(frame)

                if frame_count % int(fps) == 0 or frame_count == total_frames:
                    progress = (frame_count / total_frames) * 100
                    elapsed_time = frame_count / fps
                    print(f"Progress: {progress:.1f}% | {elapsed_time:.1f}s / {total_frames/fps:.1f}s | "
                          f"Pose detected: {stats['frames_with_pose']}/{frame_count}")

        finally:
            print("\nFinalizing video file...")
            cap.release()
            out.release()
        
        if stats['frames_with_pose'] > 0:
            stats['avg_score'] = stats['total_score'] / stats['frames_with_pose']
            stats['detection_rate'] = (stats['frames_with_pose'] / total_frames) * 100
            stats['form_accuracy'] = (stats['frames_correct_form'] / stats['frames_with_pose']) * 100
        
        print("-" * 70)
        print(f"\n{'='*70}")
        print("PROCESSING COMPLETE!")
        print(f"{'='*70}")
        print(f"\nStatistics:")
        print(f"  Total frames processed: {stats['frames_processed']}")
        print(f"  Frames with pose detected: {stats['frames_with_pose']} ({stats['detection_rate']:.1f}%)")
        print(f"  Frames with correct form: {stats['frames_correct_form']}")
        print(f"  Frames with incorrect form: {stats['frames_incorrect_form']}")
        print(f"  Average form score: {stats['avg_score']:.1f}/100")
        print(f"  Form accuracy: {stats['form_accuracy']:.1f}%")
        print(f"\nOutput saved: {output_path}")
        print(f"{'='*70}\n")
        
        return stats

    def close(self):
        """Terminates pipeline and releases system resources."""
        self.pose_engine.close()
        print("Resources released")
