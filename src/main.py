import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import ExerciseFormPipeline
def main():
    #Entry point for the exercise form detection system.
    print("\n" + "="*70)
    print("EXERCISE FORM CORRECTNESS DETECTION SYSTEM")
    print("="*70)
    print("\nDataset: MM-Fit (w06_rgb.mp4)")
    print("Exercises: Bicep Curl, Lateral Raise")
    print("Tech Stack: MediaPipe Pose + OpenCV + Rule-based Analysis")
    print("="*70)

    pipeline = ExerciseFormPipeline()



    videos_to_process = [
        {
            'input': 'data/raw/bicep_curl_1.mp4',
            'output': 'output_videos/bicep_curl_output.mp4',
            'exercise': 'bicep_curl'
        },
        {
            'input': 'data/raw/lateral_raise_1.mp4',
            'output': 'output_videos/lateral_raise_output.mp4',
            'exercise': 'lateral_raise'
        }
    ]
    
    if not os.path.exists('data/raw'):
        os.makedirs('data/raw', exist_ok=True)
        
    os.makedirs('output_videos', exist_ok=True)
    
    missing_inputs = []
    for video in videos_to_process:
        if not os.path.exists(video['input']):
            missing_inputs.append(video['input'])
            
    if missing_inputs:
        print("\n⚠ WARNING: Missing input video files:")
        for missing in missing_inputs:
            print(f"  - {missing}")
        print("\nPlease place 'bicep_curl_1.mp4' and 'lateral_raise_1.mp4' in 'data/raw/' folder.")
        print("Using placeholder logic if files are missing (pipeline will skip them).")

    all_stats = []

    for video_info in videos_to_process:
        if os.path.exists(video_info['input']):
            stats = pipeline.process_video(
                input_path=video_info['input'],
                output_path=video_info['output'],
                exercise_type=video_info['exercise'],
                draw_skeleton=True,
                draw_feedback=True
            )
            
            if stats:
                all_stats.append({
                    'exercise': video_info['exercise'],
                    'stats': stats
                })
        else:
            print(f"\nSkipping missing file: {video_info['input']}")

    if all_stats:
        print("\n" + "="*70)
        print("OVERALL SUMMARY")
        print("="*70)
        
        for result in all_stats:
            exercise_name = result['exercise'].replace('_', ' ').title()
            stats = result['stats']
            
            print(f"\n{exercise_name}:")
            print(f"  Detection rate: {stats['detection_rate']:.1f}%")
            print(f"  Average score: {stats['avg_score']:.1f}/100")
            print(f"  Form accuracy: {stats['form_accuracy']:.1f}%")
        
        print("\n" + "="*70)
        print("ALL VIDEOS PROCESSED SUCCESSFULLY!")
        print("="*70)
        print("\nOutput files:")
        for video_info in videos_to_process:
            if os.path.exists(video_info['output']):
                file_size = os.path.getsize(video_info['output']) / (1024 * 1024)  # MB
                print(f"  - {video_info['output']} ({file_size:.1f} MB)")
        
        print("\n" + "="*70)
        print("NEXT STEPS:")
        print("  1. Review output videos in output_videos/ folder")
        print("  2. Verify pose skeleton and feedback overlays")
        print("  3. Check form evaluation accuracy")
        print("  4. Prepare documentation and GitHub submission")
        print("="*70)
    else:
        print("\nNo videos were processed. Please check input files.")

    pipeline.close()

    print("\nPipeline execution complete")
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
# Refactor check 3
