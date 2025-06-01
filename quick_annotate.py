#!/usr/bin/env python3
"""
Quick annotation workflow:
1. Manually annotate handrails on frame 0
2. Apply to all frames automatically
3. Run analysis with fixed annotations
"""

import os
import sys
import subprocess
import argparse

def quick_annotate_workflow(video_path, output_video=None, threshold=30):
    """Complete workflow for quick handrail annotation"""
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return 1
    
    # Generate file paths
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.dirname(video_path)
    
    annotation_file = os.path.join(video_dir, f"{video_name}_annotations.json")
    fixed_annotation_file = os.path.join(video_dir, f"{video_name}_annotations_fixed_all_frames.json")
    
    if not output_video:
        output_video = f"data/output/{video_name}_annotated_analysis.mp4"
    
    print("="*60)
    print("QUICK HANDRAIL ANNOTATION WORKFLOW")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Annotations: {annotation_file}")
    print(f"Output: {output_video}")
    print()
    
    # Step 1: Check if annotations exist
    if os.path.exists(annotation_file):
        print(f"✅ Found existing annotations: {annotation_file}")
        
        # Ask if user wants to re-annotate
        response = input("Annotations already exist. Re-annotate? (y/N): ").lower()
        if response == 'y':
            print("🎯 Opening annotation tool...")
            try:
                # Use bash to activate environment and run annotation tool
                cmd = f"source handrail-env-mediapipe/bin/activate && python annotate_video.py {video_path}"
                subprocess.run(['bash', '-c', cmd], check=True)
            except subprocess.CalledProcessError:
                print("❌ Annotation tool failed")
                return 1
    else:
        print("🎯 Opening annotation tool for first-time annotation...")
        print("Instructions:")
        print("- Draw handrails with left click + drag")
        print("- Press 'q' when done")
        print("- Only annotate frame 0, we'll apply to all frames")
        print()
        
        try:
            # Use bash to activate environment and run annotation tool
            cmd = f"source handrail-env-mediapipe/bin/activate && python annotate_video.py {video_path}"
            subprocess.run(['bash', '-c', cmd], check=True)
        except subprocess.CalledProcessError:
            print("❌ Annotation tool failed")
            return 1
        
        if not os.path.exists(annotation_file):
            print("❌ No annotations created. Exiting.")
            return 1
    
    # Step 2: Apply annotations to all frames
    print("\n🔧 Applying annotations to all frames...")
    try:
        subprocess.run(['python3', 'apply_fixed_annotations.py', annotation_file], check=True)
    except subprocess.CalledProcessError:
        print("❌ Failed to apply fixed annotations")
        return 1
    
    if not os.path.exists(fixed_annotation_file):
        print("❌ Fixed annotation file not created")
        return 1
    
    # Step 3: Run analysis with fixed annotations
    print("\n🚀 Running handrail detection analysis...")
    try:
        # Use bash to activate environment and run analysis
        cmd = f"source handrail-env-mediapipe/bin/activate && python src/main.py {video_path} --annotations {fixed_annotation_file} --output {output_video} --threshold {threshold} --no-display"
        subprocess.run(['bash', '-c', cmd], check=True)
    except subprocess.CalledProcessError:
        print("❌ Analysis failed")
        return 1
    
    print("\n" + "="*60)
    print("✅ QUICK ANNOTATION WORKFLOW COMPLETED!")
    print("="*60)
    print(f"📹 Annotated video: {output_video}")
    print(f"📝 Annotations: {fixed_annotation_file}")
    print(f"🎯 Ready for production use!")
    
    return 0

def main():
    parser = argparse.ArgumentParser(description='Quick Handrail Annotation Workflow')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--output', '-o', help='Output video path')
    parser.add_argument('--threshold', '-t', type=int, default=30, 
                       help='Touch detection threshold (default: 30)')
    
    args = parser.parse_args()
    
    return quick_annotate_workflow(args.video_path, args.output, args.threshold)

if __name__ == "__main__":
    sys.exit(main())