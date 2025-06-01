#!/usr/bin/env python3
"""
Apply fixed annotations from frame 0 to ALL frames in the video
This creates a ground truth annotation file with consistent handrail positions
"""

import json
import sys

def apply_fixed_annotations(annotation_file):
    """Apply annotations from frame 0 to ALL frames in the video"""
    
    # Load existing annotations
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Get handrails from frame 0
    frame_0_handrails = [h for h in data['handrails'] if h['frame'] == 0]
    
    if not frame_0_handrails:
        print("No handrails found in frame 0")
        return
    
    total_frames = data['video_info']['total_frames']
    
    print(f"Found {len(frame_0_handrails)} handrails in frame 0")
    print(f"Applying to ALL {total_frames} frames")
    
    # Create handrails for ALL frames
    new_handrails = []
    
    for frame_num in range(total_frames):
        for handrail in frame_0_handrails:
            # Copy handrail with new frame number (keep exact same positions)
            new_handrail = handrail.copy()
            new_handrail['frame'] = frame_num
            
            # Keep EXACTLY the same positions (no movement/adjustment)
            new_handrail['start_point'] = handrail['start_point'].copy()
            new_handrail['end_point'] = handrail['end_point'].copy()
            
            new_handrails.append(new_handrail)
    
    # Update data
    data['handrails'] = new_handrails
    
    # Save to new file
    output_file = annotation_file.replace('.json', '_fixed_all_frames.json')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Fixed annotations saved to: {output_file}")
    print(f"Total handrails: {len(new_handrails)}")
    print(f"Handrails per frame: {len(frame_0_handrails)}")
    print(f"Frames covered: {total_frames}")
    print(f"Frame coverage: 100.0%")
    
    return output_file

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python apply_fixed_annotations.py <annotation_file.json>")
        print("Example: python apply_fixed_annotations.py data/videos/office_annotations.json")
        sys.exit(1)
    
    output_file = apply_fixed_annotations(sys.argv[1])
    print(f"\nNow you can use: --annotations {output_file}")