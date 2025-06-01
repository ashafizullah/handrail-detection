#!/usr/bin/env python3
"""
Expand annotations to multiple frames by copying from frame 0
"""

import json
import sys

def expand_annotations(annotation_file, target_frames=None):
    """Expand annotations from frame 0 to multiple frames"""
    
    if target_frames is None:
        target_frames = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 191]
    
    # Load existing annotations
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Get handrails from frame 0
    frame_0_handrails = [h for h in data['handrails'] if h['frame'] == 0]
    
    if not frame_0_handrails:
        print("No handrails found in frame 0")
        return
    
    print(f"Found {len(frame_0_handrails)} handrails in frame 0")
    print(f"Expanding to frames: {target_frames}")
    
    # Clear existing handrails and rebuild
    new_handrails = []
    
    for frame_num in target_frames:
        for handrail in frame_0_handrails:
            # Copy handrail with new frame number
            new_handrail = handrail.copy()
            new_handrail['frame'] = frame_num
            
            # Slightly adjust positions for different frames to simulate movement
            offset_x = (frame_num - 0) * 0.1  # Very slight movement
            offset_y = (frame_num - 0) * 0.05
            
            new_handrail['start_point'] = [
                int(handrail['start_point'][0] - offset_x),
                int(handrail['start_point'][1] - offset_y)
            ]
            new_handrail['end_point'] = [
                int(handrail['end_point'][0] - offset_x),
                int(handrail['end_point'][1] - offset_y)
            ]
            
            new_handrails.append(new_handrail)
    
    # Update data
    data['handrails'] = new_handrails
    
    # Save back
    output_file = annotation_file.replace('.json', '_expanded.json')
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Expanded annotations saved to: {output_file}")
    print(f"Total handrails: {len(new_handrails)}")
    print(f"Frames covered: {len(target_frames)}")
    print(f"Frame coverage: {len(target_frames)/data['video_info']['total_frames']*100:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python expand_annotations.py <annotation_file.json>")
        sys.exit(1)
    
    expand_annotations(sys.argv[1])