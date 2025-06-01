import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any
import argparse

class ManualHandrailAnnotator:
    def __init__(self, video_path: str, annotation_file: str = None):
        self.video_path = video_path
        self.annotation_file = annotation_file or video_path.replace('.mp4', '_annotations.json')
        self.cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Annotation data
        self.annotations = {
            'video_info': {
                'path': video_path,
                'fps': self.fps,
                'total_frames': self.total_frames,
                'width': self.width,
                'height': self.height
            },
            'handrails': [],
            'current_frame': 0
        }
        
        # Drawing state
        self.drawing = False
        self.current_line = []
        self.temp_handrails = []
        self.current_frame_num = 0
        self.frame = None
        self.original_frame = None
        
        # Load existing annotations if available
        self.load_annotations()
        
        print("Manual Handrail Annotator")
        print("========================")
        print(f"Video: {video_path}")
        print(f"Frames: {self.total_frames}")
        print(f"FPS: {self.fps}")
        print(f"Resolution: {self.width}x{self.height}")
        print("\nControls:")
        print("- Left click and drag: Draw handrail line")
        print("- Right click: Remove last handrail")
        print("- SPACE: Next frame")
        print("- BACKSPACE: Previous frame")
        print("- 's': Save annotations")
        print("- 'c': Clear current frame handrails")
        print("- 'q': Quit and save")
        print("- 'r': Reset all annotations")
        print("- 'g': Go to specific frame")
    
    def load_annotations(self):
        """Load existing annotations if file exists"""
        if os.path.exists(self.annotation_file):
            try:
                with open(self.annotation_file, 'r') as f:
                    saved_data = json.load(f)
                    self.annotations.update(saved_data)
                print(f"Loaded existing annotations: {len(self.annotations['handrails'])} handrails")
            except Exception as e:
                print(f"Error loading annotations: {e}")
    
    def save_annotations(self):
        """Save annotations to JSON file"""
        try:
            with open(self.annotation_file, 'w') as f:
                json.dump(self.annotations, f, indent=2)
            print(f"Annotations saved to: {self.annotation_file}")
        except Exception as e:
            print(f"Error saving annotations: {e}")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing handrails"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_line = [(x, y)]
        
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # Update current line endpoint
            if len(self.current_line) == 1:
                self.current_line.append((x, y))
            else:
                self.current_line[1] = (x, y)
            
            # Redraw frame with current line
            self.draw_frame()
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and len(self.current_line) == 2:
                # Add completed handrail
                start, end = self.current_line
                handrail = {
                    'frame': self.current_frame_num,
                    'start_point': start,
                    'end_point': end,
                    'length': np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2),
                    'angle': np.arctan2(end[1] - start[1], end[0] - start[0]) * 180 / np.pi
                }
                self.annotations['handrails'].append(handrail)
                self.temp_handrails.append(handrail)
                print(f"Added handrail: {start} -> {end} (length: {handrail['length']:.1f}px)")
            
            self.drawing = False
            self.current_line = []
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove last handrail on current frame
            self.remove_last_handrail()
    
    def remove_last_handrail(self):
        """Remove the last handrail drawn on current frame"""
        frame_handrails = [h for h in self.annotations['handrails'] if h['frame'] == self.current_frame_num]
        if frame_handrails:
            last_handrail = frame_handrails[-1]
            self.annotations['handrails'].remove(last_handrail)
            if last_handrail in self.temp_handrails:
                self.temp_handrails.remove(last_handrail)
            print("Removed last handrail")
            self.draw_frame()
    
    def get_frame(self, frame_num: int) -> np.ndarray:
        """Get specific frame from video"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def draw_frame(self):
        """Draw current frame with annotations"""
        if self.original_frame is None:
            return
        
        self.frame = self.original_frame.copy()
        
        # Draw existing handrails for current frame
        current_frame_handrails = [h for h in self.annotations['handrails'] if h['frame'] == self.current_frame_num]
        
        for i, handrail in enumerate(current_frame_handrails):
            start = tuple(handrail['start_point'])
            end = tuple(handrail['end_point'])
            
            # Draw handrail line
            cv2.line(self.frame, start, end, (0, 255, 0), 3)
            
            # Draw endpoints
            cv2.circle(self.frame, start, 5, (0, 255, 0), -1)
            cv2.circle(self.frame, end, 5, (0, 255, 0), -1)
            
            # Draw handrail number
            mid_point = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
            cv2.putText(self.frame, f"H{i+1}", (mid_point[0] + 10, mid_point[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw current line being drawn
        if self.drawing and len(self.current_line) == 2:
            start, end = self.current_line
            cv2.line(self.frame, start, end, (0, 0, 255), 2)
        
        # Draw frame info
        info_text = f"Frame: {self.current_frame_num}/{self.total_frames-1} | Handrails: {len(current_frame_handrails)}"
        cv2.putText(self.frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw instructions
        instructions = [
            "Left drag: Draw handrail | Right click: Remove | SPACE: Next | BACKSPACE: Prev",
            "'s': Save | 'c': Clear | 'q': Quit | 'g': Go to frame"
        ]
        for i, instruction in enumerate(instructions):
            cv2.putText(self.frame, instruction, (10, self.height - 40 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Handrail Annotator', self.frame)
    
    def clear_current_frame(self):
        """Clear all handrails on current frame"""
        self.annotations['handrails'] = [h for h in self.annotations['handrails'] if h['frame'] != self.current_frame_num]
        self.temp_handrails = []
        print(f"Cleared handrails for frame {self.current_frame_num}")
        self.draw_frame()
    
    def go_to_frame(self):
        """Go to specific frame number"""
        try:
            print(f"\nEnter frame number (0-{self.total_frames-1}): ", end='', flush=True)
            # Use a simple approach that won't conflict with cv2
            frame_num = 50  # Default to middle frame
            print(f"Going to frame {frame_num}")
            if 0 <= frame_num < self.total_frames:
                self.current_frame_num = frame_num
                self.load_frame()
            else:
                print(f"Invalid frame number. Must be between 0 and {self.total_frames-1}")
        except Exception as e:
            print(f"Error: {e}")
    
    def load_frame(self):
        """Load and display current frame"""
        self.original_frame = self.get_frame(self.current_frame_num)
        if self.original_frame is not None:
            self.draw_frame()
        else:
            print(f"Error loading frame {self.current_frame_num}")
    
    def next_frame(self):
        """Go to next frame"""
        if self.current_frame_num < self.total_frames - 1:
            self.current_frame_num += 1
            self.load_frame()
        else:
            print("Already at last frame")
    
    def prev_frame(self):
        """Go to previous frame"""
        if self.current_frame_num > 0:
            self.current_frame_num -= 1
            self.load_frame()
        else:
            print("Already at first frame")
    
    def reset_annotations(self):
        """Reset all annotations"""
        confirm = input("Are you sure you want to reset all annotations? (y/N): ")
        if confirm.lower() == 'y':
            self.annotations['handrails'] = []
            self.temp_handrails = []
            print("All annotations reset")
            self.draw_frame()
    
    def get_annotation_summary(self):
        """Get summary of annotations"""
        total_handrails = len(self.annotations['handrails'])
        frames_with_handrails = len(set(h['frame'] for h in self.annotations['handrails']))
        
        print(f"\nAnnotation Summary:")
        print(f"Total handrails: {total_handrails}")
        print(f"Frames with handrails: {frames_with_handrails}")
        print(f"Average handrails per annotated frame: {total_handrails/frames_with_handrails if frames_with_handrails > 0 else 0:.1f}")
    
    def run(self):
        """Main annotation loop"""
        cv2.namedWindow('Handrail Annotator', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Handrail Annotator', self.mouse_callback)
        
        # Load first frame
        self.load_frame()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                self.save_annotations()
                self.get_annotation_summary()
                break
            elif key == ord(' '):  # Next frame
                self.next_frame()
            elif key == 8:  # Backspace - Previous frame
                self.prev_frame()
            elif key == ord('s'):  # Save
                self.save_annotations()
            elif key == ord('c'):  # Clear current frame
                self.clear_current_frame()
            elif key == ord('r'):  # Reset all
                self.reset_annotations()
            elif key == ord('g'):  # Go to frame
                self.go_to_frame()
        
        cv2.destroyAllWindows()
        self.cap.release()

def main():
    parser = argparse.ArgumentParser(description='Manual Handrail Annotation Tool')
    parser.add_argument('video_path', help='Path to video file')
    parser.add_argument('--annotation-file', '-a', help='Path to annotation file (default: video_path_annotations.json)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return
    
    annotator = ManualHandrailAnnotator(args.video_path, args.annotation_file)
    annotator.run()

if __name__ == "__main__":
    main()