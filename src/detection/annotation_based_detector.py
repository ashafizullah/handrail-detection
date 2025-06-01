import json
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Any

class AnnotationBasedHandrailDetector:
    def __init__(self, annotation_file: str):
        self.annotation_file = annotation_file
        self.annotations = {}
        self.handrail_cache = {}  # Cache interpolated handrails
        self.load_annotations()
    
    def load_annotations(self):
        """Load manual annotations from JSON file"""
        try:
            with open(self.annotation_file, 'r') as f:
                self.annotations = json.load(f)
            print(f"Loaded annotations with {len(self.annotations.get('handrails', []))} handrails")
            
            # Group handrails by frame for faster lookup
            self.handrails_by_frame = {}
            for handrail in self.annotations.get('handrails', []):
                frame_num = handrail['frame']
                if frame_num not in self.handrails_by_frame:
                    self.handrails_by_frame[frame_num] = []
                self.handrails_by_frame[frame_num].append(handrail)
            
            print(f"Handrails found in {len(self.handrails_by_frame)} frames")
            
        except Exception as e:
            print(f"Error loading annotations: {e}")
            self.annotations = {'handrails': []}
            self.handrails_by_frame = {}
    
    def get_handrails_for_frame(self, frame_number: int) -> List[Tuple[int, int, int, int]]:
        """Get handrail lines for specific frame with interpolation"""
        
        # Check cache first
        if frame_number in self.handrail_cache:
            return self.handrail_cache[frame_number]
        
        # Direct annotation exists
        if frame_number in self.handrails_by_frame:
            handrails = []
            for handrail in self.handrails_by_frame[frame_number]:
                start = handrail['start_point']
                end = handrail['end_point']
                handrails.append((start[0], start[1], end[0], end[1]))
            
            self.handrail_cache[frame_number] = handrails
            return handrails
        
        # Try interpolation between nearest annotated frames
        interpolated = self.interpolate_handrails(frame_number)
        self.handrail_cache[frame_number] = interpolated
        return interpolated
    
    def interpolate_handrails(self, frame_number: int) -> List[Tuple[int, int, int, int]]:
        """Interpolate handrails between annotated frames"""
        annotated_frames = sorted(self.handrails_by_frame.keys())
        
        if not annotated_frames:
            return []
        
        # Find nearest annotated frames
        prev_frame = None
        next_frame = None
        
        for annotated_frame in annotated_frames:
            if annotated_frame <= frame_number:
                prev_frame = annotated_frame
            if annotated_frame >= frame_number and next_frame is None:
                next_frame = annotated_frame
                break
        
        # If we're before first annotation or after last annotation
        if prev_frame is None:
            return self.get_handrails_for_annotated_frame(annotated_frames[0])
        if next_frame is None:
            return self.get_handrails_for_annotated_frame(annotated_frames[-1])
        
        # If we're exactly on an annotated frame
        if prev_frame == frame_number:
            return self.get_handrails_for_annotated_frame(frame_number)
        
        # Interpolate between prev_frame and next_frame
        if prev_frame == next_frame:
            return self.get_handrails_for_annotated_frame(prev_frame)
        
        return self.interpolate_between_frames(prev_frame, next_frame, frame_number)
    
    def get_handrails_for_annotated_frame(self, frame_number: int) -> List[Tuple[int, int, int, int]]:
        """Get handrails for a frame that has direct annotations"""
        handrails = []
        for handrail in self.handrails_by_frame.get(frame_number, []):
            start = handrail['start_point']
            end = handrail['end_point']
            handrails.append((start[0], start[1], end[0], end[1]))
        return handrails
    
    def interpolate_between_frames(self, frame1: int, frame2: int, target_frame: int) -> List[Tuple[int, int, int, int]]:
        """Interpolate handrails between two annotated frames"""
        handrails1 = self.handrails_by_frame.get(frame1, [])
        handrails2 = self.handrails_by_frame.get(frame2, [])
        
        # Simple approach: use handrails from nearest frame
        # More sophisticated approach would try to match handrails between frames
        
        if abs(target_frame - frame1) <= abs(target_frame - frame2):
            return self.get_handrails_for_annotated_frame(frame1)
        else:
            return self.get_handrails_for_annotated_frame(frame2)
    
    def match_handrails_between_frames(self, handrails1: List[Dict], handrails2: List[Dict]) -> List[Tuple[Dict, Dict]]:
        """Match handrails between two frames based on position similarity"""
        matches = []
        used_indices2 = set()
        
        for h1 in handrails1:
            best_match = None
            best_distance = float('inf')
            best_idx = -1
            
            for i, h2 in enumerate(handrails2):
                if i in used_indices2:
                    continue
                
                # Calculate distance between handrail centers
                center1 = (
                    (h1['start_point'][0] + h1['end_point'][0]) / 2,
                    (h1['start_point'][1] + h1['end_point'][1]) / 2
                )
                center2 = (
                    (h2['start_point'][0] + h2['end_point'][0]) / 2,
                    (h2['start_point'][1] + h2['end_point'][1]) / 2
                )
                
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = h2
                    best_idx = i
            
            if best_match and best_distance < 100:  # Threshold for matching
                matches.append((h1, best_match))
                used_indices2.add(best_idx)
        
        return matches
    
    def advanced_interpolate_between_frames(self, frame1: int, frame2: int, target_frame: int) -> List[Tuple[int, int, int, int]]:
        """Advanced interpolation with handrail matching and position interpolation"""
        handrails1 = self.handrails_by_frame.get(frame1, [])
        handrails2 = self.handrails_by_frame.get(frame2, [])
        
        if not handrails1 and not handrails2:
            return []
        
        if not handrails1:
            return self.get_handrails_for_annotated_frame(frame2)
        
        if not handrails2:
            return self.get_handrails_for_annotated_frame(frame1)
        
        # Match handrails between frames
        matches = self.match_handrails_between_frames(handrails1, handrails2)
        
        # Interpolate matched handrails
        interpolated = []
        alpha = (target_frame - frame1) / (frame2 - frame1)  # Interpolation factor
        
        for h1, h2 in matches:
            # Interpolate start and end points
            start_x = int(h1['start_point'][0] * (1 - alpha) + h2['start_point'][0] * alpha)
            start_y = int(h1['start_point'][1] * (1 - alpha) + h2['start_point'][1] * alpha)
            end_x = int(h1['end_point'][0] * (1 - alpha) + h2['end_point'][0] * alpha)
            end_y = int(h1['end_point'][1] * (1 - alpha) + h2['end_point'][1] * alpha)
            
            interpolated.append((start_x, start_y, end_x, end_y))
        
        # Add unmatched handrails from closer frame
        if abs(target_frame - frame1) <= abs(target_frame - frame2):
            matched_h1 = {h1 for h1, h2 in matches}
            for h1 in handrails1:
                if h1 not in matched_h1:
                    start = h1['start_point']
                    end = h1['end_point']
                    interpolated.append((start[0], start[1], end[0], end[1]))
        else:
            matched_h2 = {h2 for h1, h2 in matches}
            for h2 in handrails2:
                if h2 not in matched_h2:
                    start = h2['start_point']
                    end = h2['end_point']
                    interpolated.append((start[0], start[1], end[0], end[1]))
        
        return interpolated
    
    def detect_handrail_edges(self, frame: np.ndarray, frame_number: int) -> List[Tuple[int, int, int, int]]:
        """Main detection method that uses manual annotations"""
        return self.get_handrails_for_frame(frame_number)
    
    def draw_handrails(self, frame: np.ndarray, lines: List[Tuple[int, int, int, int]], 
                      show_annotation_info: bool = True) -> np.ndarray:
        """Draw handrails with annotation information"""
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line
            
            # Draw handrail line (thicker for manual annotations)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            
            # Draw endpoints
            cv2.circle(frame, (x1, y1), 6, (0, 255, 0), -1)
            cv2.circle(frame, (x2, y2), 6, (0, 255, 0), -1)
            
            if show_annotation_info:
                # Draw handrail ID
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.putText(frame, f"A{i+1}", (mid_x + 10, mid_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return frame
    
    def get_annotation_stats(self) -> Dict[str, Any]:
        """Get statistics about annotations"""
        total_handrails = len(self.annotations.get('handrails', []))
        annotated_frames = len(self.handrails_by_frame)
        
        if total_handrails == 0:
            return {
                'total_handrails': 0,
                'annotated_frames': 0,
                'avg_handrails_per_frame': 0,
                'frame_coverage': 0
            }
        
        total_frames = self.annotations.get('video_info', {}).get('total_frames', 1)
        
        return {
            'total_handrails': total_handrails,
            'annotated_frames': annotated_frames,
            'avg_handrails_per_frame': total_handrails / annotated_frames,
            'frame_coverage': annotated_frames / total_frames * 100,
            'handrails_by_frame': {f: len(h) for f, h in self.handrails_by_frame.items()}
        }