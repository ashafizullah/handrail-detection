import cv2
import numpy as np
from typing import List, Tuple, Optional

class ProximityAnalyzer:
    def __init__(self, touch_threshold: int = 30):
        self.touch_threshold = touch_threshold
    
    def calculate_distance_to_line(self, point: Tuple[int, int], line: Tuple[int, int, int, int]) -> float:
        """Calculate perpendicular distance from point to line"""
        x0, y0 = point
        x1, y1, x2, y2 = line
        
        # For vertical lines, also consider horizontal distance
        line_angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        if 70 <= line_angle <= 110:  # Vertical line
            # For vertical handrails, horizontal distance is more important
            line_center_x = (x1 + x2) / 2
            horizontal_distance = abs(x0 - line_center_x)
            
            # Check if point is within vertical range of the line
            min_y, max_y = min(y1, y2), max(y1, y2)
            if min_y <= y0 <= max_y:
                return horizontal_distance
            else:
                # Point is outside vertical range, calculate distance to nearest endpoint
                dist_to_start = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
                dist_to_end = np.sqrt((x0 - x2)**2 + (y0 - y2)**2)
                return min(dist_to_start, dist_to_end)
        else:
            # Original calculation for non-vertical lines
            # Line equation: ax + by + c = 0
            a = y2 - y1
            b = x1 - x2
            c = (x2 - x1) * y1 - (y2 - y1) * x1
            
            # Distance = |ax0 + by0 + c| / sqrt(a^2 + b^2)
            distance = abs(a * x0 + b * y0 + c) / np.sqrt(a**2 + b**2)
            return distance
    
    def is_hand_touching_handrail(self, hand_pos: Tuple[int, int], handrail_lines: List[Tuple[int, int, int, int]]) -> Tuple[bool, float]:
        """Check if hand is touching any handrail"""
        if not hand_pos or not handrail_lines:
            return False, float('inf')
        
        min_distance = float('inf')
        
        for line in handrail_lines:
            distance = self.calculate_distance_to_line(hand_pos, line)
            min_distance = min(min_distance, distance)
        
        is_touching = min_distance <= self.touch_threshold
        return is_touching, min_distance
    
    def analyze_frame(self, left_hand: Optional[Tuple[int, int]], 
                     right_hand: Optional[Tuple[int, int]], 
                     handrail_lines: List[Tuple[int, int, int, int]]) -> dict:
        """Analyze both hands for handrail contact"""
        result = {
            'left_hand_touching': False,
            'right_hand_touching': False,
            'left_hand_distance': float('inf'),
            'right_hand_distance': float('inf'),
            'any_hand_touching': False
        }
        
        if left_hand:
            is_touching, distance = self.is_hand_touching_handrail(left_hand, handrail_lines)
            result['left_hand_touching'] = is_touching
            result['left_hand_distance'] = distance
        
        if right_hand:
            is_touching, distance = self.is_hand_touching_handrail(right_hand, handrail_lines)
            result['right_hand_touching'] = is_touching
            result['right_hand_distance'] = distance
        
        result['any_hand_touching'] = result['left_hand_touching'] or result['right_hand_touching']
        
        return result
    
    def draw_analysis(self, frame: np.ndarray, 
                     left_hand: Optional[Tuple[int, int]], 
                     right_hand: Optional[Tuple[int, int]], 
                     analysis_result: dict) -> np.ndarray:
        """Draw analysis results on frame"""
        # Draw hand positions
        if left_hand:
            color = (0, 255, 0) if analysis_result['left_hand_touching'] else (0, 0, 255)
            cv2.circle(frame, left_hand, 10, color, -1)
            cv2.putText(frame, f"L: {analysis_result['left_hand_distance']:.1f}px", 
                       (left_hand[0] + 15, left_hand[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if right_hand:
            color = (0, 255, 0) if analysis_result['right_hand_touching'] else (0, 0, 255)
            cv2.circle(frame, right_hand, 10, color, -1)
            cv2.putText(frame, f"R: {analysis_result['right_hand_distance']:.1f}px", 
                       (right_hand[0] + 15, right_hand[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw overall status
        status = "HOLDING HANDRAIL" if analysis_result['any_hand_touching'] else "NOT HOLDING HANDRAIL"
        color = (0, 255, 0) if analysis_result['any_hand_touching'] else (0, 0, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        return frame