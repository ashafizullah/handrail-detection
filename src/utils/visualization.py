import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any

class Visualizer:
    def __init__(self):
        self.colors = {
            'handrail': (0, 255, 0),      # Green
            'hand_touching': (0, 255, 0),  # Green
            'hand_not_touching': (0, 0, 255),  # Red
            'pose': (255, 0, 0),          # Blue
            'text': (255, 255, 255)       # White
        }
    
    def create_info_panel(self, frame: np.ndarray, analysis_data: Dict[str, Any]) -> np.ndarray:
        """Create information panel on the frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay (larger for more info)
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, height - 210), (500, height - 10), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Add text information
        y_offset = height - 190
        line_height = 25
        
        texts = [
            f"Frame: {analysis_data.get('frame_number', 0)}",
            f"People Count: {analysis_data.get('total_people', 0)}",
            f"Using Handrail: {analysis_data.get('using_handrail', 0)}",
            f"NOT Using Handrail: {analysis_data.get('not_using_handrail', 0)}",
            f"Pose Confidence: {analysis_data.get('pose_confidence', 0):.2f}",
            f"Safety Status: {'SAFE' if analysis_data.get('any_hand_touching', False) else 'UNSAFE'}",
            f"On Stairs: {'YES' if analysis_data.get('on_stairs', False) else 'NO'}",
            f"Handrails Detected: {analysis_data.get('handrail_count', 0)}"
        ]
        
        for i, text in enumerate(texts):
            color = self.colors['text']
            if 'TOUCHING' in text and 'NOT' not in text:
                color = self.colors['hand_touching']
            elif 'NOT TOUCHING' in text or 'NOT Using' in text:
                color = self.colors['hand_not_touching']
            elif 'SAFE' in text:
                color = self.colors['hand_touching']
            elif 'UNSAFE' in text:
                color = self.colors['hand_not_touching']
            
            cv2.putText(frame, text, (20, y_offset + i * line_height), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def draw_detection_zones(self, frame: np.ndarray, handrail_lines: List, threshold: int = 30) -> np.ndarray:
        """Draw detection zones around handrails"""
        for line in handrail_lines:
            x1, y1, x2, y2 = line
            
            # Calculate perpendicular vectors for zone boundaries
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                # Normalize and create perpendicular vector
                norm_x = -dy / length
                norm_y = dx / length
                
                # Create zone points
                zone_points = np.array([
                    [x1 + norm_x * threshold, y1 + norm_y * threshold],
                    [x2 + norm_x * threshold, y2 + norm_y * threshold],
                    [x2 - norm_x * threshold, y2 - norm_y * threshold],
                    [x1 - norm_x * threshold, y1 - norm_y * threshold]
                ], np.int32)
                
                # Draw semi-transparent zone
                overlay = frame.copy()
                cv2.fillPoly(overlay, [zone_points], (0, 255, 255))
                frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        return frame
    
    def create_summary_plot(self, analysis_history: List[Dict]) -> np.ndarray:
        """Create summary plot of handrail usage over time"""
        if not analysis_history:
            return None
        
        frames = [data['frame_number'] for data in analysis_history]
        left_touching = [data.get('left_hand_touching', False) for data in analysis_history]
        right_touching = [data.get('right_hand_touching', False) for data in analysis_history]
        any_touching = [data.get('any_hand_touching', False) for data in analysis_history]
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
        
        # Left hand
        ax1.plot(frames, left_touching, 'b-', linewidth=2, label='Left Hand')
        ax1.set_ylabel('Left Hand Touching')
        ax1.set_ylim(-0.1, 1.1)
        ax1.grid(True, alpha=0.3)
        
        # Right hand
        ax2.plot(frames, right_touching, 'r-', linewidth=2, label='Right Hand')
        ax2.set_ylabel('Right Hand Touching')
        ax2.set_ylim(-0.1, 1.1)
        ax2.grid(True, alpha=0.3)
        
        # Overall safety
        ax3.plot(frames, any_touching, 'g-', linewidth=2, label='Any Hand Touching')
        ax3.set_ylabel('Safety Status')
        ax3.set_xlabel('Frame Number')
        ax3.set_ylim(-0.1, 1.1)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/output/handrail_analysis_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return 'data/output/handrail_analysis_summary.png'