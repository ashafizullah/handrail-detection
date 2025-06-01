import cv2
import numpy as np
from typing import List, Tuple, Optional

class PoseDetector:
    def __init__(self):
        # Using OpenPose DNN model (you can download from OpenCV official)
        # For now, we'll use simple hand detection with contours
        self.hand_cascade = None
        try:
            # Try to load hand cascade if available
            self.hand_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_hand.xml')
        except:
            pass
    
    def detect_pose(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect hand positions in frame using simple detection"""
        hands = []
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Simple hand detection using skin color
        hands = self.detect_hands_by_skin_color(frame)
        
        return hands
    
    def detect_hands_by_skin_color(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detect hands using skin color segmentation"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Create mask for skin color
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        hands = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Minimum area for hand
                # Get centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    hands.append((cx, cy))
        
        return hands
    
    def get_hand_coordinates(self, hands: List[Tuple[int, int]]) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Extract left and right hand coordinates"""
        if not hands:
            return None, None
        
        # Sort hands by x coordinate (left to right)
        hands_sorted = sorted(hands, key=lambda x: x[0])
        
        left_hand = hands_sorted[0] if len(hands_sorted) > 0 else None
        right_hand = hands_sorted[-1] if len(hands_sorted) > 1 else None
        
        # If only one hand detected, assume it's the right hand
        if len(hands_sorted) == 1:
            right_hand = hands_sorted[0]
            left_hand = None
        
        return left_hand, right_hand
    
    def draw_pose(self, frame: np.ndarray, hands: List[Tuple[int, int]]) -> np.ndarray:
        """Draw detected hands on frame"""
        for hand in hands:
            cv2.circle(frame, hand, 15, (255, 0, 0), 3)
            cv2.putText(frame, "HAND", (hand[0] + 20, hand[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return frame