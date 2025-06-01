import cv2
import numpy as np
from typing import List, Tuple, Optional

class HandrailDetector:
    def __init__(self):
        self.kernel = np.ones((3, 3), np.uint8)
    
    def detect_handrail_edges(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect handrail using edge detection and line detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Morphological operations to connect broken lines
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.kernel)
        
        # Detect lines using HoughLinesP with stricter parameters
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=80,      # Increased threshold
            minLineLength=150, # Longer minimum length
            maxLineGap=5       # Smaller gap tolerance
        )
        
        handrail_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Filter for roughly vertical lines (handrails are usually vertical)
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if 60 <= angle <= 120:  # Vertical-ish lines (60-120 degrees)
                    # Also check line length - handrails should be reasonably long
                    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    if line_length > 50:  # Minimum length for handrail
                        handrail_lines.append((x1, y1, x2, y2))
        
        return handrail_lines
    
    def detect_vertical_handrails(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Specialized method for detecting vertical handrails"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use Sobel filter to enhance vertical edges
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobelx = np.uint8(np.absolute(sobelx))
        
        # Threshold to get strong vertical edges
        _, thresh = cv2.threshold(sobelx, 50, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to connect vertical lines
        kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_vertical)
        
        # Detect lines using HoughLinesP with stricter parameters for vertical lines
        lines = cv2.HoughLinesP(
            thresh,
            rho=1,
            theta=np.pi/180,
            threshold=60,      # Higher threshold
            minLineLength=120, # Longer minimum length
            maxLineGap=15      # Smaller gap
        )
        
        handrail_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Filter for vertical lines
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if 70 <= angle <= 110:  # Nearly vertical lines
                    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    if line_length > 60:  # Minimum length
                        handrail_lines.append((x1, y1, x2, y2))
        
        return handrail_lines
    
    def detect_handrail_contours(self, frame: np.ndarray) -> List[np.ndarray]:
        """Alternative method: detect handrail using contours for vertical structures"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours that could be vertical handrails
        handrail_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 300:  # Minimum area threshold
                # Check aspect ratio for vertical structures
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w  # Height over width for vertical
                if aspect_ratio > 2:  # Tall structures (vertical)
                    handrail_contours.append(contour)
        
        return handrail_contours
    
    def draw_handrails(self, frame: np.ndarray, lines: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """Draw detected handrail lines on frame"""
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        return frame