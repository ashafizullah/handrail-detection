import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Dict

class StablePoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        
        # Primary pose detector (more conservative settings)
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.7,  # Increased from 0.5
            min_tracking_confidence=0.7    # Increased from 0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Tracking state for stability
        self.previous_detections = []
        self.detection_history = []
        self.stable_detection_threshold = 3  # Need 3 consecutive detections
        
    def detect_multiple_people(self, frame: np.ndarray) -> List[Dict]:
        """Detect multiple people with stability filtering"""
        people = []
        
        # Method 1: Primary MediaPipe detection (most reliable)
        primary_person = self.detect_single_pose(frame)
        if primary_person and self.is_detection_stable(primary_person):
            people.append(primary_person)
        
        # Method 2: Only use motion detection if primary detection confidence is high
        if primary_person and primary_person.get('confidence', 0) > 0.8:
            # Only look for additional people if we have a confident primary detection
            additional_people = self.detect_additional_people_conservative(frame)
            for person in additional_people:
                if self.is_detection_stable(person):
                    people.append(person)
        
        # Update detection history
        self.update_detection_history(people)
        
        # Filter out unstable detections
        stable_people = self.filter_stable_detections(people)
        
        return stable_people
    
    def detect_single_pose(self, frame: np.ndarray):
        """Detect single pose in full frame with high confidence"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            confidence = self.get_pose_confidence(results.pose_landmarks)
            if confidence > 0.6:  # Only accept high confidence detections
                return {
                    'pose_landmarks': results.pose_landmarks,
                    'bbox': self.extract_person_bbox(results.pose_landmarks, frame.shape),
                    'confidence': confidence,
                    'detection_method': 'primary_mediapipe'
                }
        return None
    
    def detect_additional_people_conservative(self, frame: np.ndarray) -> List[Dict]:
        """Conservative additional people detection (disabled for now to reduce noise)"""
        # Temporarily disable additional detection to prevent flickering
        # This was the main source of false positives
        return []
    
    def is_detection_stable(self, detection: Dict) -> bool:
        """Check if detection is stable (not flickering)"""
        if not detection or not detection.get('bbox'):
            return False
        
        bbox = detection['bbox']
        confidence = detection.get('confidence', 0)
        
        # Require minimum confidence
        if confidence < 0.5:
            return False
        
        # Check if detection is similar to previous ones
        for prev_detection in self.previous_detections[-5:]:  # Check last 5 detections
            if self.calculate_bbox_overlap(bbox, prev_detection.get('bbox', (0,0,0,0))) > 0.3:
                return True
        
        # For new detections, require higher confidence
        return confidence > 0.7
    
    def calculate_bbox_overlap(self, bbox1: Tuple[int, int, int, int], 
                              bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update_detection_history(self, detections: List[Dict]):
        """Update detection history for stability analysis"""
        self.previous_detections = detections.copy()
        self.detection_history.append(len(detections))
        
        # Keep only recent history
        if len(self.detection_history) > 30:
            self.detection_history = self.detection_history[-30:]
    
    def filter_stable_detections(self, detections: List[Dict]) -> List[Dict]:
        """Filter out unstable/flickering detections"""
        stable_detections = []
        
        for detection in detections:
            # Check detection method
            method = detection.get('detection_method', 'unknown')
            confidence = detection.get('confidence', 0)
            
            # Always accept high-confidence primary MediaPipe detections
            if method == 'primary_mediapipe' and confidence > 0.6:
                stable_detections.append(detection)
            # Be more strict with other detection methods
            elif method != 'primary_mediapipe' and confidence > 0.8:
                stable_detections.append(detection)
        
        return stable_detections
    
    def extract_person_bbox(self, pose_landmarks, frame_shape):
        """Extract bounding box from pose landmarks"""
        if not pose_landmarks:
            return None
        
        height, width = frame_shape[:2]
        
        # Get all visible landmarks
        x_coords = []
        y_coords = []
        
        for landmark in pose_landmarks.landmark:
            if landmark.visibility > 0.5:
                x_coords.append(int(landmark.x * width))
                y_coords.append(int(landmark.y * height))
        
        if not x_coords or not y_coords:
            return None
        
        # Calculate bounding box with some padding
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        padding = 20
        x = max(0, min_x - padding)
        y = max(0, min_y - padding)
        w = min(width - x, max_x - min_x + 2 * padding)
        h = min(height - y, max_y - min_y + 2 * padding)
        
        return (x, y, w, h)
    
    def detect_pose(self, frame: np.ndarray):
        """Main detection method - returns stable people only"""
        return self.detect_multiple_people(frame)
    
    def get_hand_coordinates(self, landmarks, frame_shape) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        """Extract left and right hand coordinates from MediaPipe landmarks"""
        if not landmarks:
            return None, None
        
        height, width = frame_shape[:2]
        
        # Get wrist coordinates (more reliable than hand landmarks)
        left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        
        left_hand = None
        right_hand = None
        
        # Check visibility and get coordinates
        if left_wrist.visibility > 0.5:
            left_hand = (int(left_wrist.x * width), int(left_wrist.y * height))
        
        if right_wrist.visibility > 0.5:
            right_hand = (int(right_wrist.x * width), int(right_wrist.y * height))
        
        return left_hand, right_hand
    
    def draw_pose(self, frame: np.ndarray, people_data) -> np.ndarray:
        """Draw pose landmarks for multiple people"""
        if not people_data:
            return frame
            
        # Handle both single landmarks (backward compatibility) and multiple people
        if hasattr(people_data, 'landmark'):  # Single landmarks object
            people_list = [{'pose_landmarks': people_data, 'bbox': None}]
        elif isinstance(people_data, list):  # List of people
            people_list = people_data
        else:
            return frame
        
        for person_data in people_list:
            landmarks = person_data.get('pose_landmarks')
            if not landmarks:
                continue
                
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                frame, 
                landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Highlight hands specifically
            height, width = frame.shape[:2]
            left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            
            if left_wrist.visibility > 0.5:
                x, y = int(left_wrist.x * width), int(left_wrist.y * height)
                cv2.circle(frame, (x, y), 12, (0, 255, 0), -1)  # Green circle for left hand
                cv2.putText(frame, "L", (x-6, y+6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            if right_wrist.visibility > 0.5:
                x, y = int(right_wrist.x * width), int(right_wrist.y * height)
                cv2.circle(frame, (x, y), 12, (255, 0, 0), -1)  # Blue circle for right hand
                cv2.putText(frame, "R", (x-6, y+6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame
    
    def get_pose_confidence(self, landmarks) -> float:
        """Calculate overall pose detection confidence"""
        if not landmarks:
            return 0.0
        
        # Calculate average visibility of key landmarks
        key_landmarks = [
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER
        ]
        
        total_visibility = sum(landmarks.landmark[lm].visibility for lm in key_landmarks)
        return total_visibility / len(key_landmarks)
    
    def is_person_on_stairs(self, landmarks, frame_shape) -> bool:
        """Detect if person is likely on stairs based on pose"""
        if not landmarks:
            return False
        
        height, width = frame_shape[:2]
        
        # Get key points
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        right_ankle = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        
        # Check if person is in walking/climbing position
        if (left_hip.visibility > 0.5 and right_hip.visibility > 0.5 and
            left_ankle.visibility > 0.5 and right_ankle.visibility > 0.5):
            
            # Calculate center of hips
            hip_center_y = (left_hip.y + right_hip.y) / 2
            ankle_center_y = (left_ankle.y + right_ankle.y) / 2
            
            # If person is standing/walking (hips above ankles)
            if hip_center_y < ankle_center_y:
                return True
        
        return False