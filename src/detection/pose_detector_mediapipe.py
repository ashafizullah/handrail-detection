import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, Dict

class PoseDetectorMediaPipe:
    def __init__(self):
        # Use MediaPipe Holistic for better multi-person detection
        self.mp_holistic = mp.solutions.holistic
        self.mp_pose = mp.solutions.pose
        
        # Primary pose detector
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Secondary pose detector for multi-person (using different ROI)
        self.pose_secondary = self.mp_pose.Pose(
            static_image_mode=True,  # Static mode for better multi-detection
            model_complexity=1,
            min_detection_confidence=0.3
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Person detection using background subtraction
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    
    def detect_multiple_people(self, frame: np.ndarray) -> List[Dict]:
        """Detect multiple people in frame"""
        people = []
        
        # Method 1: Full frame detection
        full_frame_person = self.detect_single_pose(frame)
        if full_frame_person:
            people.append(full_frame_person)
        
        # Method 2: Detect people using background subtraction + ROI
        additional_people = self.detect_people_by_motion(frame)
        people.extend(additional_people)
        
        # Remove duplicates (people too close to each other)
        people = self.remove_duplicate_people(people, frame.shape)
        
        return people
    
    def detect_single_pose(self, frame: np.ndarray):
        """Detect single pose in full frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            return {
                'pose_landmarks': results.pose_landmarks,
                'bbox': self.extract_person_bbox(results.pose_landmarks, frame.shape),
                'confidence': self.get_pose_confidence(results.pose_landmarks)
            }
        return None
    
    def detect_people_by_motion(self, frame: np.ndarray) -> List[Dict]:
        """Detect additional people using motion detection + pose estimation"""
        people = []
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours (potential people)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:  # Minimum area for a person
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (person-like)
                aspect_ratio = h / w
                if 1.5 <= aspect_ratio <= 4.0:  # Typical person aspect ratio
                    # Extract ROI and try pose detection
                    roi = frame[y:y+h, x:x+w]
                    if roi.size > 0:
                        person_data = self.detect_pose_in_roi(roi, (x, y))
                        if person_data:
                            people.append(person_data)
        
        return people
    
    def detect_pose_in_roi(self, roi: np.ndarray, offset: Tuple[int, int]) -> Optional[Dict]:
        """Detect pose in a specific ROI"""
        if roi.shape[0] < 50 or roi.shape[1] < 50:  # Too small ROI
            return None
            
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        results = self.pose_secondary.process(rgb_roi)
        
        if results.pose_landmarks:
            # Adjust landmarks to global coordinates
            adjusted_landmarks = self.adjust_landmarks_to_global(
                results.pose_landmarks, roi.shape, offset
            )
            
            return {
                'pose_landmarks': adjusted_landmarks,
                'bbox': (offset[0], offset[1], roi.shape[1], roi.shape[0]),
                'confidence': self.get_pose_confidence(adjusted_landmarks)
            }
        return None
    
    def adjust_landmarks_to_global(self, landmarks, roi_shape, offset):
        """Adjust ROI landmarks to global frame coordinates"""
        # For now, return original landmarks as coordinate adjustment is complex
        # In production, this would require proper coordinate transformation
        return landmarks
    
    def remove_duplicate_people(self, people: List[Dict], frame_shape) -> List[Dict]:
        """Remove duplicate people detections"""
        if len(people) <= 1:
            return people
        
        filtered_people = []
        min_distance = 100  # Minimum distance between people centers
        
        for person in people:
            bbox = person.get('bbox')
            if not bbox:
                continue
                
            x, y, w, h = bbox
            center = (x + w//2, y + h//2)
            
            is_duplicate = False
            for existing_person in filtered_people:
                existing_bbox = existing_person.get('bbox')
                if existing_bbox:
                    ex, ey, ew, eh = existing_bbox
                    existing_center = (ex + ew//2, ey + eh//2)
                    
                    distance = np.sqrt((center[0] - existing_center[0])**2 + 
                                     (center[1] - existing_center[1])**2)
                    
                    if distance < min_distance:
                        # Keep the one with higher confidence
                        if person.get('confidence', 0) > existing_person.get('confidence', 0):
                            filtered_people.remove(existing_person)
                        else:
                            is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered_people.append(person)
        
        return filtered_people
    
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
        """Main detection method - returns multiple people"""
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
    
    def get_all_keypoints(self, landmarks, frame_shape) -> dict:
        """Get all relevant pose keypoints"""
        if not landmarks:
            return {}
        
        height, width = frame_shape[:2]
        keypoints = {}
        
        # Key landmarks for handrail analysis
        landmark_names = [
            'LEFT_WRIST', 'RIGHT_WRIST',
            'LEFT_ELBOW', 'RIGHT_ELBOW',
            'LEFT_SHOULDER', 'RIGHT_SHOULDER',
            'LEFT_HIP', 'RIGHT_HIP'
        ]
        
        for name in landmark_names:
            landmark = landmarks.landmark[getattr(self.mp_pose.PoseLandmark, name)]
            if landmark.visibility > 0.5:
                keypoints[name.lower()] = (
                    int(landmark.x * width),
                    int(landmark.y * height),
                    landmark.visibility
                )
        
        return keypoints
    
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