import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Person:
    id: int
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    left_hand: Optional[Tuple[int, int]]
    right_hand: Optional[Tuple[int, int]]
    pose_landmarks: any
    is_using_handrail: bool
    confidence: float
    last_seen_frame: int

class PeopleTracker:
    def __init__(self):
        self.people: Dict[int, Person] = {}
        self.next_person_id = 1
        self.max_distance_threshold = 100  # Max distance to consider same person
        self.frames_to_forget = 30  # Forget person after N frames
        
    def extract_person_bbox(self, pose_landmarks, frame_shape) -> Optional[Tuple[int, int, int, int]]:
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
    
    def calculate_bbox_distance(self, bbox1: Tuple[int, int, int, int], 
                               bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate distance between two bounding boxes (center points)"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        center1 = (x1 + w1 // 2, y1 + h1 // 2)
        center2 = (x2 + w2 // 2, y2 + h2 // 2)
        
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def update_tracks(self, detections: List[Dict], frame_number: int):
        """Update person tracks with new detections"""
        current_people = {}
        
        # Match detections to existing people
        for detection in detections:
            bbox = detection['bbox']
            pose_landmarks = detection['pose_landmarks']
            left_hand = detection['left_hand']
            right_hand = detection['right_hand']
            is_using_handrail = detection['is_using_handrail']
            confidence = detection['confidence']
            
            # Find best matching existing person
            best_match_id = None
            best_distance = float('inf')
            
            for person_id, person in self.people.items():
                if frame_number - person.last_seen_frame <= self.frames_to_forget:
                    distance = self.calculate_bbox_distance(bbox, person.bbox)
                    if distance < self.max_distance_threshold and distance < best_distance:
                        best_distance = distance
                        best_match_id = person_id
            
            # Update existing person or create new one
            if best_match_id is not None:
                person_id = best_match_id
            else:
                person_id = self.next_person_id
                self.next_person_id += 1
            
            current_people[person_id] = Person(
                id=person_id,
                bbox=bbox,
                left_hand=left_hand,
                right_hand=right_hand,
                pose_landmarks=pose_landmarks,
                is_using_handrail=is_using_handrail,
                confidence=confidence,
                last_seen_frame=frame_number
            )
        
        # Remove old tracks
        self.people = {pid: person for pid, person in current_people.items()}
    
    def get_current_people(self) -> List[Person]:
        """Get list of currently tracked people"""
        return list(self.people.values())
    
    def get_people_count(self) -> Dict[str, int]:
        """Get count of people using/not using handrails"""
        using_handrail = sum(1 for person in self.people.values() if person.is_using_handrail)
        not_using_handrail = len(self.people) - using_handrail
        
        return {
            'total_people': len(self.people),
            'using_handrail': using_handrail,
            'not_using_handrail': not_using_handrail
        }
    
    def draw_people_info(self, frame: np.ndarray) -> np.ndarray:
        """Draw bounding boxes and info for tracked people"""
        for person in self.people.values():
            x, y, w, h = person.bbox
            
            # Draw bounding box
            color = (0, 255, 0) if person.is_using_handrail else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw person ID and status
            status = "SAFE" if person.is_using_handrail else "UNSAFE"
            label = f"Person {person.id}: {status}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw confidence
            conf_text = f"Conf: {person.confidence:.2f}"
            cv2.putText(frame, conf_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame