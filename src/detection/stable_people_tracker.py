import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class StablePerson:
    id: int
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    left_hand: Optional[Tuple[int, int]]
    right_hand: Optional[Tuple[int, int]]
    pose_landmarks: any
    is_using_handrail: bool
    confidence: float
    last_seen_frame: int
    detection_count: int  # Number of consecutive detections
    stability_score: float  # Stability metric

class StablePeopleTracker:
    def __init__(self):
        self.people: Dict[int, StablePerson] = {}
        self.next_person_id = 1
        self.max_distance_threshold = 80  # Reduced from 100
        self.frames_to_forget = 10  # Reduced from 30 for faster cleanup
        self.min_detection_count = 3  # Require 3 consecutive detections
        self.min_stability_score = 0.6
        
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
        
        center1 = (x1 + w1//2, y1 + h1//2)
        center2 = (x2 + w2//2, y2 + h2//2)
        
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def calculate_stability_score(self, detection: Dict, existing_person: StablePerson = None) -> float:
        """Calculate stability score for a detection"""
        base_score = detection.get('confidence', 0)
        
        # Boost score for MediaPipe primary detections
        if detection.get('detection_method') == 'primary_mediapipe':
            base_score += 0.2
        
        # If matching existing person, boost stability
        if existing_person:
            base_score += min(existing_person.detection_count * 0.1, 0.3)
            base_score += existing_person.stability_score * 0.2
        
        return min(base_score, 1.0)
    
    def update_tracks(self, detections: List[Dict], frame_number: int):
        """Update person tracks with new detections (more conservative)"""
        if not detections:
            # Remove people not seen recently
            self.cleanup_old_tracks(frame_number)
            return
        
        current_people = {}
        used_detection_indices = set()
        
        # Match detections to existing people first
        for person_id, person in self.people.items():
            if frame_number - person.last_seen_frame <= self.frames_to_forget:
                best_match_idx = None
                best_distance = float('inf')
                
                for i, detection in enumerate(detections):
                    if i in used_detection_indices:
                        continue
                    
                    bbox = detection.get('bbox')
                    if not bbox:
                        continue
                    
                    distance = self.calculate_bbox_distance(bbox, person.bbox)
                    if distance < self.max_distance_threshold and distance < best_distance:
                        best_distance = distance
                        best_match_idx = i
                
                if best_match_idx is not None:
                    # Update existing person
                    detection = detections[best_match_idx]
                    used_detection_indices.add(best_match_idx)
                    
                    stability_score = self.calculate_stability_score(detection, person)
                    
                    current_people[person_id] = StablePerson(
                        id=person_id,
                        bbox=detection['bbox'],
                        left_hand=detection['left_hand'],
                        right_hand=detection['right_hand'],
                        pose_landmarks=detection['pose_landmarks'],
                        is_using_handrail=detection['is_using_handrail'],
                        confidence=detection['confidence'],
                        last_seen_frame=frame_number,
                        detection_count=person.detection_count + 1,
                        stability_score=stability_score
                    )
        
        # Create new people for unmatched detections (with stricter criteria)
        for i, detection in enumerate(detections):
            if i in used_detection_indices:
                continue
            
            bbox = detection.get('bbox')
            if not bbox:
                continue
            
            # Only create new person if detection is high confidence
            confidence = detection.get('confidence', 0)
            if confidence < 0.7:  # Higher threshold for new people
                continue
            
            person_id = self.next_person_id
            self.next_person_id += 1
            
            stability_score = self.calculate_stability_score(detection)
            
            current_people[person_id] = StablePerson(
                id=person_id,
                bbox=bbox,
                left_hand=detection['left_hand'],
                right_hand=detection['right_hand'],
                pose_landmarks=detection['pose_landmarks'],
                is_using_handrail=detection['is_using_handrail'],
                confidence=confidence,
                last_seen_frame=frame_number,
                detection_count=1,  # New person starts with count 1
                stability_score=stability_score
            )
        
        # Only keep stable people
        stable_people = {}
        for person_id, person in current_people.items():
            # Require minimum detection count and stability score
            if (person.detection_count >= self.min_detection_count and 
                person.stability_score >= self.min_stability_score):
                stable_people[person_id] = person
            # Keep people with high immediate confidence even if new
            elif person.confidence > 0.8 and person.detection_count >= 1:
                stable_people[person_id] = person
        
        self.people = stable_people
    
    def cleanup_old_tracks(self, frame_number: int):
        """Remove old tracks that haven't been seen recently"""
        active_people = {}
        for person_id, person in self.people.items():
            if frame_number - person.last_seen_frame <= self.frames_to_forget:
                active_people[person_id] = person
        self.people = active_people
    
    def get_current_people(self) -> List[StablePerson]:
        """Get list of currently tracked stable people"""
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
        """Draw bounding boxes and info for tracked people (with stability indicators)"""
        for person in self.people.values():
            x, y, w, h = person.bbox
            
            # Color based on stability and handrail usage
            if person.stability_score < 0.7:
                # Unstable detection - yellow warning
                color = (0, 255, 255)
                status_prefix = "?"
            elif person.is_using_handrail:
                # Stable and safe - green
                color = (0, 255, 0)
                status_prefix = "✓"
            else:
                # Stable but unsafe - red
                color = (0, 0, 255)
                status_prefix = "✗"
            
            # Draw bounding box with thickness based on stability
            thickness = 3 if person.stability_score > 0.8 else 2
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw person ID and status
            status = "SAFE" if person.is_using_handrail else "UNSAFE"
            label = f"{status_prefix}P{person.id}: {status}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw stability info
            stability_text = f"S:{person.stability_score:.1f} D:{person.detection_count}"
            cv2.putText(frame, stability_text, (x, y + h + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw confidence
            conf_text = f"C:{person.confidence:.2f}"
            cv2.putText(frame, conf_text, (x, y + h + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    def get_tracking_stats(self) -> Dict[str, any]:
        """Get tracking statistics for debugging"""
        if not self.people:
            return {
                'total_tracked': 0,
                'avg_stability': 0.0,
                'avg_confidence': 0.0,
                'avg_detection_count': 0
            }
        
        return {
            'total_tracked': len(self.people),
            'avg_stability': np.mean([p.stability_score for p in self.people.values()]),
            'avg_confidence': np.mean([p.confidence for p in self.people.values()]),
            'avg_detection_count': np.mean([p.detection_count for p in self.people.values()]),
            'people_ids': [p.id for p in self.people.values()]
        }