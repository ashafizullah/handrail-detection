#!/usr/bin/env python3
"""
Handrail Detection System
Detects whether employees hold handrails while using stairs
"""

import os
import sys
import argparse
from typing import List, Dict, Any
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detection.pose_detector_mediapipe import PoseDetectorMediaPipe
from detection.handrail_detector import HandrailDetector
from detection.annotation_based_detector import AnnotationBasedHandrailDetector
from detection.proximity_analyzer import ProximityAnalyzer
from detection.people_tracker import PeopleTracker
from utils.video_processor import VideoProcessor
from utils.visualization import Visualizer

class HandrailDetectionSystem:
    def __init__(self, touch_threshold: int = 30, annotation_file: str = None):
        self.pose_detector = PoseDetectorMediaPipe()
        self.handrail_detector = HandrailDetector()
        self.annotation_detector = None
        self.proximity_analyzer = ProximityAnalyzer(touch_threshold)
        self.people_tracker = PeopleTracker()
        self.visualizer = Visualizer()
        self.analysis_history: List[Dict[str, Any]] = []
        self.use_annotations = False
        
        # Initialize annotation-based detector if file provided
        if annotation_file and os.path.exists(annotation_file):
            try:
                self.annotation_detector = AnnotationBasedHandrailDetector(annotation_file)
                self.use_annotations = True
                print(f"Using manual annotations from: {annotation_file}")
                
                # Print annotation stats
                stats = self.annotation_detector.get_annotation_stats()
                print(f"Annotation stats: {stats['total_handrails']} handrails in {stats['annotated_frames']} frames")
                print(f"Frame coverage: {stats['frame_coverage']:.1f}%")
                
            except Exception as e:
                print(f"Error loading annotations: {e}")
                print("Falling back to automatic detection")
    
    def process_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """Process a single frame and return annotated result"""
        # Detect multiple people using MediaPipe
        people_data = self.pose_detector.detect_pose(frame)
        
        # Initialize frame-level analysis
        frame_analysis = {
            'frame_number': frame_number,
            'total_people': len(people_data) if people_data else 0,
            'using_handrail': 0,
            'not_using_handrail': 0,
            'people_details': []
        }
        
        # Detect handrails - use annotations if available, otherwise automatic detection
        if self.use_annotations and self.annotation_detector:
            handrail_lines = self.annotation_detector.detect_handrail_edges(frame, frame_number)
            frame_analysis['detection_method'] = 'manual_annotation'
        else:
            handrail_lines_edges = self.handrail_detector.detect_handrail_edges(frame)
            handrail_lines_vertical = self.handrail_detector.detect_vertical_handrails(frame)
            handrail_lines = handrail_lines_edges + handrail_lines_vertical
            handrail_lines = self.filter_duplicate_handrails(handrail_lines)
            frame_analysis['detection_method'] = 'automatic'
        
        frame_analysis['handrail_count'] = len(handrail_lines)
        
        # Process each detected person
        detections_for_tracker = []
        
        if people_data:
            for person_data in people_data:
                pose_landmarks = person_data.get('pose_landmarks')
                if not pose_landmarks:
                    continue
                    
                # Get hand coordinates for this person
                left_hand, right_hand = self.pose_detector.get_hand_coordinates(pose_landmarks, frame.shape)
                
                # Analyze handrail usage for this person
                person_analysis = self.proximity_analyzer.analyze_frame(left_hand, right_hand, handrail_lines)
                person_confidence = self.pose_detector.get_pose_confidence(pose_landmarks)
                
                # Track this person
                bbox = person_data.get('bbox') or self.people_tracker.extract_person_bbox(pose_landmarks, frame.shape)
                if bbox:
                    detection = {
                        'bbox': bbox,
                        'pose_landmarks': pose_landmarks,
                        'left_hand': left_hand,
                        'right_hand': right_hand,
                        'is_using_handrail': person_analysis['any_hand_touching'],
                        'confidence': person_confidence
                    }
                    detections_for_tracker.append(detection)
                
                # Add to frame analysis
                frame_analysis['people_details'].append({
                    'left_hand': left_hand,
                    'right_hand': right_hand,
                    'using_handrail': person_analysis['any_hand_touching'],
                    'confidence': person_confidence
                })
                
                if person_analysis['any_hand_touching']:
                    frame_analysis['using_handrail'] += 1
                else:
                    frame_analysis['not_using_handrail'] += 1
        
        # Update people tracker
        self.people_tracker.update_tracks(detections_for_tracker, frame_number)
        
        # Get final people count from tracker
        people_count = self.people_tracker.get_people_count()
        frame_analysis.update(people_count)
        
        # Store analysis history
        self.analysis_history.append(frame_analysis)
        
        # Visualize results
        result_frame = frame.copy()
        
        # Draw pose with MediaPipe (multiple people)
        result_frame = self.pose_detector.draw_pose(result_frame, people_data)
        
        # Draw handrails (use appropriate drawer based on detection method)
        if self.use_annotations and self.annotation_detector:
            result_frame = self.annotation_detector.draw_handrails(result_frame, handrail_lines)
        else:
            result_frame = self.handrail_detector.draw_handrails(result_frame, handrail_lines)
        
        # Draw detection zones
        result_frame = self.visualizer.draw_detection_zones(result_frame, handrail_lines, 
                                                           self.proximity_analyzer.touch_threshold)
        
        # Draw people tracking
        result_frame = self.people_tracker.draw_people_info(result_frame)
        
        # Draw analysis results for all people
        for person_detail in frame_analysis['people_details']:
            # Create proper analysis result format
            analysis_for_drawing = {
                'any_hand_touching': person_detail['using_handrail'],
                'left_hand_touching': False,  # Simplified for now
                'right_hand_touching': False,  # Simplified for now
                'left_hand_distance': 999,
                'right_hand_distance': 999
            }
            result_frame = self.proximity_analyzer.draw_analysis(
                result_frame, 
                person_detail['left_hand'], 
                person_detail['right_hand'], 
                analysis_for_drawing
            )
        
        # Add info panel
        result_frame = self.visualizer.create_info_panel(result_frame, frame_analysis)
        
        return result_frame
    
    def filter_duplicate_handrails(self, handrail_lines):
        """Remove duplicate or very similar handrail lines"""
        if not handrail_lines:
            return []
        
        filtered_lines = []
        min_distance = 30  # Minimum distance between handrails
        
        for line in handrail_lines:
            x1, y1, x2, y2 = line
            is_duplicate = False
            
            for existing_line in filtered_lines:
                ex1, ey1, ex2, ey2 = existing_line
                
                # Calculate distance between line centers
                center1 = ((x1 + x2) / 2, (y1 + y2) / 2)
                center2 = ((ex1 + ex2) / 2, (ey1 + ey2) / 2)
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                if distance < min_distance:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_lines.append(line)
        
        return filtered_lines
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate summary report of the analysis"""
        if not self.analysis_history:
            return {}
        
        total_frames = len(self.analysis_history)
        frames_with_handrail = sum(1 for data in self.analysis_history if data.get('any_hand_touching', False))
        frames_without_handrail = total_frames - frames_with_handrail
        
        safety_percentage = (frames_with_handrail / total_frames) * 100 if total_frames > 0 else 0
        
        # People statistics
        total_people_detected = sum(data.get('total_people', 0) for data in self.analysis_history)
        people_using_handrail = sum(data.get('using_handrail', 0) for data in self.analysis_history)
        people_not_using_handrail = sum(data.get('not_using_handrail', 0) for data in self.analysis_history)
        
        # Unique people count (max people seen in any frame)
        max_people_in_frame = max(data.get('total_people', 0) for data in self.analysis_history)
        
        # Average people count per frame
        avg_people_per_frame = total_people_detected / total_frames if total_frames > 0 else 0
        
        report = {
            'total_frames': total_frames,
            'frames_using_handrail': frames_with_handrail,
            'frames_not_using_handrail': frames_without_handrail,
            'safety_percentage': safety_percentage,
            'average_handrails_detected': np.mean([data.get('handrail_count', 0) for data in self.analysis_history]),
            
            # People statistics
            'max_people_detected': max_people_in_frame,
            'avg_people_per_frame': avg_people_per_frame,
            'total_people_instances': total_people_detected,
            'people_using_handrail_instances': people_using_handrail,
            'people_not_using_handrail_instances': people_not_using_handrail,
            'people_safety_percentage': (people_using_handrail / total_people_detected * 100) if total_people_detected > 0 else 0
        }
        
        return report

def main():
    parser = argparse.ArgumentParser(description='Handrail Detection System')
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('--output', '-o', help='Path to output video file')
    parser.add_argument('--threshold', '-t', type=int, default=30, 
                       help='Touch detection threshold in pixels (default: 30)')
    parser.add_argument('--annotations', '-a', help='Path to manual annotations JSON file')
    parser.add_argument('--no-display', action='store_true', 
                       help='Disable live video display')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_video):
        print(f"Error: Input video file '{args.input_video}' not found")
        return 1
    
    # Initialize detection system
    detector_system = HandrailDetectionSystem(
        touch_threshold=args.threshold,
        annotation_file=args.annotations
    )
    
    # Process video
    try:
        with VideoProcessor(args.input_video, args.output) as processor:
            print(f"Processing video: {args.input_video}")
            
            # Get video info
            video_info = processor.get_video_info()
            print(f"Video info: {video_info['frame_count']} frames, "
                  f"{video_info['fps']:.2f} FPS, {video_info['duration']:.2f}s")
            
            # Process video
            processor.process_video(
                detector_system.process_frame, 
                show_live=not args.no_display
            )
            
        # Generate and display report
        report = detector_system.generate_report()
        print("\n" + "="*60)
        print("HANDRAIL DETECTION ANALYSIS REPORT")
        print("="*60)
        
        # Show detection method
        detection_method = detector_system.analysis_history[0].get('detection_method', 'unknown') if detector_system.analysis_history else 'unknown'
        print(f"Detection method: {detection_method.replace('_', ' ').title()}")
        
        print(f"Total frames analyzed: {report['total_frames']}")
        print(f"Frames using handrail: {report['frames_using_handrail']}")
        print(f"Frames not using handrail: {report['frames_not_using_handrail']}")
        print(f"Frame-based safety compliance: {report['safety_percentage']:.1f}%")
        print(f"Average handrails detected: {report['average_handrails_detected']:.1f}")
        print()
        print("PEOPLE COUNTING ANALYSIS:")
        print("-" * 30)
        print(f"Maximum people detected in any frame: {report['max_people_detected']}")
        print(f"Average people per frame: {report['avg_people_per_frame']:.1f}")
        print(f"Total people instances: {report['total_people_instances']}")
        print(f"People instances using handrail: {report['people_using_handrail_instances']}")
        print(f"People instances NOT using handrail: {report['people_not_using_handrail_instances']}")
        print(f"People-based safety compliance: {report['people_safety_percentage']:.1f}%")
        
        # Generate summary plot
        summary_plot = detector_system.visualizer.create_summary_plot(detector_system.analysis_history)
        if summary_plot:
            print(f"Summary plot saved: {summary_plot}")
        
        if args.output:
            print(f"Output video saved: {args.output}")
            
    except Exception as e:
        print(f"Error processing video: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())