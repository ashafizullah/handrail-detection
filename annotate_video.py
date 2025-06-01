#!/usr/bin/env python3
"""
Handrail Manual Annotation Tool

Usage:
    python annotate_video.py data/videos/office.mp4
    python annotate_video.py data/videos/office.mp4 --annotation-file custom_annotations.json
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.manual_annotation import main

if __name__ == "__main__":
    main()