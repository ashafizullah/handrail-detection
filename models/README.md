# Models Directory

This directory is for storing machine learning models and configurations.

## Model Types
- **MediaPipe Models**: Automatically downloaded pose estimation models
- **Custom Models**: Any custom trained models for handrail detection
- **Configuration Files**: Model parameter files and settings

## Auto-Downloaded Models
MediaPipe will automatically download required models on first run:
- `pose_landmark_full.tflite`
- `pose_detection.tflite`
- Related model files

## Custom Models
Place any custom trained models here:
- Handrail detection models
- Person detection models
- Classification models

## Note
Model files are ignored by git due to file size. Only the directory structure and documentation are tracked.