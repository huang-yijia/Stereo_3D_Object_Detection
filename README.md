# Stereo 3D Object Detection

This project presents a modular pipeline for 3D Object Detection, integrating object detection, depth estimation, and 3D bounding box projection. The object detection module supports both a YOLOv11-based approach inspired by Faster-RCNN and a transformer-based approach using DETR. For depth estimation, it includes both Depth Anything v2 for monocular depth estimation and a stereo camera-based depth estimation algorithm that leverages binocular disparity to recover scene depth more accurately. The final 3D bounding box module combines 2D detection results with depth information to generate spatially aware 3D boxes, complete with Bird’s Eye View (BEV) visualization and pseudo-3D overlays. This pipeline is designed to be flexible and extensible for future research and experimentation in 3D perception.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenCV
- NumPy
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/huang-yijia/Stereo_3D_Object_Detection.git
   cd Stereo_3D_object_Detection
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Download model weights (will be downloaded automatically on first run)

## Usage

Run the main script:

```bash
python run.py
```

### Configuration Options

You can modify the following parameters in `run.py`:

- **Input/Output**:
  - `source`: Path to input video file or webcam index (0 for default camera)
  - `output_path`: Path to output video file

- **Model Settings**:
  - `yolo_model_size`: YOLOv11 model size ("nano", "small", "medium", "large", "extra")
  - `depth_model_size`: Depth Anything v2 model size ("small", "base", "large")

- **Detection Settings**:
  - `conf_threshold`: Confidence threshold for object detection
  - `iou_threshold`: IoU threshold for NMS
  - `classes`: Filter by class, e.g., [0, 1, 2] for specific classes, None for all classes

- **Feature Toggles**:
  - `enable_tracking`: Enable object tracking
  - `enable_bev`: Enable Bird's Eye View visualization
  - `enable_pseudo_3d`: Enable 3D visualization
