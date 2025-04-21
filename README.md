# Stereo 3D Object Detection

This project presents a modular pipeline for 3D Object Detection, integrating object detection, depth estimation, and 3D bounding box projection. The object detection module supports both a YOLOv11-based approach inspired by Faster-RCNN and a transformer-based approach using DETR. For depth estimation, it includes both Depth Anything v2 for single camera depth estimation and an OpenCV-SGBM for stereo camera depth estimation. The final 3D bounding box module combines 2D detection results with depth information to generate spatially aware 3D boxes. This pipeline is designed to be flexible and extensible for future research and experimentation in 3D perception.

## Requirements

- Python 3.8+
- PyTorch 2.0+
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
python3 run.py
```

Custom run the main script:

```bash
python3 run.py --camera [single|stereo] --detector [YOLOv11|DETR] --depth [depthanything|opencvsgbm|transformer] --data_path /path/to/your/dataset/folder
```