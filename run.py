#!/usr/bin/env python3
import os
import cv2
import argparse

# Import our modules
from detection_model import YOLOv11Detector, DETRDetector
from depth_model import DepthAnythingEstimator, OpenCVStereoEstimator, TrainedStereoTransformer
from bbox3d_utils import BBox3DEstimator

def parse_args():
    parser = argparse.ArgumentParser(description="3D Object Detection Pipeline")
    parser.add_argument('--camera', type=str, choices=['single', 'stereo'], default='single',
                        help='Camera type: "single" or "stereo"')
    parser.add_argument('--detector', type=str, choices=['yolov11', 'detr'], default='yolov11',
                        help='2D object detector models: "yolov11" or "detr"')
    parser.add_argument('--depth', type=str, choices=['depthanything', 'opencvsgbm', 'transformer'],
                        default='depthanything',
                        help='Depth model: "depthanything", "opencvsgbm", "transformer"')
    parser.add_argument('--data', type=str, default='',
                        help='Path to your dataset (required if camera=stereo)')
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()

    # Configuration variables
    output_path = "output.mp4"  # Path to output video file
    device = 'cpu'  # Force CPU for stability
    conf_threshold = 0.25  # Confidence threshold for object detection
    iou_threshold = 0.45  # IoU threshold for NMS
    classes = None  # Filter by class, e.g., [0, 1, 2] for specific classes, None for all classes
    enable_tracking = True  # Enable object tracking
    
    print(f"Using device: {device}")
    print("Initializing models...")

    # Initialize 2D object detector models
    if args.detector == 'yolov11':
        detector = YOLOv11Detector(
            model_size="nano",
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device=device
        )
    elif args.detector == 'detr':
        detector = DETRDetector(device=device)
    else:
        raise ValueError("Unsupported detection model")

    # Initialize depth estimator
    if args.depth == 'depthanything':
        depth_estimator = DepthAnythingEstimator(model_size="small", device=device)
    elif args.depth == 'opencvsgbm':
        depth_estimator = OpenCVStereoEstimator()
    elif args.depth == 'transformer':
        depth_estimator = TrainedStereoTransformer(weights_path="transformer_depth_kitti.pth", device=device)
    else:
        raise ValueError("Unsupported depth model")
    
    # Initialize 3D bounding box estimator models
    bbox3d_estimator = BBox3DEstimator()

    # Initialize image acquisition methods
    if args.camera == 'single':
        cap = cv2.VideoCapture(0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:  fps = 30
        right_frame = None 
    elif args.camera == 'stereo':
        if not args.data:
            print("Error: --kitti_path must be specified for stereo mode")
            return

        left_dir = os.path.join(args.data, 'image_2')
        right_dir = os.path.join(args.data, 'image_3')

        left_files = sorted([os.path.join(left_dir, f) for f in os.listdir(left_dir) if f.endswith(".png")])
        right_files = sorted([os.path.join(right_dir, f) for f in os.listdir(right_dir) if f.endswith(".png")])
        
        if len(left_files) != len(right_files):
            print("Error: Mismatch in number of left and right images.")
            return
        
        stereo_index = 0
        total_frames = len(left_files)
        fps = 10 
        left_sample = cv2.imread(left_files[0])
        height, width = left_sample.shape[:2]
        cap = None
    else:
        raise ValueError("Unsupported camera type")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("Starting processing...")
    
    while True:  
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
            print("Exiting program...")
            break
        
        if args.camera == 'single':
            ret, frame = cap.read()
            if not ret:
                break
        elif args.camera == 'stereo':
            if stereo_index >= total_frames:
                break
            frame = cv2.imread(left_files[stereo_index])
            right_frame = cv2.imread(right_files[stereo_index])
            stereo_index += 1
        else:
            raise ValueError("Unsupported camera type")
        
        detection_frame = frame.copy()
        result_frame = frame.copy()
        
        # Step 1: Object Detection
        detection_frame, detections = detector.detect(detection_frame, track=enable_tracking)
        
        # Step 2: Depth Estimation
        if args.camera == 'single':
            original_frame = frame.copy()
            depth_map = depth_estimator.estimate_depth(original_frame)
        elif args.camera == 'stereo':
            original_left_frame = frame.copy()
            original_right_frame = right_frame.copy()
            if args.depth == 'depthanything':
                depth_map = depth_estimator.estimate_depth(original_left_frame)
            else:
                depth_map = depth_estimator.estimate_depth(original_left_frame, original_right_frame)
        else:
            raise ValueError("Unsupported camera type")

        depth_colored = depth_estimator.colorize_depth(depth_map)
    
        # Step 3: 3D Bounding Box Estimation
        boxes_3d = []
        active_ids = []
        
        for detection in detections:
            bbox, score, class_id, obj_id = detection
            
            class_name = detector.get_class_names()[class_id]
            
            if class_name.lower() in ['person', 'cat', 'dog']:
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                depth_value = depth_estimator.get_depth_at_point(depth_map, center_x, center_y)
                depth_method = 'center'
            else:
                depth_value = depth_estimator.get_depth_in_region(depth_map, bbox, method='median')
                depth_method = 'median'
            
            box_3d = {
                'bbox_2d': bbox,
                'depth_value': depth_value,
                'depth_method': depth_method,
                'class_name': class_name,
                'object_id': obj_id,
                'score': score
            }
            
            boxes_3d.append(box_3d)
            
            if obj_id is not None:
                active_ids.append(obj_id)
        
        bbox3d_estimator.cleanup_trackers(active_ids)
        
        # Step 4: Visualization
        for box_3d in boxes_3d:
            class_name = box_3d['class_name'].lower()
            if 'car' in class_name or 'vehicle' in class_name:
                color = (0, 0, 255) 
            elif 'person' in class_name:
                color = (0, 255, 0)  
            elif 'bicycle' in class_name or 'motorcycle' in class_name:
                color = (255, 0, 0) 
            elif 'potted plant' in class_name or 'plant' in class_name:
                color = (0, 255, 255)  
            else:
                color = (255, 255, 255) 
            
            result_frame = bbox3d_estimator.draw_box_3d(result_frame, box_3d, color=color)
            
        out.write(result_frame)
        
        cv2.imshow("3D Object Detection", result_frame)
        cv2.imshow("Depth Map", depth_colored)
        cv2.imshow("Object Detection", detection_frame)
    
    print("Cleaning up resources...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Processing complete. Output saved to {output_path}")

if __name__ == "__main__":
    main()
