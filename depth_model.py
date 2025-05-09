import os
import torch
import numpy as np
import cv2
from transformers import pipeline
from PIL import Image

from transformer_depth_model import TransformerDepthEstimator 

class TrainedStereoTransformer:
    def __init__(self, weights_path, device='cpu'):
        self.device = device
        self.model = TransformerDepthEstimator().to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
        self.depth_raw = None

    def estimate_depth(self, left, right):
        with torch.no_grad():
            left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
            right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

            left_tensor = torch.tensor(left_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.
            right_tensor = torch.tensor(right_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.

            left_tensor = torch.nn.functional.interpolate(left_tensor, size=(256, 512))
            right_tensor = torch.nn.functional.interpolate(right_tensor, size=(256, 512))

            pred = self.model(left_tensor.to(self.device), right_tensor.to(self.device))
            depth = pred.squeeze().cpu().numpy()

            self.depth_raw = depth
            depth_norm = depth / np.percentile(depth[depth > 0], 95)
            return np.clip(depth_norm, 0, 1)

    def colorize_depth(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_uint8, cmap)

    def get_depth_at_point(self, depth_map, x, y):
        if depth_map is None:
            return 0.0
        return float(depth_map[y, x])

    def get_depth_in_region(self, depth_map, bbox, method='median'):
        if depth_map is None:
            return 0.0
        x1, y1, x2, y2 = map(int, bbox)
        region = depth_map[y1:y2, x1:x2]
        if region.size == 0: return 0.0
        if method == 'mean': return float(np.mean(region))
        if method == 'min': return float(np.min(region))
        return float(np.median(region))

class OpenCVStereoEstimator:
    def __init__(self, focal_length=721.5377, baseline=0.54):
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=5,
            P1=8 * 3 * 5 ** 2,
            P2=32 * 3 * 5 ** 2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        self.focal_length = focal_length
        self.baseline = baseline
        self.depth_raw = None

    def estimate_depth(self, left_image, right_image):
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        with np.errstate(divide='ignore'):
            depth = (self.focal_length * self.baseline) / disparity
            depth[disparity <= 0] = 0

        self.depth_raw = depth

        depth_norm = np.copy(depth)
        depth_norm[depth_norm == np.inf] = 0
        max_depth = np.percentile(depth_norm[depth_norm > 0], 95)
        depth_norm = np.clip(depth_norm / max_depth, 0, 1)

        return depth_norm

    def colorize_depth(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        return cv2.applyColorMap(depth_uint8, cmap)

    def get_depth_at_point(self, depth_map, x, y):
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            return float(depth_map[y, x])
        return 0.0

    def get_depth_in_region(self, depth_map, bbox, method='median'):
        if depth_map is None:
            return 0.0

        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_map.shape[1] - 1, x2)
        y2 = min(depth_map.shape[0] - 1, y2)

        region = depth_map[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0

        if method == 'median':
            return float(np.median(region))
        elif method == 'mean':
            return float(np.mean(region))
        elif method == 'min':
            return float(np.min(region))
        else:
            return float(np.median(region))

class DepthAnythingEstimator:
    """
    Depth estimation using Depth Anything v2
    """
    def __init__(self, model_size='small', device=None):
        """
        Initialize the depth estimator
        
        Args:
            model_size (str): Model size ('small', 'base', 'large')
            device (str): Device to run inference on ('cuda', 'cpu', 'mps')
        """
        # Determine device
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        self.device = device
        
        # Set MPS fallback for operations not supported on Apple Silicon
        if self.device == 'mps':
            print("Using MPS device with CPU fallback for unsupported operations")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            # For Depth Anything v2, we'll use CPU directly due to MPS compatibility issues
            self.pipe_device = 'cpu'
            print("Forcing CPU for depth estimation pipeline due to MPS compatibility issues")
        else:
            self.pipe_device = self.device
        
        print(f"Using device: {self.device} for depth estimation (pipeline on {self.pipe_device})")
        
        # Map model size to model name
        model_map = {
            'small': 'depth-anything/Depth-Anything-V2-Small-hf',
            'base': 'depth-anything/Depth-Anything-V2-Base-hf',
            'large': 'depth-anything/Depth-Anything-V2-Large-hf'
        }
        
        model_name = model_map.get(model_size.lower(), model_map['small'])
        
        # Create pipeline
        try:
            self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.pipe_device)
            print(f"Loaded Depth Anything v2 {model_size} model on {self.pipe_device}")
        except Exception as e:
            # Fallback to CPU if there are issues
            print(f"Error loading model on {self.pipe_device}: {e}")
            print("Falling back to CPU for depth estimation")
            self.pipe_device = 'cpu'
            self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.pipe_device)
            print(f"Loaded Depth Anything v2 {model_size} model on CPU (fallback)")
    
    def estimate_depth(self, image):
        """
        Estimate depth from an image
        
        Args:
            image (numpy.ndarray): Input image (BGR format)
            
        Returns:
            numpy.ndarray: Depth map (normalized to 0-1)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Get depth map
        try:
            depth_result = self.pipe(pil_image)
            depth_map = depth_result["depth"]
            
            # Convert PIL Image to numpy array if needed
            if isinstance(depth_map, Image.Image):
                depth_map = np.array(depth_map)
            elif isinstance(depth_map, torch.Tensor):
                depth_map = depth_map.cpu().numpy()
        except RuntimeError as e:
            # Handle potential MPS errors during inference
            if self.device == 'mps':
                print(f"MPS error during depth estimation: {e}")
                print("Temporarily falling back to CPU for this frame")
                # Create a CPU pipeline for this frame
                cpu_pipe = pipeline(task="depth-estimation", model=self.pipe.model.config._name_or_path, device='cpu')
                depth_result = cpu_pipe(pil_image)
                depth_map = depth_result["depth"]
                
                # Convert PIL Image to numpy array if needed
                if isinstance(depth_map, Image.Image):
                    depth_map = np.array(depth_map)
                elif isinstance(depth_map, torch.Tensor):
                    depth_map = depth_map.cpu().numpy()
            else:
                # Re-raise the error if not MPS
                raise
        
        # Normalize depth map to 0-1
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max > depth_min:
            depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        
        return depth_map
    
    def colorize_depth(self, depth_map, cmap=cv2.COLORMAP_INFERNO):
        """
        Colorize depth map for visualization
        
        Args:
            depth_map (numpy.ndarray): Depth map (normalized to 0-1)
            cmap (int): OpenCV colormap
            
        Returns:
            numpy.ndarray: Colorized depth map (BGR format)
        """
        depth_map_uint8 = (depth_map * 255).astype(np.uint8)
        colored_depth = cv2.applyColorMap(depth_map_uint8, cmap)
        return colored_depth
    
    def get_depth_at_point(self, depth_map, x, y):
        """
        Get depth value at a specific point
        
        Args:
            depth_map (numpy.ndarray): Depth map
            x (int): X coordinate
            y (int): Y coordinate
            
        Returns:
            float: Depth value at (x, y)
        """
        if 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]:
            return depth_map[y, x]
        return 0.0
    
    def get_depth_in_region(self, depth_map, bbox, method='median'):
        """
        Get depth value in a region defined by a bounding box
        
        Args:
            depth_map (numpy.ndarray): Depth map
            bbox (list): Bounding box [x1, y1, x2, y2]
            method (str): Method to compute depth ('median', 'mean', 'min')
            
        Returns:
            float: Depth value in the region
        """
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(depth_map.shape[1] - 1, x2)
        y2 = min(depth_map.shape[0] - 1, y2)
        
        # Extract region
        region = depth_map[y1:y2, x1:x2]
        
        if region.size == 0:
            return 0.0
        
        # Compute depth based on method
        if method == 'median':
            return float(np.median(region))
        elif method == 'mean':
            return float(np.mean(region))
        elif method == 'min':
            return float(np.min(region))
        else:
            return float(np.median(region)) 