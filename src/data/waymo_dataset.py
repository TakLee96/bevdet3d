import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import box_utils
from waymo_open_dataset.utils import transform_utils
import cv2
from typing import Dict, List, Optional, Tuple, Any
import logging

# Disable TensorFlow logging
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class WaymoDataset(Dataset):
    """
    Waymo Open Dataset for 3D object detection.
    
    This class loads multi-camera images and LiDAR point clouds from Waymo tfrecord files
    and prepares them for PETR/BEVFormer training.
    """
    
    # Class labels mapping
    CLASSES = {
        0: 'UNKNOWN',
        1: 'VEHICLE', 
        2: 'PEDESTRIAN',
        3: 'SIGN',
        4: 'CYCLIST'
    }
    
    # Camera names
    CAMERA_NAMES = [
        'FRONT',
        'FRONT_LEFT', 
        'FRONT_RIGHT',
        'SIDE_LEFT',
        'SIDE_RIGHT'
    ]
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'training',
                 num_frames_per_segment: int = 10,
                 use_camera_sync_labels: bool = True,
                 image_size: Tuple[int, int] = (1600, 640),
                 point_cloud_range: List[float] = [-74.88, -74.88, -2, 74.88, 74.88, 4],
                 load_point_cloud: bool = True,
                 max_num_points: int = 100000):
        """
        Args:
            data_root: Path to waymo dataset root
            split: 'training', 'validation', or 'testing' 
            num_frames_per_segment: Number of frames to load per tfrecord segment
            use_camera_sync_labels: Whether to use camera-synced labels instead of LiDAR-synced
            image_size: Target image size (width, height)
            point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max] in meters
            load_point_cloud: Whether to load LiDAR point cloud data
            max_num_points: Maximum number of LiDAR points to keep
        """
        self.data_root = data_root
        self.split = split 
        self.num_frames_per_segment = num_frames_per_segment
        self.use_camera_sync_labels = use_camera_sync_labels
        self.image_size = image_size
        self.point_cloud_range = np.array(point_cloud_range)
        self.load_point_cloud = load_point_cloud
        self.max_num_points = max_num_points
        
        # Find all tfrecord files
        self.tfrecord_files = self._get_tfrecord_files()
        
        # Load frame info from all files
        self.frame_infos = self._load_frame_infos()
        
        logging.info(f"Loaded {len(self.frame_infos)} frames from {len(self.tfrecord_files)} files")
        
    def _get_tfrecord_files(self) -> List[str]:
        """Get list of tfrecord files for the split."""
        split_dir = os.path.join(self.data_root, 'individual_files', self.split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} does not exist")
            
        tfrecord_files = []
        for filename in os.listdir(split_dir):
            if filename.endswith('.tfrecord'):
                tfrecord_files.append(os.path.join(split_dir, filename))
                
        tfrecord_files.sort()
        return tfrecord_files
        
    def _load_frame_infos(self) -> List[Dict]:
        """Load frame info from tfrecord files."""
        frame_infos = []
        
        for tfrecord_path in self.tfrecord_files:
            try:
                dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
                frame_count = 0
                
                for data in dataset.take(self.num_frames_per_segment):
                    frame = open_dataset.Frame()
                    frame.ParseFromString(data.numpy())
                    
                    frame_info = {
                        'tfrecord_path': tfrecord_path,
                        'frame_name': frame.context.name,
                        'timestamp': frame.timestamp_micros,
                        'num_cameras': len(frame.images),
                        'num_labels': len(frame.laser_labels)
                    }
                    frame_infos.append(frame_info)
                    frame_count += 1
                    
            except Exception as e:
                logging.warning(f"Error loading {tfrecord_path}: {e}")
                continue
                
        return frame_infos
        
    def __len__(self) -> int:
        return len(self.frame_infos)
        
    def _load_frame_from_tfrecord(self, tfrecord_path: str, frame_idx: int) -> open_dataset.Frame:
        """Load a specific frame from tfrecord file."""
        dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
        
        for i, data in enumerate(dataset):
            if i == frame_idx:
                frame = open_dataset.Frame()
                frame.ParseFromString(data.numpy())
                return frame
                
        raise IndexError(f"Frame {frame_idx} not found in {tfrecord_path}")
        
    def _process_images(self, frame: open_dataset.Frame) -> Tuple[torch.Tensor, List[Dict]]:
        """Process camera images from frame."""
        images = []
        camera_infos = []
        
        # Sort images by camera name for consistent ordering
        sorted_images = sorted(frame.images, key=lambda x: x.name)
        
        for camera_image in sorted_images:
            # Decode image
            image_bytes = camera_image.image
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, self.image_size)
            
            # Convert to tensor and normalize
            image = torch.from_numpy(image).float() / 255.0
            image = image.permute(2, 0, 1)  # HWC -> CHW
            images.append(image)
            
            # Get camera calibration
            camera_calib = None
            for calib in frame.context.camera_calibrations:
                if calib.name == camera_image.name:
                    camera_calib = calib
                    break
                    
            if camera_calib is not None:
                # Extract camera intrinsics and extrinsics
                intrinsic = np.array(camera_calib.intrinsic).reshape(3, 3)
                extrinsic = np.array(camera_calib.extrinsic.transform).reshape(4, 4)
                
                # Scale intrinsics for resized image
                scale_x = self.image_size[0] / camera_calib.width
                scale_y = self.image_size[1] / camera_calib.height
                intrinsic[0, :] *= scale_x
                intrinsic[1, :] *= scale_y
                
                camera_info = {
                    'camera_name': open_dataset.CameraName.Name.Name(camera_image.name),
                    'intrinsic': intrinsic,
                    'extrinsic': extrinsic,
                    'image_shape': self.image_size
                }
                camera_infos.append(camera_info)
                
        images = torch.stack(images, dim=0)  # [N_cams, 3, H, W]
        return images, camera_infos
        
    def _process_point_cloud(self, frame: open_dataset.Frame) -> Optional[torch.Tensor]:
        """Process LiDAR point cloud from frame."""
        if not self.load_point_cloud:
            return None
            
        # For now, return a dummy point cloud
        # TODO: Implement proper point cloud extraction from range images
        # This requires implementing the range image to point cloud conversion
        dummy_points = np.random.randn(1000, 3) * 10  # Random points for testing
        
        # Filter points within range
        mask = (
            (dummy_points[:, 0] >= self.point_cloud_range[0]) &
            (dummy_points[:, 0] <= self.point_cloud_range[3]) &
            (dummy_points[:, 1] >= self.point_cloud_range[1]) &
            (dummy_points[:, 1] <= self.point_cloud_range[4]) &
            (dummy_points[:, 2] >= self.point_cloud_range[2]) &
            (dummy_points[:, 2] <= self.point_cloud_range[5])
        )
        points_filtered = dummy_points[mask]
        
        # Downsample if too many points
        if len(points_filtered) > self.max_num_points:
            indices = np.random.choice(len(points_filtered), self.max_num_points, replace=False)
            points_filtered = points_filtered[indices]
            
        return torch.from_numpy(points_filtered).float()
        
    def _process_labels(self, frame: open_dataset.Frame) -> Dict:
        """Process 3D object labels from frame."""
        boxes_3d = []
        labels = []
        
        for laser_label in frame.laser_labels:
            # Choose which box to use
            if self.use_camera_sync_labels and laser_label.camera_synced_box.ByteSize():
                box = laser_label.camera_synced_box
            else:
                box = laser_label.box
                
            # Convert to numpy array [x, y, z, length, width, height, heading]
            box_array = np.array([
                box.center_x, box.center_y, box.center_z,
                box.length, box.width, box.height,
                box.heading
            ])
            boxes_3d.append(box_array)
            labels.append(laser_label.type)
            
        if len(boxes_3d) > 0:
            boxes_3d = np.stack(boxes_3d, axis=0)
        else:
            boxes_3d = np.zeros((0, 7))
            
        return {
            'gt_bboxes_3d': torch.from_numpy(boxes_3d).float(),
            'gt_labels_3d': torch.from_numpy(np.array(labels)).long(),
        }
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a data sample."""
        frame_info = self.frame_infos[idx]
        
        # Find frame index within the tfrecord file
        frame_idx = 0
        current_file = frame_info['tfrecord_path']
        for i in range(idx):
            if self.frame_infos[i]['tfrecord_path'] == current_file:
                frame_idx += 1
            else:
                frame_idx = 0
                current_file = self.frame_infos[i]['tfrecord_path']
                
        # Load frame from tfrecord
        frame = self._load_frame_from_tfrecord(frame_info['tfrecord_path'], frame_idx)
        
        # Process data
        images, camera_infos = self._process_images(frame)
        labels = self._process_labels(frame)
        
        sample = {
            'images': images,  # [N_cams, 3, H, W]
            'camera_infos': camera_infos,
            'frame_name': frame_info['frame_name'],
            'timestamp': frame_info['timestamp'],
            **labels
        }
        
        if self.load_point_cloud:
            points = self._process_point_cloud(frame)
            sample['points'] = points
            
        return sample


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for DataLoader."""
    collated = {}
    
    # Stack tensors
    for key in ['images']:
        if key in batch[0]:
            collated[key] = torch.stack([item[key] for item in batch])
            
    # Keep lists for variable-length data  
    for key in ['gt_bboxes_3d', 'gt_labels_3d', 'camera_infos', 'frame_name', 'timestamp']:
        if key in batch[0]:
            collated[key] = [item[key] for item in batch]
            
    # Handle point clouds (variable length)
    if 'points' in batch[0] and batch[0]['points'] is not None:
        # Pad point clouds to same length
        max_points = max(item['points'].shape[0] if item['points'] is not None else 0 
                        for item in batch)
        
        points_list = []
        for item in batch:
            if item['points'] is not None:
                points = item['points']
                if points.shape[0] < max_points:
                    # Pad with zeros
                    padding = torch.zeros(max_points - points.shape[0], points.shape[1])
                    points = torch.cat([points, padding], dim=0)
                points_list.append(points)
            else:
                points_list.append(torch.zeros(max_points, 3))
                
        collated['points'] = torch.stack(points_list)
        
    return collated


if __name__ == "__main__":
    # Test the dataset
    dataset = WaymoDataset(
        data_root="/teamspace/studios/this_studio/bevdet3d/waymo_open_dataset_v_1_4_3",
        split="training",
        num_frames_per_segment=2,  # Small number for testing
        load_point_cloud=False     # Skip point cloud for faster testing
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Images shape: {sample['images'].shape}")
    print(f"GT boxes shape: {sample['gt_bboxes_3d'].shape}")
    print(f"GT labels shape: {sample['gt_labels_3d'].shape}")