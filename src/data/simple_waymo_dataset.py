import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
import cv2
from typing import Dict, List, Optional, Tuple, Any
import pickle

# Disable TensorFlow logging completely
import warnings
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('FATAL')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class SimpleCachedWaymoDataset(Dataset):
    """
    Simplified cached Waymo dataset that pre-loads and caches data for faster training.
    """
    
    def __init__(self, 
                 data_root: str,
                 split: str = 'training',
                 max_files: int = 3,
                 max_frames_per_file: int = 5,
                 image_size: Tuple[int, int] = (400, 200),
                 cache_dir: str = './cache'):
        """
        Args:
            data_root: Path to waymo dataset root
            split: 'training', 'validation', or 'testing'
            max_files: Maximum number of tfrecord files to load
            max_frames_per_file: Maximum frames per file
            image_size: Target image size (width, height)
            cache_dir: Directory to cache processed data
        """
        self.data_root = data_root
        self.split = split
        self.max_files = max_files
        self.max_frames_per_file = max_frames_per_file
        self.image_size = image_size
        self.cache_dir = cache_dir
        
        os.makedirs(cache_dir, exist_ok=True)
        
        # Cache file path
        cache_file = os.path.join(cache_dir, f'{split}_cached_data.pkl')
        
        if os.path.exists(cache_file):
            print(f"Loading cached data from {cache_file}...")
            with open(cache_file, 'rb') as f:
                self.samples = pickle.load(f)
        else:
            print(f"Creating cache from tfrecord files...")
            self.samples = self._load_and_cache_data()
            with open(cache_file, 'wb') as f:
                pickle.dump(self.samples, f)
            print(f"Cached {len(self.samples)} samples to {cache_file}")
            
    def _get_tfrecord_files(self) -> List[str]:
        """Get list of tfrecord files for the split."""
        split_dir = os.path.join(self.data_root, 'individual_files', self.split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory {split_dir} does not exist")
            
        tfrecord_files = []
        for filename in sorted(os.listdir(split_dir)):
            if filename.endswith('.tfrecord'):
                tfrecord_files.append(os.path.join(split_dir, filename))
                if len(tfrecord_files) >= self.max_files:
                    break
                    
        return tfrecord_files
        
    def _load_and_cache_data(self) -> List[Dict]:
        """Load and cache data from tfrecord files."""
        tfrecord_files = self._get_tfrecord_files()
        samples = []
        
        print(f"Processing {len(tfrecord_files)} tfrecord files...")
        
        for file_idx, tfrecord_path in enumerate(tfrecord_files):
            print(f"  Processing file {file_idx + 1}/{len(tfrecord_files)}: {os.path.basename(tfrecord_path)}")
            
            try:
                dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
                frame_count = 0
                
                for data in dataset:
                    if frame_count >= self.max_frames_per_file:
                        break
                        
                    frame = open_dataset.Frame()
                    frame.ParseFromString(data.numpy())
                    
                    # Process and cache this frame
                    sample = self._process_frame(frame)
                    if sample is not None:
                        samples.append(sample)
                        frame_count += 1
                        
            except Exception as e:
                print(f"    Error processing {tfrecord_path}: {e}")
                continue
                
        return samples
        
    def _process_frame(self, frame: open_dataset.Frame) -> Optional[Dict]:
        """Process a frame and return cached sample data."""
        try:
            # Process images
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
                
                # Store camera name
                camera_info = {
                    'camera_name': open_dataset.CameraName.Name.Name(camera_image.name)
                }
                camera_infos.append(camera_info)
                
            if len(images) == 0:
                return None
                
            images = torch.stack(images, dim=0)  # [N_cams, 3, H, W]
            
            # Process labels
            boxes_3d = []
            labels = []
            
            for laser_label in frame.laser_labels:
                # Use camera-synced box if available, otherwise use regular box
                if laser_label.camera_synced_box.ByteSize():
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
                labels = np.array(labels)
            else:
                boxes_3d = np.zeros((0, 7))
                labels = np.array([])
                
            return {
                'images': images,
                'camera_infos': camera_infos,
                'gt_bboxes_3d': torch.from_numpy(boxes_3d).float(),
                'gt_labels_3d': torch.from_numpy(labels).long(),
                'frame_name': frame.context.name,
                'timestamp': frame.timestamp_micros
            }
            
        except Exception as e:
            print(f"    Error processing frame: {e}")
            return None
            
    def __len__(self) -> int:
        return len(self.samples)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a cached sample."""
        return self.samples[idx]


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate function for DataLoader."""
    collated = {}
    
    # Stack image tensors
    collated['images'] = torch.stack([item['images'] for item in batch])
    
    # Keep lists for variable-length data  
    collated['gt_bboxes_3d'] = [item['gt_bboxes_3d'] for item in batch]
    collated['gt_labels_3d'] = [item['gt_labels_3d'] for item in batch]
    collated['camera_infos'] = [item['camera_infos'] for item in batch]
    collated['frame_name'] = [item['frame_name'] for item in batch]
    collated['timestamp'] = [item['timestamp'] for item in batch]
        
    return collated


if __name__ == "__main__":
    # Test the simplified dataset
    dataset = SimpleCachedWaymoDataset(
        data_root="/teamspace/studios/this_studio/bevdet3d/waymo_open_dataset_v_1_4_3",
        split="training",
        max_files=2,
        max_frames_per_file=3,
        image_size=(400, 200)
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test loading a sample
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Images shape: {sample['images'].shape}")
        print(f"GT boxes shape: {sample['gt_bboxes_3d'].shape}")
        print(f"GT labels shape: {sample['gt_labels_3d'].shape}")
        print("✓ Simplified dataset test passed!")
    else:
        print("✗ No samples loaded")