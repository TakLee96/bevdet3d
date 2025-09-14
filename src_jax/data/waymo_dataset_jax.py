import os
import numpy as np
import jax.numpy as jnp
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


def numpy_to_jax_batch(batch_data):
    """Convert numpy batch to JAX arrays."""
    jax_batch = {}
    for key, value in batch_data.items():
        if isinstance(value, np.ndarray):
            jax_batch[key] = jnp.array(value)
        elif isinstance(value, list):
            jax_batch[key] = [jnp.array(v) if isinstance(v, np.ndarray) else v for v in value]
        else:
            jax_batch[key] = value
    return jax_batch


class SimpleCachedWaymoDatasetJAX:
    """
    Simplified cached Waymo dataset for JAX/Flax training.
    Converts PyTorch tensors to JAX arrays and uses NHWC format.
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
        
        # Cache file path for JAX version
        cache_file = os.path.join(cache_dir, f'{split}_cached_data_jax.pkl')
        
        if os.path.exists(cache_file):
            print(f"Loading JAX cached data from {cache_file}...")
            with open(cache_file, 'rb') as f:
                self.samples = pickle.load(f)
        else:
            print(f"Creating JAX cache from tfrecord files...")
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
        
        print(f"Processing {len(tfrecord_files)} tfrecord files for JAX...")
        
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
        
    def _process_frame(self, frame) -> Optional[Dict]:
        """Process a single frame and convert to JAX-compatible format."""
        try:
            # Extract camera images (convert to NHWC format for JAX)
            images = []
            camera_names = [
                open_dataset.CameraName.FRONT,
                open_dataset.CameraName.FRONT_LEFT, 
                open_dataset.CameraName.FRONT_RIGHT,
                open_dataset.CameraName.SIDE_LEFT,
                open_dataset.CameraName.SIDE_RIGHT
            ]
            
            for cam_name in camera_names:
                cam_image = None
                for camera in frame.images:
                    if camera.name == cam_name:
                        cam_image = camera
                        break
                        
                if cam_image is None:
                    return None  # Skip if missing cameras
                    
                # Decode image
                img_data = tf.image.decode_jpeg(cam_image.image).numpy()
                
                # Resize to target size (width, height) -> (height, width, channels)
                img_resized = cv2.resize(img_data, self.image_size)
                
                # Normalize to [0, 1] and convert to float32
                img_normalized = img_resized.astype(np.float32) / 255.0
                
                images.append(img_normalized)
            
            # Stack images: [num_cameras, height, width, channels] (NHWC format)
            images = np.stack(images, axis=0)
            
            # Extract 3D bounding boxes
            labels = []
            boxes_3d = []
            
            for laser_label in frame.laser_labels:
                # Get class label (1=Vehicle, 2=Pedestrian, 4=Cyclist)
                label_type = laser_label.type
                if label_type in [1, 2, 4]:  # Only keep these classes
                    # Map to 0-based indexing
                    class_id = 0 if label_type == 1 else (1 if label_type == 2 else 2)
                    labels.append(class_id)
                    
                    # Extract 3D box (center_x, center_y, center_z, length, width, height, heading)
                    box = laser_label.box
                    box_3d = [
                        box.center_x, box.center_y, box.center_z,
                        box.length, box.width, box.height,
                        box.heading
                    ]
                    boxes_3d.append(box_3d)
            
            # Convert to numpy arrays
            labels = np.array(labels, dtype=np.int32)
            boxes_3d = np.array(boxes_3d, dtype=np.float32)
            
            # Pad or truncate to fixed size
            max_objects = 50
            if len(labels) > max_objects:
                labels = labels[:max_objects]
                boxes_3d = boxes_3d[:max_objects]
            elif len(labels) < max_objects:
                # Pad with background class and zero boxes
                pad_size = max_objects - len(labels)
                labels = np.concatenate([labels, np.full(pad_size, 3, dtype=np.int32)])  # 3 = background
                boxes_3d = np.concatenate([boxes_3d, np.zeros((pad_size, 7), dtype=np.float32)])
            
            return {
                'images': images,           # [5, H, W, 3] in NHWC format
                'labels': labels,           # [max_objects]
                'boxes_3d': boxes_3d,       # [max_objects, 7]
                'num_objects': len([l for l in labels if l < 3])  # Exclude background
            }
            
        except Exception as e:
            print(f"    Error processing frame: {e}")
            return None
    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        """Get a sample and return JAX arrays."""
        sample = self.samples[idx]
        
        # Convert to JAX format
        return {
            'images': jnp.array(sample['images']),
            'labels': jnp.array(sample['labels']),
            'boxes_3d': jnp.array(sample['boxes_3d']),
            'num_objects': sample['num_objects']
        }
    
    def get_batch(self, batch_size: int, start_idx: int = 0):
        """Get a batch of samples as JAX arrays."""
        batch_samples = []
        for i in range(batch_size):
            idx = (start_idx + i) % len(self.samples)
            batch_samples.append(self[idx])
        
        # Stack into batch
        batch = {}
        for key in batch_samples[0].keys():
            if key == 'num_objects':
                batch[key] = jnp.array([sample[key] for sample in batch_samples])
            else:
                batch[key] = jnp.stack([sample[key] for sample in batch_samples])
        
        return batch
    
    def data_loader(self, batch_size: int = 2, shuffle: bool = True):
        """Simple data loader iterator for JAX training."""
        indices = np.arange(len(self.samples))
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, len(indices), batch_size):
            end_idx = min(start_idx + batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch samples
            batch_samples = [self[idx] for idx in batch_indices]
            
            # Stack into batch
            batch = {}
            for key in batch_samples[0].keys():
                if key == 'num_objects':
                    batch[key] = jnp.array([sample[key] for sample in batch_samples])
                else:
                    batch[key] = jnp.stack([sample[key] for sample in batch_samples])
            
            yield batch


def create_waymo_dataset_jax(
    data_root: str,
    split: str = 'training',
    max_files: int = 2,
    max_frames_per_file: int = 3,
    image_size: Tuple[int, int] = (400, 200),
    cache_dir: str = './cache'
):
    """Factory function to create JAX Waymo dataset."""
    return SimpleCachedWaymoDatasetJAX(
        data_root=data_root,
        split=split,
        max_files=max_files,
        max_frames_per_file=max_frames_per_file,
        image_size=image_size,
        cache_dir=cache_dir
    )


if __name__ == "__main__":
    # Test the JAX dataset
    dataset = create_waymo_dataset_jax(
        data_root="waymo_open_dataset_v_1_4_3",
        split="training",
        max_files=1,
        max_frames_per_file=2,
        image_size=(400, 200)
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test single sample
    sample = dataset[0]
    print(f"Sample shapes:")
    print(f"  Images: {sample['images'].shape}")  # Should be [5, 200, 400, 3]
    print(f"  Labels: {sample['labels'].shape}")
    print(f"  Boxes 3D: {sample['boxes_3d'].shape}")
    print(f"  Num objects: {sample['num_objects']}")
    
    # Test batch loading
    batch = dataset.get_batch(batch_size=2)
    print(f"\nBatch shapes:")
    print(f"  Images: {batch['images'].shape}")  # Should be [2, 5, 200, 400, 3]
    print(f"  Labels: {batch['labels'].shape}")
    print(f"  Boxes 3D: {batch['boxes_3d'].shape}")
    print(f"  Num objects: {batch['num_objects'].shape}")
    
    # Test data loader
    print(f"\nTesting data loader...")
    for i, batch in enumerate(dataset.data_loader(batch_size=2)):
        print(f"  Batch {i}: Images {batch['images'].shape}, Labels {batch['labels'].shape}")
        if i >= 2:  # Only test a few batches
            break
    
    print("✓ JAX dataset test passed!")