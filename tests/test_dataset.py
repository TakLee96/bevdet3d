import os
import sys

import numpy as np
import torch
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
import cv2

# Disable TensorFlow logging
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def test_single_frame():
    """Test loading a single frame from a tfrecord file."""
    print("Testing single frame loading...")
    
    # Pick a single tfrecord file
    tfrecord_path = '/mnt/c/Users/jiaha/Developer/bevdet3d/waymo_open_dataset_v_1_4_3/individual_files/training/segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord'
    
    if not os.path.exists(tfrecord_path):
        print(f"File {tfrecord_path} does not exist")
        return
    
    # Load first frame
    dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type='')
    data = next(dataset.as_numpy_iterator())
    frame = open_dataset.Frame()
    frame.ParseFromString(data)
    
    print(f"Frame name: {frame.context.name}")
    print(f"Timestamp: {frame.timestamp_micros}")
    print(f"Number of camera images: {len(frame.images)}")
    print(f"Number of laser labels: {len(frame.laser_labels)}")
    
    # Test image processing
    images = []
    target_size = (800, 320)  # Smaller for testing
    
    for i, camera_image in enumerate(frame.images):
        print(f"Processing camera {i}: {open_dataset.CameraName.Name.Name(camera_image.name)}")
        
        # Decode image
        image_bytes = camera_image.image
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"  Original image shape: {image.shape}")
        
        # Resize image
        image = cv2.resize(image, target_size)
        print(f"  Resized image shape: {image.shape}")
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1)  # HWC -> CHW
        images.append(image_tensor)
        
    images_tensor = torch.stack(images, dim=0)
    print(f"Final images tensor shape: {images_tensor.shape}")
    
    # Test label processing
    boxes_3d = []
    labels = []
    
    for laser_label in frame.laser_labels:
        if laser_label.camera_synced_box.ByteSize():
            box = laser_label.camera_synced_box
        else:
            box = laser_label.box
            
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
        
        print(f"GT boxes shape: {boxes_3d.shape}")
        print(f"GT labels shape: {labels.shape}")
        print(f"Unique labels: {np.unique(labels)}")
    else:
        print("No labels found")
    
    print("Single frame test completed successfully!")
    return True

if __name__ == "__main__":
    success = test_single_frame()
    if success:
        print("\n✅ Dataset test passed!")
    else:
        print("\n❌ Dataset test failed!")