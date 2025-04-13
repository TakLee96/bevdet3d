# bevdet3d
Goal: Reproduce PETR / BEVFormer / BEVFusion on WOD in plain PyTorch

## Waymo Open Dataset

Register your account on [WOMD](https://waymo.com/open/licensing/?continue=%2Fopen%2Fdownload%2F) website

Follow this [guide](https://cloud.google.com/sdk/docs/install) to install gcloud cli

If you want to download the dataset, run the following:
```
gcloud storage cp --recursive gs://waymo_open_dataset_v_1_4_3/ .
gcloud storage cp --recursive gs://waymo_open_dataset_end_to_end_camera_v_1_0_0/ .
gcloud storage cp --recursive gs://waymo_open_dataset_v_2_0_1/ .
gcloud storage cp --recursive gs://waymo_open_dataset_motion_v_1_3_0/ .
```

Alternatively, you can follow this [link](https://cloud.google.com/storage/docs/cloud-storage-fuse/quickstart-mount-bucket) to setup gcloud fuse
```
gcsfuse waymo_open_dataset_v_1_4_3 $(pwd)/waymo_open_dataset_v_1_4_3
```

You might need to force install older version of protobuf if you encounter this error
```
TypeError: expected bytes, bytearray found

pip install -U protobuf==3.20.1
```

Follow `tutorial_*.ipynb` notebooks to learn how to load data from WOMD.

Here we focus on waymo_open_dataset_v_1_4_3 for 3D object detection using camera + lidar.
