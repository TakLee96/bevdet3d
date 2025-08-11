# Train PETR and BEVFormer on Waymo Open Dataset

Here are the goals of this project:
1. Implement PETR and BEVFormer in pure pytorch (do not use mmdet3d, but ok to use einops)
2. Implement Data Loader for Waymo Open Dataset (tfrecord)
3. Train PETR and BEVFormer and obtain metrics on Waymo Open Dataset

Reference:
1. Read submodule `PETR` for mmdet3d implementation of PETR
2. Read submodule `BEVFormer` for mmdet3d implementation of BEVFormer
3. Read CLAUDE.md and README.md for overall project context
4. Read *.ipynb for Waymo Open Dataset how-to
5. If you have to, read submodule waymo-open-dataset for more details on dataset usage

Please remember to:
1. Breakdown overall goal into individual TODOs and write them down into README.md
2. Whenever you made progress, add unit test, document progress in README.md and make git commit
3. If you wish to install additional packages (e.g. einops), remember to update requirements.txt

You're currently running inside windows git bash on a local laptop with Nvidia 4060 GPU (python/tensorflow/pytorch is available and already installed in conda environment `tensorflow`; feel free to `pip list` or `python --version` or `nvidia-smi` to check what's available). Only `waymo_open_dataset_v_1_4_3/individual_files/validation` is downloaded (for now, only use validation data; if the code is working, I will migrate this project to a linux server and download the entire dataset there).
