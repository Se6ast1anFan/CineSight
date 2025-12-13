## Installation

1. Before starting, please obtain your Doubao AI **Access Key**, **Secret Access Key**, and **ENDPOINT_ID**, and create a `.env` file with the following content in the same folder:

   ```
   VOLC_ACCESSKEY= your Access Key
   VOLC_SECRETKEY= your Secret Access Key
   ENDPOINT_ID= your ENDPOINT_ID
   ```

2. create a new python environment (optional):

   ```bash
   conda create -n {your_env_name, e.g., db} python=3.7.11
   conda activate {your_env_name, e.g., db}
   ```

3. install requiments:

   ```bash
   pip install -r requirements.txt
   ```

4. install torch (for NVIDIA GeForce RTX 3050 and more advanced):

   ```bash
   pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
   ```

5. download weights file(ckpt.t7) from [[deepsort]](https://drive.google.com/drive/folders/1xhG0kRH1EX5B9_Iz8gQJb7UNnn_riXi6) to this folder:

   ```
   ./deep_sort/deep_sort/deep/checkpoint/
   ```

6. test on your video:

   ```bash
   set KMP_DUPLICATE_LIB_OK=TRUE
   python yolo_slowfast.py --input demo.mp4 --output demo_out.wav --device cuda
   ```

   The first time execute this command may take some times to download the yolov5 code and it's weights file from torch.hub, keep your network connection. For subsequent use, the system can be launched through the GUI by clicking run_gui.bat in the folder.


## References

[1] [Ultralytics/Yolov5](https://github.com/ultralytics/yolov5)

[2] [ZQPei/deepsort](https://github.com/ZQPei/deep_sort_pytorch) 

[3] [FAIR/PytorchVideo](https://github.com/facebookresearch/pytorchvideo)

[4] AVA: A Video Dataset of Spatio-temporally Localized Atomic Visual Actions. [paper](https://arxiv.org/pdf/1705.08421.pdf)

[5] SlowFast Networks for Video Recognition. [paper](https://arxiv.org/pdf/1812.03982.pdf)
