# Video Pose Estimation

Video pose estimation based on yolox and dwpose.

# Installation
1. Download and install Miniconda.
2. Create a conda environment:
    ```
    conda create --name dwpose python=3.8 -y
    conda activate dwpose
    ```
3. Install pytorch: https://pytorch.org/get-started/locally/
4. Install MMEngine, MMCV, MMDetection, and MMPose:
    ```
    pip install -U openmim
    mim install mmengine
    mim install "mmcv>=2.0.1"
    mim install "mmdet>=3.1.0"
    mim install "mmpose>=1.1.0"
    ```
5. Go to "/miniconda3/envs/fome/lib/python3.8/site-packages/mmdet/\_\_init__.py". 
   Change `mmcv_maximum_version = '2.2.0'` to `mmcv_maximum_version = '2.2.1'`

# Model Weights Downloading
Yolox: https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth

DWPose: https://huggingface.co/yzd-v/DWPose/blob/main/dw-ll_ucoco_384.pth

Once downloaded, move yolox weight to yolox_config folder and move dwpose weight to dwpose_config folder.


# Function Details
    video_pose_estimation(video_path, save_to_video_path=None, save_to_images_path=None,
                          batch_size_1=100, batch_size_2=150,
                          extraction_ratio=0, max_length=0):

    :param video_path: the path to your video
    :param save_to_video_path: the path to save the estimated video
    :param save_to_images_path: the path to save the estimated video in images
    :param batch_size_1: batch size for human detection
    :param batch_size_2: batch size for pose estimation
    :param extraction_ratio: an integer number x representing the extraction ratio.
    Positive: extract 1 frame for each x frames.
    Negative: extract x frames for each frame.
    Can not be 1. Default is 0.
    :param max_length: limit the maximum video length, unit: seconds
    :return: pose sequences, bounding box list, video_meta


## Pose Sequence

- Shape: a list with dimension of (N, 406), N is the number of frames, and 406 = 7 + 133 * 3
- First 7 elements: [0, 0, x1, y1, x2, y2, c]. (x1, y1) is the 
  top left corner. (x2, y2) is bottom right corner. c is the confidence
  score of the bounding box for current frame.
- For the other element: [x1, y1, c1, x2, y2, c2, ...] for each key point, it has x coordinate,
  y coordinate and its confidence score. 
- For the details of key point name and position, please refer to https://github.com/jin-s13/COCO-WholeBody?tab=readme-ov-file#what-is-coco-wholebody
- For xth key point, its corresponding element indexes are (3x+4, 3x+5, 3x+6)


## Bounding Box 

- Shape: a list with dimension of (N, 1, 4)
- For dimension (1, 4): [[x1, y1, x2, y2]]. (x1, y1) is the 
  top left corner. (x2, y2) is bottom right corner
- Note, the coordinate of bbox is absolute coordinate while
  the coordinate of key points is a ratio of the whole frame.
- x_bbox should compare with W * x_kpt
- y_bbox should compare with H * y_kpt


## Video Meta

- Metadata for provided video.
- 'fps': video fps
- 'frame': video frames
- 'H': video height
- 'W': video width
- 'channel': frame channel
- 'seconds': video length in seconds


## Recommended batch size
For A100 with 80G memory, the batch size is:

    batch_size_1 = 600
    batch_size_2 = 1220

For RTX4090 with 24G memory, the batch size is:

    batch_size_1 = 185
    batch_size_2 = 375

For T4 with 16G memory, the batch size is:

    batch_size_1 = 115
    batch_size_2 = 240