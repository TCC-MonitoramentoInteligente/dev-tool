# Development tool

The object detector is [A PyTorch implementation of a YOLO v3 Object Detector](https://github.com/ayooshkathuria/pytorch-yolo-v3). This tool provide graphical interface to help in the algorithms development.

# Requirements
- Ubuntu 16.04 with good GPU
- OpenCV
- Nvidia CUDA
- Python 3

## Setup
1. Install OpenCV library `$ sudo apt-get install libopencv-dev python-opencv`
2. Install Nvidia CUDA-9.0, following this [tutorial](https://yangcha.github.io/CUDA90/) or running the `cuda.sh` script for Ubuntu 16.04
3. Create virtualenv `$ virtualenv --system-site-packages -p python3 venv`
4. Activate virtualenv `$ source venv/bin/activate`
5. Install requirements `$ pip install -r requirements.txt`
6. Download YOLO pre-trained weight file `$ wget https://pjreddie.com/media/files/yolov3.weights -P weights/`

## Testing
### Testing the module
There is a simple script to test if the detector module is working, to verify that the environment has been correctly configured. Run `$ python3 test/detector_test.py --image /path/to/image`. You should see a list with all the detected objects.

## Run
`$ python3 dev_tool.py --video /path/to/video`