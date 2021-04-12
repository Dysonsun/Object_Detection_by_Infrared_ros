## Object_Detection_by_Infrared_ros


Robust Object Classification of Occluded Objects in Forward Looking Infrared (FLIR) Cameras using Ultralytics YOLOv3 and Dark Chocolate.And you can run this project in ROS.


#### Instructions

- Must have NVIDIA GPUs with Turing Architecture, Ubuntu and CUDA X installed if you want to reproduce results.

- Add the data provided by FLIR to a folder path called ```/coco/FLIR_Dataset```. 

- Place the custom pre-trained weights you downloaded from above into: ```/weights/*.pt``` 

- Converted labels from [Dark Chocolate](https://github.com/joehoeller/Dark-Chocolate) are located in data/labels, which you unzipped above.

- The custom *.cfg with modified hyperparams is located in ```/scripts/cfg/yolov3-spp-r.cfg```.

- Class names and custom data is in ```/scripts/data/custom.names``` and ```custom.data```.


#### Downloads needed to run codebase
  
   1. Download pre-trained weights here: [link](https://drive.google.com/drive/folders/1dV0OmvG4eZFtnh5WF0mby-jhkVy-HVco?usp=sharing)
   
   2. FLIR Thermal Images Dataset: [Download](https://www.flir.com/oem/adas/adas-dataset-form/)

   3. Go into ```scripts/data``` folder and unzip ```scripts/labels.zip```
   
   4. Addt'l instructions on how to run [Ultralytics Yolov3](https://github.com/ultralytics/yolov3)

#### Requirements
Python 3.5 or later with all requirements.txt dependencies installed
```
cd scripts
pip install -r requirements.txt
```

#### Install & Run Code:

- build messages

```
catkin build
```
- build cv_bridge
  Because ros ```cv_bridge``` don't compatible with ```Python3```, you need to build cv_bridge with python3 in your workspace.

  you can reference this answer 
  ```https://stackoverflow.com/questions/49221565/unable-to-use-cv-bridge-with-ros-kinetic-and-python3 ```



- Run Code
  
Before you run this project, you need edit ```object_detection_by_camera.zsh``` line 8,
```
source ~/software/catkin_workspace/install/setup.zsh --extend
```
to your cv_bridge install path.

and run code

```
sudo chmod +x object_detection_by_camera.zsh
./object_detection_by_camera.zsh
```

#### Result
![avatar](/image/result1.png)



