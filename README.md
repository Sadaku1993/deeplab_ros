# Deeplab for ROS
In this repository, I compiled the source code using ROS and Pyhon2.7

## Overview
[DeepLab: Deep Labelling for Semantic Image Segmentation](https://github.com/tensorflow/models/tree/master/research/deeplab) for ROS

## Requirements
- ROS Kinetic(ubuntu 16.04)
- Python2.7+
- [Opencv](https://opencv.org/)3.3+
- [tensorflow](https://www.tensorflow.org/install/)1.4+
- [DeepLab](https://github.com/tensorflow/models/tree/master/research/deeplab)v3

## How to Install
### DeepLab: Deep Labelling for Semantic Image Segmentation
- [DeepLab](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/installation.md)
### ROS Kinetic
- [ROS Kinetic installation](http://wiki.ros.org/ja/kinetic/Installation/Ubuntu)
### Clone this Repository
```
$ git clone https://github.com/Sadaku1993/deeplab_ros
$ cd catkin_ws
$ catkin make
```

## Download model
See the [TensorFlow DeepLab Model Zoo](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md)
```
$ roscd deeplab_ros
$ cd deeplab
$ wget http://download.tensorflow.org/models/deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz 
```

## How to RUN
### Change model name
```
$ roscd deeplab_ros/deeplab
$ vim deeplab_ros.py
```
**deeplab_ros.py**
```python
    MODEL_NAME = 'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz'
    # MODEL_NAME = 'deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz'
```

### Change param
```
$ roscd deeplab_ros/launch
$ vim deeplab_ros.launch
```
**deeplab_ros.launch**
```
image : Subscribe Topic(sensor_msgs/Image)
deeplab/image : Publish Topic(sensor_msgs/Image)
```
