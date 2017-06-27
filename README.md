# Multibin

This is a partial implementation of the paper : Mousavian, Arsalan, et al. "3D Bounding Box Estimation Using Deep Learning and Geometry." arXiv preprint arXiv:1612.00496 (2016).

The aim of this project is to predict the size of the bounding box and orientation of the object in 3D space from a single two dimensional image. The paper implements a 3D location estimation algorithm as well which we haven't yet implemented. Although, it'll be a good addition. Moreover, we consider only one bin for orientation while the paper suggests two. 

## Prerequisites
1. TensorFlow 1.0
2. Numpy
3. OpenCV 2
4. Python 2.7
5. tqdm

## Installation
1. Git clone this project : git clone https://github.com/shashwat14/Multibin.git
2. Download the KITTI object detection dataset and save it within the Multibin folder or save it somewhere else and make changes as mentioned in point 3.
3. Open helper.py and edit the following paths and make sure the path names are correct : 
  path = /path/to/Multibin/
  train_path = path + 'training_data/'
  train_images_path = train_path + 'images_1/'
  train_labels_path = train_path + 'labels_1/'
4. Download the weights file (vgg16_weights.npz) and place in Multibin directory. https://www.cs.toronto.edu/~frossard/vgg16/vgg16_weights.npz
5. python main.py
6. python train.py
7. python test.py
