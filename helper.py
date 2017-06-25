# -*- coding: utf-8 -*-
import pickle
import numpy as np
path = '/media/vision/New Volume/Multibin/'
train_path = path + 'training_data/'
train_images_path = train_path + 'images/'
train_labels_path = train_path + 'labels/'
pkl_file = 'kitti_labels.pkl'
xref_pkl_file = 'kitti_xref_data.pkl'
FOCAL_LENGTH = 721.5377
MULTIBIN_NUMBER = 1
def load_labels():
    with open(path + pkl_file, 'rb') as f:
        return pickle.load(f)
        
def load_xref_labels():
	with open(path + xref_pkl_file, 'rb') as f:
		return pickle.load(f)

def getDimsMean():
    data = load_labels()
    length = []
    width= []
    height = []
    for key in data:
        labels = data[key]
        for each in labels:
            cls = each['class']
            dims = each['dims']
            l, w, h = dims
            if l > 0 and w > 0 and h > 0 and cls == 'Car':
                length.append(l)
                width.append(w)
                height.append(h)
    
    length = np.array(length)
    width = np.array(width)
    height = np.array(height)
    
    meanL = np.mean(length)
    meanW = np.mean(width)
    meanH = np.mean(height)
    
    return meanL, meanW, meanH