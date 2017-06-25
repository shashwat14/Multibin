# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle
from helper import *
import cv2
import math
from tqdm import tqdm
from random import shuffle
def gen_pickle():
    files = os.listdir(train_labels_path)
    dataset = {}
    for file in files:
        with open(train_labels_path + file, 'r') as f:
            labels = []
            for line in f:
                obj_dict = {}
                obj_class, truncated, occluded, alpha, bx1, by1, bx2, by2, dz, dy, dx, tx, ty, tz, rot_y = line.split()
                obj_dict['class'] = obj_class
                obj_dict['truncated'] = float(truncated)
                obj_dict['occluded'] = float(occluded)
                obj_dict['boxes'] = [int(float(bx1)), int(float(by1)), int(float(bx2)), int(float(by2))]
                obj_dict['alpha'] = float(alpha)
                obj_dict['dims'] = [float(dx), float(dy), float(dz)]
                obj_dict['trans'] = [float(tx), float(ty), float(tz)]
                obj_dict['rot_y'] = float(rot_y)
                labels.append(obj_dict)
        dataset[file] = labels
    with open(path + 'kitti_labels.pkl', 'wb') as f:
        pickle.dump(dataset, f, -1)

def gen_reference_data():
    dataset = load_labels()
    X = []
    label = []
    for file, item in tqdm(dataset.items()):
        for obj in item:
            if obj['class'] == 'Car' and obj['dims'][0]>0 and obj['occluded'] == 0 and obj['truncated'] == 0:
                boxes = obj['boxes']
                dims = obj['dims']
                theta = -obj['rot_y']*180./3.14
                img_path = train_images_path + file[:-4] + '.png'
                img = cv2.imread(img_path)
                rows, cols, _ = img.shape
                x1,y1,x2,y2 = boxes
                x,y = (x1+x2)/2 - cols/2, (y1+y2)/2 - rows/2
                theta_r = 90. - math.atan2(x,FOCAL_LENGTH)*180./3.14
                theta_l = 360 - (theta_r - theta) % 360.
                X.append((img_path, boxes))
                label.append((theta_l,dims))
    print len(X)
    with open(path + 'kitti_xref_data.pkl', 'wb') as f:
        pickle.dump((X,label), f, -1)

train_counter = 0
total_samples = 11017

class Dataset():

    def __init__(self,split=0.95):
        self.lm,self.wm,self.hm = getDimsMean()
        
        #pointers (not c++) for last picked index
        self.train_ptr = 0
        self.val_ptr = 0

        #file paths and their labels
        file_paths, labels = load_xref_labels()

        #data size and splitting size
        db_size = len(file_paths)
        self.train_len = int(db_size*split)
        self.val_len = db_size - self.train_len
        #shuffle this shizz
        c = list(zip(file_paths, labels))
        shuffle(c)
        file_paths, labels = zip(*c)

        #split the data intp train and val
        self.train_x = file_paths[:self.train_len]
        self.train_y = labels[:self.train_len]
        self.val_x = file_paths[self.train_len:]
        self.val_y = labels[self.train_len:]

        #

    def next_train_batch(self, batch_size):
        X = []
        dims = []
        coss = []
        sins = []
        for i in range(batch_size):
            img = self.load_image(self.train_x[self.train_ptr%self.train_len])
            theta, dimens = self.train_y[self.train_ptr%self.train_len]
            theta_l = 180. - theta
            l,w,h = dimens
            dim = np.array([self.lm - l, self.wm-w, self.hm-h])
            cos = math.cos(theta_l*3.14/180.)
            sin = math.sin(theta_l*3.14/180.)
            dims.append(dim)
            coss.append(cos)
            sins.append(sin)
            X.append(img)
            self.train_ptr += 1
        return np.array(X), np.array(dims).reshape(batch_size,3), np.array(coss).reshape(batch_size,1), np.array(sins).reshape(batch_size,1)
        

    def next_val_batch(self, batch_size):
        X = []
        dims = []
        coss = []
        sins = []
        for i in range(batch_size):
            img = self.load_image(self.val_x[self.val_ptr%self.val_len])
            theta, dimens = self.val_y[self.val_ptr%self.val_len]
            theta_l = 180. - theta
            l,w,h = dimens
            dim = np.array([self.lm - l, self.wm-w, self.hm-h])
            cos = math.cos(theta_l*3.14/180.)
            sin = math.sin(theta_l*3.14/180.)
            X.append(img)
            dims.append(dim)
            coss.append(cos)
            sins.append(sin)
            self.val_ptr += 1
        return np.array(X), np.array(dims).reshape(batch_size,3), np.array(coss).reshape(batch_size,1), np.array(sins).reshape(batch_size,1)

    def load_image(self, path):

        path, boxes = path
        img = cv2.imread(path)
        img = img[boxes[1]:boxes[3], boxes[0]:boxes[2],:]
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        return img


def get_data(batch_size):
    global train_counter
    lm,wm,hm = getDimsMean()
    file,labels = load_xref_labels()
    X = []
    xlabels = []
    dims = []
    sins = []
    coss = []
    for i in tqdm(range(batch_size)):
        filepath, boxes = file[train_counter%total_samples]
        img = cv2.imread(filepath)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropped_img = img[boxes[1]:boxes[3], boxes[0]:boxes[2],:]
        cropped_img = cv2.resize(cropped_img, (224,224))
        
        theta, dimens = labels[train_counter%total_samples]
        
        theta_l = 180. - theta
        l,w,h = dimens
        dim = np.array([lm - l, wm-w, hm-h])
        cos = math.cos(theta_l*3.14/180.)
        sin = math.sin(theta_l*3.14/180.)
        try:
            dims.append(dim)
            coss.append(cos)
            sins.append(sin)
            X.append(cropped_img)
            train_counter+=1
            cv2.imwrite('/media/vision/New Volume/BoundingBox/images/' + str(train_counter) + '.png', cropped_img )
            xlabels.append('/media/vision/New Volume/BoundingBox/images/' + str(train_counter) + '.png')
        except:
            continue
    X = np.array(X)
    dims = np.array(dims)
    coss = np.array(coss)
    sins = np.array(sins)
    #train_counter = 0
    return X,dims, coss.reshape(batch_size,1), sins.reshape(batch_size,1)
