from model import Multibin
import tensorflow as tf


import sys
sys.path.insert(0, '/media/vision/New Volume/Multibin/')
from data_prep import *

LOGDIR = '/media/vision/New Volume/BoundingBox/'

loss_so_far = 1000000.
db = Dataset()
sess = tf.InteractiveSession()
inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
model = Multibin(sess, orientation_loss_weight=4., dim_loss_weight=1., mode='test')
model.load('/media/vision/New Volume/BoundingBox/model.ckpt')

for i in range(20000):

	X, dims, coss, sins = db.next_val_batch(1)
	print dims, coss, sins
	dims, cos, sin = model.predict(X)
	print dims, cos, sin
	#total_loss, dim_loss, orientation_loss = model.get_loss(X, dims, coss, sins)
	#save model
	#print 'Last Loss : {}, New Loss : {}, Orientation Loss : {}, Dimension Loss : {}'.format(loss_so_far, total_loss, orientation_loss, dim_loss)
	break