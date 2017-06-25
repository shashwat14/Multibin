from model import Multibin
import tensorflow as tf


import sys
sys.path.insert(0, '/media/vision/New Volume/Multibin/')
from data_prep import *

LOGDIR = '/media/vision/New Volume/BoundingBox/'

#use block below for hyperparameter search 

'''
hpar = [
(1,1),
(0.5,1),
(0.1, 1),
(0.01, 1),
(1, 0.5),
(1, 0.1),
(1, 0.01),
(4, 1),
(8, 2),
(1, 0.2),
]

for ow, dw in hpar:
	tf.reset_default_graph()
	sess = tf.InteractiveSession()
	inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
	model = Multibin(sess, orientation_loss_weight=ow, dim_loss_weight=dw,  weights_path='/media/vision/New Volume/Multibin/vgg16_weights.npz')
	model.compile(LOGDIR + model.get_namespace())
	for i in range(1000):
		X, dims, coss, sins = get_data(8)
		model.train(X, dims, coss, sins)
'''
loss_so_far = 1000000.
db = Dataset()
sess = tf.InteractiveSession()
inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
model = Multibin(sess, orientation_loss_weight=4., dim_loss_weight=1.,  weights_path='/media/vision/New Volume/Multibin/vgg16_weights.npz')
model.compile(LOGDIR + model.get_namespace())
for i in range(20000):
	X, dims, coss, sins = db.next_train_batch(8)
	model.train(X, dims, coss, sins)

	#run validation
	if i % 5 == 0:
		X, dims, coss, sins = db.next_val_batch(32)
		total_loss, dim_loss, orientation_loss = model.get_loss(X, dims, coss, sins)
		if total_loss < loss_so_far:

			#save model
			print 'Last Loss : {}, New Loss : {}, Orientation Loss : {}, Dimension Loss : {}'.format(loss_so_far, total_loss, orientation_loss, dim_loss)
			model.save('/media/vision/New Volume/BoundingBox/model.ckpt')
			loss_so_far = total_loss
