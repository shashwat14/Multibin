"""
The weights have been used from the files made available by Davi Frossard
The code is changed almost entirely to make it more modular, cleaner and to allow fine-tuning and visualization in TensorBoard.
The weights are obtained from the following web page : https://www.cs.toronto.edu/~frossard/post/vgg16/
"""

from layers import *
import tensorflow as tf
import numpy as np

class VGG16():

	def __init__(self, sess, inputs, only_convolution=False, num_out=1000):
		'''
		Creates an instance of VGG 16
		Param : tf.Session(), input placeholder, only_convolution
		'''
		self.only_convolution = only_convolution
		self.paramters = []
		self.inputs = inputs
		self.labels = tf.placeholder(tf.float32, shape=[None, num_out])
		self.num_out = num_out
		self.sess = sess
		self.preprocess()
		self.convolution_layers()
		if not self.only_convolution:
			self.fully_connected_layer(num_out)	
			self.belief()
			self.loss()

		#Input visualization
		tf.summary.image('Image', inputs, 3)

	def preprocess(self):
		with tf.name_scope('preprocess') as scope:
			mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1,1,1,3], name='img_mean')
			self.preprocessed = self.inputs - mean
			return self.preprocessed

	def convolution_layers(self):
		'''
		Param: None
		Return: convolution codes after reshaping to [None,7*7*512]
		'''
		#Block 1
		W, b, conv1_1 = conv2d(self.preprocessed, 3, 64, name='conv1_1')
		self.paramters += [W,b]
		W, b, conv1_2 = conv2d(conv1_1, 64, 64, name='conv1_2')
		self.paramters += [W,b]
		pool_1 = max_pool(conv1_2, name='pool_1')

		#Block 2
		W, b, conv2_1 = conv2d(pool_1, 64, 128, name='conv2_1')
		self.paramters += [W,b]
		W, b, conv2_2 = conv2d(conv2_1, 128, 128, name='conv2_2')
		self.paramters += [W,b]
		pool_2 = max_pool(conv2_2, name='pool_2')

		#Block 3
		W, b, conv3_1 = conv2d(pool_2, 128, 256, name='conv3_1')
		self.paramters += [W,b]
		W, b, conv3_2 = conv2d(conv3_1, 256, 256, name='conv3_2')
		self.paramters += [W,b]
		W, b, conv3_3 = conv2d(conv3_2, 256, 256, name='conv3_3')
		self.paramters += [W,b]
		pool_3 = max_pool(conv3_3, name='pool_3')

		#Block 4
		W, b, conv4_1 = conv2d(pool_3, 256, 512, name='conv4_1')
		self.paramters += [W,b]
		W, b, conv4_2 = conv2d(conv4_1, 512, 512, name='conv4_2')
		self.paramters += [W,b]
		W, b, conv4_3 = conv2d(conv4_2, 512, 512, name='conv4_3')
		self.paramters += [W,b]
		pool_4 = max_pool(conv4_3, name='pool_4')

		#Block 5
		W, b, conv5_1 = conv2d(pool_4, 512, 512, name='conv5_1')
		self.paramters += [W,b]
		W, b, conv5_2 = conv2d(conv5_1, 512, 512, name='conv5_2')
		self.paramters += [W,b]
		W, b, conv5_3 = conv2d(conv5_2, 512, 512, name='conv5_3')
		self.paramters += [W,b]
		pool_5 = max_pool(conv5_3, name='pool_5')

		self.convolution_codes = tf.reshape(pool_5, [-1, 7*7*512])

	def fully_connected_layer(self, num_out):
		'''
		Build fully connected module
		Param: None
		Return: Logits
		'''
		W, b, fc1 = fully_connected(self.convolution_codes, 7*7*512, 4096, name='fc1')
		self.paramters += [W,b]
		W, b, fc2 = fully_connected(fc1, 4096, 4096, name='fc2')
		self.paramters += [W,b]

		#customizable output
		W, b, fc3 = fully_connected(fc2, 4096, num_out, name='fc3')
		self.paramters += [W,b]
		self.logits = fc3

	def belief(self):
		'''
		Param: None
		Return: Probability scores
		'''
		with tf.name_scope('probability_scores') as scope:
			self.probs = tf.nn.softmax(self.logits)

	def loss(self):

		with tf.name_scope('xent') as scope:
			self.xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels), name='xent')
			tf.summary.scalar('xent', self.xent)

		with tf.name_scope('train') as scope:
			self.optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(self.xent)
			

	def get_loss(self, X, y):
		return self.sess.run(self.xent, feed_dict={self.inputs:X, self.labels:y})

	def predict(self, X):
		return self.sess.run(self.probs, feed_dict={self.inputs:X})

	def train(self, X, y, i):
		xent, probs,_,s = self.sess.run([self.xent, self.probs, self.optimizer, self.summary], feed_dict={self.inputs:X, self.labels:y})
		self.writer.add_summary(s,i)
		return xent, probs

	def compile(self, path='/'):
		tf.global_variables_initializer().run()
		self.summary = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter(path)
		self.writer.add_graph(self.sess.graph)

	def load_weights(self, path):
		
		weights = np.load(path)
		keys = sorted(weights.keys())
		for i, k in enumerate(keys):

			#Skip intializing fully_connected layers
			if i > 25 and self.only_convolution:
				continue

			#only skip the last fully connected layer when the number of output is not same as 1000. Used for fine-tuning. 
			if i > 29 and self.num_out != 1000:
				continue
			#Load weights to layers
			print i, k, np.shape(weights[k])
			self.sess.run(self.paramters[i].assign(weights[k]))
	