'''
Defines different layers like convolution and fully-connected. Also creates a histogram summary for all the weights and biases
'''

import tensorflow as tf

def weight(shape, initializer='truncated_normal', stddev=1e-2):
	'''
	Only truncated normal is implemented
	'''
	initializer = tf.truncated_normal(shape, dtype=tf.float32, stddev=stddev)
	W = tf.Variable(initializer, name = 'W')
	return W

def bias(value, shape):
	initializer = tf.constant(value=value, shape=shape, dtype=tf.float32)
	b = tf.Variable(initializer, trainable=True, name = 'b')
	return b

def conv2d(input_layer, size_in, size_out, ksize=(3,3), strides=[1,1,1,1], padding='SAME', init = 'truncated_normal', stddev = 1e-2, name='conv'):
	'''
	Does a 2D convolution on the input
	'''
	with tf.name_scope(name) as scope:
		shape = [ksize[0], ksize[1], size_in, size_out]
		W = weight(shape, init, stddev)
		b = bias(1.0, [size_out])
		conv = tf.nn.conv2d(input_layer, W, strides, padding)
		out = tf.nn.bias_add(conv, b)
		act = tf.nn.relu(out)
		#Set up visualization histograms
		tf.summary.histogram('weights', W)
		tf.summary.histogram('biases', b)
		tf.summary.histogram('activations', act)
	return W, b, act
	

def fully_connected(input_layer, size_in, size_out, init='truncated_normal', stddev=1e-2, activation='relu', name='fully_connected'):
	'''
	This is a fully-connected layers
	'''
	with tf.name_scope(name) as scope:
		shape = [size_in, size_out]
		W = weight(shape, init, stddev)
		b = bias(1.0, [size_out])
		out = tf.nn.bias_add(tf.matmul(input_layer, W), b)
		if activation == 'relu':
			act = tf.nn.relu(out)
		elif activation == 'linear':
			act = out
		else:
			print "Activation unavailable. Exiting program."
			exit(0)
		#Set up visualizations
		tf.summary.histogram('weights', W)
		tf.summary.histogram('biases', b)
		tf.summary.histogram('activations', act)
	return W, b, act

def max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool'):
	'''
	This layer performs a max pool
	'''
	with tf.name_scope(name) as scope:
		return tf.nn.max_pool(input, ksize, strides, padding, name='pool')