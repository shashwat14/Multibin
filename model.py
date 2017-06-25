import tensorflow as tf
from layers import conv2d, fully_connected
from vgg import VGG16

class Multibin():

	def __init__(self, sess, learning_rate=1e-4, stddev=0.01, n_bins=1, overlap = 30., orientation_loss_weight = 8.,  dim_loss_weight = 4., weights_path='/', mode='train'):

		'''
		Input:
		Learning Rate
		stddev : Standard Deviation
		n_bins : Number of bins for orientation prediction
		overlap : Overlap of bins in degrees for the orientation prediction
		orientation_loss_weight : weight of the orientation loss in total loss
		dim_loss_weight : weight of the dimension loss in total loss
		weights_path : Pretrained VGG weights_path
		'''
		self.sess = sess

		#set input and output
		self.inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input')
		self.sin_placeholder = tf.placeholder(tf.float32, shape=[None, 1], name='output_sin')
		self.cos_placeholder = tf.placeholder(tf.float32, shape=[None, 1], name='output_cos')
		self.dim_placeholder = tf.placeholder(tf.float32, shape=[None, 3], name='output_dim')
		#Set hyperparameters
		self.set_hparams(learning_rate, stddev, n_bins, overlap, orientation_loss_weight, dim_loss_weight)
		#Set the VGG network with just convolution layers
		self.vgg = VGG16(self.sess, self.inputs, only_convolution=True)
		self.convolution_codes = self.vgg.convolution_codes
		#Initialize new fully connected module - 3D Module
		self.fully_connected_layer()
		self.set_losses()
		#Load weights to only convolution network - [VGG class automatically loads only convolution if only_convolution = True]
		tf.global_variables_initializer().run()
		if mode == 'train':
			self.vgg.load_weights(weights_path)
		self.saver = tf.train.Saver()

		self.iteration = 0
	def set_hparams(self, *args):
		'''
		Not for direct usage
		'''
		self.hparams = {}
		self.hparams['learning_rate'] = args[0]
		self.hparams['stddev'] = args[1]
		self.hparams['n_bins'] = args[2]
		self.hparams['overlap'] = args[3]
		self.hparams['orientation_loss_weight'] = args[4]
		self.hparams['dim_loss_weight'] = args[5]

		str_builder = ''
		for key, value in self.hparams.items():
			str_builder += key + '_' + str(value) + '_'
		self.namespace = str_builder
		
	def get_hparams(self):
		'''
		Inputs: None
		Return: Hyperparameter dictionary
		'''
		return self.hparams

	def get_namespace(self):
		return self.namespace

	def fully_connected_layer(self):
		'''
		Sets the custom 3D module. 
		Input: None

		'''
		#Dimensions
		W, b, dim_h = fully_connected(self.convolution_codes, 7*7*512, 512, name='dim_h')
		W, b, self.dim = fully_connected(dim_h, 512, 3, name='dim_out', activation='linear')

		
		#Orientation
		W, b, orientation_h = fully_connected(self.convolution_codes, 7*7*512, 256, name='orientation_h')
		W, b, sin_u = fully_connected(orientation_h, 256, 1, name='unnormalized_sin', activation='linear')
		W, b, cos_u = fully_connected(orientation_h, 256, 1, name='unnormalized_cos', activation='linear')

		l2norm = tf.sqrt(tf.square(sin_u) + tf.square(cos_u))
		self.sin = tf.divide(sin_u, l2norm, name='sin_out')
		self.cos = tf.divide(cos_u, l2norm, name='cos_out')

	def orientation_loss(self):
		with tf.name_scope('orienation_loss') as scope:
			self.orientation_loss_value = tf.reduce_mean(tf.square(self.sin - self.sin_placeholder) + tf.square(self.cos - self.cos_placeholder))
			tf.summary.scalar('orientation_loss', self.orientation_loss_value)

	def dim_loss(self):
		with tf.name_scope('dim_loss') as scope:
			self.dim_loss_value = tf.reduce_mean(tf.square(self.dim - self.dim_placeholder))
			tf.summary.scalar('dim_loss', self.dim_loss_value)			

	def loss(self):
		with tf.name_scope('total_loss') as scope:
			self.total_loss_value = self.hparams['orientation_loss_weight']*self.orientation_loss_value + self.hparams['dim_loss_weight']*self.dim_loss_value
			tf.summary.scalar('total_loss_value', self.total_loss_value)

	def predict(self, X):
		return self.sess.run([self.dim, self.cos, self.sin], feed_dict={self.inputs:X})

	def get_dim_loss(self, X, dim, cos, sin):
		return self.sess.run(self.dim_loss_value, feed_dict={self.inputs:X, self.sin_placeholder:sin, self.cos_placeholder:cos, self.dim_placeholder:dim})

	def get_orientation_loss(self, X, dim, cos, sin):
		return self.sess.run(self.orientation_loss_value, feed_dict={self.inputs:X, self.sin_placeholder:sin, self.cos_placeholder:cos, self.dim_placeholder:dim})

	def get_loss(self, X, dim, cos, sin):
		return self.sess.run([self.total_loss_value, self.dim_loss_value, self.orientation_loss_value], feed_dict={self.inputs:X, self.sin_placeholder:sin, self.cos_placeholder:cos, self.dim_placeholder:dim})

	def set_losses(self):
		self.orientation_loss()
		self.dim_loss()
		self.loss()
		
	def train(self, X, dim, cos, sin):
		self.iteration+=1
		total_loss_value, orientation_loss_value, dim_loss_value, _, summary = self.sess.run([self.total_loss_value, self.orientation_loss_value, self.dim_loss_value, self.optimizer, self.summary], feed_dict={self.inputs:X, self.sin_placeholder:sin, self.cos_placeholder:cos, self.dim_placeholder:dim})
		self.writer.add_summary(summary,self.iteration)
		return total_loss_value, orientation_loss_value, dim_loss_value

	def compile(self, path='/'):
		with tf.name_scope('train') as scope:
			self.optimizer = tf.train.GradientDescentOptimizer(self.hparams['learning_rate']).minimize(self.total_loss_value)

		self.summary = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter(path)
		self.writer.add_graph(self.sess.graph)

	def save(self,path):
		self.saver.save(self.sess, path)
		print 'Saved Model at  : {}'.format(path)

	def load(self, path):
		self.saver.restore(self.sess, path)
		print 'Model loaded successfully'