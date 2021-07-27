import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Reshape, GlobalMaxPool1D, Reshape, LocallyConnected2D, Add, Activation, Conv2D, BatchNormalization, LeakyReLU, MaxPool2D, AveragePooling2D, Concatenate, Lambda, LayerNormalization, SeparableConv2D
import tensorflow.keras.backend as K

from tensorflow.python.ops import array_ops

# import kerastuner as kt

AXIS = 1 if K.image_data_format() == "channels_first" else -1

class TransformConstaint(tf.keras.constraints.Constraint):

	def __call__(self, w):
		w_shape = w.shape
		if w_shape.rank is None or w_shape.rank != 4:
			raise ValueError('The weight tensor must be of rank 4, but is of shape: %s' % w_shape)
		
		height, width, channels, kernels = w_shape
		w = K.reshape(w, (height, width, channels * kernels))

		# Map the constraint on the kernel
		w = K.stack(array_ops.unstack(w, axis=-1), axis=0)
		w = tf.map_fn(self._kernel_constraint, w)
		return K.reshape(K.stack(array_ops.unstack(w, axis=0), axis=-1), (height, width, channels, kernels))

	def _kernel_constraint(self, kernel):
		kernel_shape = K.shape(kernel)[0] # assumes square tensor

		# Copy all the values in the diagonals
		i = tf.constant(0)
		while_condition = lambda i, _: tf.less(i, kernel_shape // 2)
		def diag(i, array):
			selection = array[i, i]
			indexes = [[i, (kernel_shape - 1) - i], [(kernel_shape - 1) - i, i], [(kernel_shape - 1) - i, (kernel_shape - 1) - i]]

			moddedArr = tf.tensor_scatter_nd_update(array, indexes, tf.fill([3], selection))
			
			return [i + 1, moddedArr]

		_, kernel = tf.while_loop(while_condition, diag, [i, kernel])

		# Copy all the values in along the middle axes
		def cross(i, array):
			selection = array[i, kernel_shape // 2]
			indexes = [[kernel_shape // 2, i], [kernel_shape // 2, (kernel_shape - 1) - i], [(kernel_shape - 1) - i, kernel_shape // 2]]

			moddedArr = tf.tensor_scatter_nd_update(array, indexes, tf.fill([3], selection))

			return [i + 1, moddedArr]

		_, kernel = tf.while_loop(while_condition, cross, [i, kernel])

		# Nesting these loops probably does not help in terms of overhead time, however its more intuitive
		i = tf.constant(1)
		def outer(i, array): # move up the kernel
			j = tf.constant(0) # might be faster to just copy this?
			inner_condition = lambda j, _: tf.less(j, i)
			def inner(j, innerArr): # move in the kernel
				selection = innerArr[(kernel_shape // 2) - i - 1, (kernel_shape // 2) - j - 1]
				indexes = [[(kernel_shape // 2) - j - 1, (kernel_shape // 2) - i - 1], [(kernel_shape // 2) + j + 1, (kernel_shape // 2) - i - 1], [(kernel_shape // 2) + i + 1, (kernel_shape // 2) - j - 1], [(kernel_shape // 2) + i + 1, (kernel_shape // 2) + j + 1], [(kernel_shape // 2) + j + 1, (kernel_shape // 2) + i + 1], [(kernel_shape // 2) - j - 1, (kernel_shape // 2) + i + 1], [(kernel_shape // 2) - i - 1, (kernel_shape // 2) + j + 1]]
				
				retArr = tf.tensor_scatter_nd_update(innerArr, indexes, tf.fill([7], selection))
				return [j + 1, retArr]
			_, moddedArr = tf.while_loop(inner_condition, inner, [j, array])
			return [i + 1, moddedArr]

		_, kernel = tf.while_loop(while_condition, outer, [i, kernel])

		return kernel # return necessary?

@tf.function
def customAccuracy(y_true, y_pred):
	trans = tf.transpose([y_true, y_pred],perm=[1,0,2])
	accur = 0 
	i = 0
	for elem in trans:
		move = tf.argmax(elem[1],-1)
		v = elem[0][move]
		accur += 1 if v == 1 else 0
		i += 1
	return accur / i

@tf.function
def customLoss(y_true, y_pred):
	crossEnt = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
	return tf.reduce_mean(crossEnt)

@tf.function
def f1_loss(y_true, y_pred):
	tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
	tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
	fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
	fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

	p = tp / (tp + fp + K.epsilon())
	r = tp / (tp + fn + K.epsilon())

	f1 = 2*p*r / (p+r+K.epsilon())
	f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
	return 1 - K.mean(f1)

def buildModel():
	reg1 = tf.keras.regularizers.L1L2(0.0001,0.0)
	reg2 = tf.keras.regularizers.L1L2(0.0,0.0001)
	reg1 = None

	constraint = TransformConstaint()

	numFilters = 16
	activation = 'selu'

	inputLay = tf.keras.Input(shape=(2,10,10) if AXIS == 1 else (10,10,2))

	c1 = Conv2D(filters=numFilters // 2,kernel_size=1,activation=activation, padding='same')(inputLay) # DESTORY/REPLACE

	c3 = Conv2D(filters=numFilters // 2, kernel_size=3, activation=activation, padding='same', kernel_regularizer=reg1, kernel_constraint=constraint)(inputLay)

	c5 = Conv2D(filters=numFilters, kernel_size=5, activation=activation, padding='same', kernel_regularizer=reg1)(inputLay) # separable

	c7 = Conv2D(filters=numFilters // 2, kernel_size=3, activation=activation, padding='same', dilation_rate=3, kernel_regularizer=reg1, kernel_constraint=constraint)(inputLay) # Separable

	ap = AveragePooling2D(3, 1, padding='same')(inputLay) # what should we look at misses?
	ap = Conv2D(filters=numFilters // 2, kernel_size=3, activation=activation, padding='same', kernel_regularizer=reg1)(ap)

	conc0 = Concatenate(axis=AXIS)([c1,c3,c5,c7,ap])
	c0 = Conv2D(filters=32, kernel_size=5, activation=activation, padding='same', kernel_regularizer=reg2)(conc0)# bump up

	sums = Lambda(lambda x: tf.math.count_nonzero(x, axis=AXIS, keepdims=True, dtype=tf.float32))(inputLay)
	conc1 = Concatenate(axis=AXIS)([c0,sums])

	fc = LocallyConnected2D(1,19, activation='sigmoid', padding='same', use_bias=True, implementation=2, kernel_regularizer=reg2)(conc1) # no slower than regular 3 sigmoid

# 	superConv = Conv2D(1, 19, activation='sigmoid', padding='same', use_bias=True, kernel_regularizer=reg2)(conc1)

	out = Flatten()(fc) #fc
	return tf.keras.Model(inputs=inputLay, outputs=out)