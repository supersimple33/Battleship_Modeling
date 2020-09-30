import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Flatten, Dense, Reshape, GlobalMaxPool1D, Reshape, LocallyConnected1D, Add, Activation, Conv2D, BatchNormalization, LeakyReLU, MaxPool2D, AveragePooling2D, Concatenate
import tensorflow.keras.backend as K

import kerastuner as kt

@tf.function
def customAccuracy(y_true, y_pred):
	trans = tf.transpose([y_true, y_pred],perm=[1,0,2])
	accur = 0 
	i = 0
	for elem in trans:
		move = tf.argmax(elem[1],-1)
		v = elem[0][move] #which is first?
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

def buildModel0(hp):
	# numFilters = hp.Int(name="num_filter", min_value=16, max_value=128, step=16, default=64)
	# activation = hp.Choice(["relu, selu"],name='activation')
	numFilters = 48
	activation = 'relu'


	inputLay = tf.keras.Input(shape=(6,10,10))

	c1 = Conv2D(filters=numFilters,kernel_size=1,activation=activation)(inputLay) # locally connected instead?

	c3 = Conv2D(filters=numFilters, kernel_size=1, activation=activation)(inputLay)
	c3 = Conv2D(filters=numFilters, kernel_size=3, activation=activation)(c3)

	c5 = Conv2D(filters=numFilters, kernel_size=1, activation=activation)(inputLay)
	c5 = Conv2D(filters=numFilters, kernel_size=5, activation=activation)(c3)

	ap = AveragePooling2D(3, 1)(inputLay)
	ap = Conv2D(filters=numFilters, kernel_size=1, activation=activation)(ap)

	conc = Concatenate(axis=1)([c1,c3,c5,ap])
	c0 = Conv2D(filters=32,kernelSize=3, activation=activation)(conc)

	f = Flatten()(c0)
	out = Dense(100, activation='sigmoid')(f)
	return tf.keras.Model(inputs=inputLay, outputs=out)

def buildModel(hp):
	m = tf.keras.Sequential()
	m.add(InputLayer(input_shape=(10,10,6)))
	m.add(Flatten())
	for i in range(hp.Int('num_layers', 1, 5)):
		m.add(Dense(units=hp.Int('units_' + str(i),
					min_value=32,
					max_value=512,
					step=32),
					activation='relu'))
	m.add(Dense(100, activation='sigmoid'))
	return m

def buildModel1():
	inputLay = tf.keras.Input(shape=(10,10,6))
	f = Flatten()(inputLay)
	d1 = Dense(150,activation='relu')(f)
	d2 = Dense(100)(d1)

	r1 = Reshape((100,6))(inputLay)
	a = GlobalMaxPool1D('channels_first')(r1)
	r2 = Reshape((100,1))(a)
	lc = LocallyConnected1D(1,1)(r2)
	f2 = Flatten()(lc)
	# gm1 = GlobalMaxPool2D('channels_first')(inputLay)
	# gm2 = GlobalMaxPool1D('channels_first')(gm1)

	a = Add()([d2,f2])
	out = Activation('sigmoid')(a)
	return tf.keras.Model(inputs=inputLay, outputs=out)

def convLayerCluster(inp, filters, axis, ch):
	m = Conv2D(filters=filters,kernel_size=(3,3),padding="same",use_bias=False,activation='linear',kernel_regularizer=reg,data_format=ch)(inp)
	m = BatchNormalization(axis=axis)(m)
	return LeakyReLU()(m)
def residualLayerCluster(inp, filters, axis, ch):
	m = convLayerCluster(inp, filters, axis, ch)
	m = Conv2D(filters=filters,kernel_size=(3,3),padding="same",use_bias=False,activation='linear',kernel_regularizer=reg,data_format=ch)(m)
	m = BatchNormalization(axis=axis)(m)
	m = Add()([inp,m])
	return LeakyReLU()(m)
reg = None # see if no reg helps # reg = tf.keras.regularizers.L2(l2=0.0001)
def buildModel2(filters, axis, ch):
	inputLay = tf.keras.Input(shape=(10,10,6))#12

	m = Conv2D(filters=filters,kernel_size=(3,3),padding="same",use_bias=False,activation='linear',kernel_regularizer=reg,data_format=ch)(inputLay)
	# m = Conv2D(64,3,data_format=CHANNEL_TYPE)(inputLay)
	m = BatchNormalization(axis=axis)(m) #-1 for channels last
	m = LeakyReLU()(m)
	m = residualLayerCluster(m, filters, axis, ch)
	# m = residualLayerCluster(m) # removed on cluster for added simplricity

	m = Conv2D(filters=6,kernel_size=(1,1),padding="same",use_bias=False,activation='linear',kernel_regularizer=reg,data_format=ch)(m) #12
	m = BatchNormalization(axis=axis)(m)
	m = LeakyReLU()(m)
	m = Flatten()(m)
	out = Dense(100,activation='sigmoid')(m) #
	return tf.keras.Model(inputs=inputLay, outputs=out)