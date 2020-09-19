# builtins.__dict__.update(locals())
# t = timeit.Timer('conved = list(map(lambda x: list(map(lambda y: y.value[0], x)), prevObs))')
# time = t.timeit(10000)
# print(time / 10000)
import faulthandler
import multiprocessing
# logger = open('log.txt', 'w')
# faulthandler.enable(file=logger)

import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv2D, BatchNormalization, LeakyReLU, Add, Flatten, Dense, Concatenate, Dot, Reshape, Dropout
import tensorflow.keras.backend as K
print(tf.__version__)

import tensorflow.keras.backend as K
# tf.autograph.set_verbosity(10,alsologtostdout=True)

import numpy as np

import gym
import gym_battleship1

import builtins
# import timeit # DEBUG Only
import time
# from collections import deques

print(tf.__version__)

# MODEL TWEAKS
NUM_GAMES = 60
FILTERS = 64 # 64 because its cool
EPSILON = 1.0 # Epsilon must start close to one or model training will scew incredibelly
LEARNING_RATE = 0.001
MOMENTUM = 0.05
CHANNEL_TYPE = "channels_last"
AXIS = 1 if CHANNEL_TYPE == "channels_first" else -1

def convLayerCluster(inp):
	m = Conv2D(filters=FILTERS,kernel_size=(3,3),padding="same",use_bias=False,activation='linear',kernel_regularizer=reg,data_format=CHANNEL_TYPE)(inp)
	m = BatchNormalization(axis=AXIS)(m)
	return LeakyReLU()(m)

def residualLayerCluster(inp):
	m = convLayerCluster(inp)
	m = Conv2D(filters=FILTERS,kernel_size=(3,3),padding="same",use_bias=False,activation='linear',kernel_regularizer=reg,data_format=CHANNEL_TYPE)(m)
	m = BatchNormalization(axis=AXIS)(m)
	m = Add()([inp,m])
	return LeakyReLU()(m)

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

# def softmax_cross_entropy_with_logits(y_true, y_pred):


#BUILDING MODEL
# reg = tf.keras.regularizers.L2(l2=0.0001)
reg = None # see if no reg helps 
def buildModel():
	inputLay = tf.keras.Input(shape=(10,10,6))#12

	m = Conv2D(filters=FILTERS,kernel_size=(3,3),padding="same",use_bias=False,activation='linear',kernel_regularizer=reg,data_format=CHANNEL_TYPE)(inputLay)
	# m = Conv2D(64,3,data_format=CHANNEL_TYPE)(inputLay)
	m = BatchNormalization(axis=AXIS)(m) #-1 for channels last
	m = LeakyReLU()(m)

	m = residualLayerCluster(m)
	# m = residualLayerCluster(m)
	# m = residualLayerCluster(m) # removed on cluster for added simplricity

	m = Conv2D(filters=6,kernel_size=(1,1),padding="same",use_bias=False,activation='linear',kernel_regularizer=reg,data_format=CHANNEL_TYPE)(m) #12
	m = BatchNormalization(axis=AXIS)(m)
	m = LeakyReLU()(m)

	m = Flatten()(m)
	# og = Flatten()(inputLay)
	# r = Concatenate(axis=1)([m,og])
	out = Dense(100,activation='sigmoid')(m) #

	return tf.keras.Model(inputs=inputLay, outputs=out)

def oldBuildModel():
	# inputLay = tf.keras.Input(shape=(10,10,6))
	m = tf.keras.Sequential()
	m.add(InputLayer(input_shape=(10,10,6)))
	m.add(Flatten())
	m.add(Dense(250,activation='relu'))
	m.add(Dropout(.2))
	m.add(Dense(150,activation='relu'))
	m.add(Dropout(.2))
	m.add(Dense(100,activation='sigmoid'))
	return m


# CONFIGURING THE MODEL

# model = buildModel()
model = oldBuildModel()
# model = tf.keras.models.load_model('saved_model/my_model',compile=False)

# lossFunc = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
# optim = tf.keras.optimizers.SGD(lr=LEARNING_RATE, momentum = MOMENTUM) # optim = tf.keras.optimizers.SGD(lr=LEARNING_RATE, momentum = MOMENTUM)
optim = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
error = tf.keras.metrics.MeanAbsoluteError()
lossAvg = tf.keras.metrics.Mean()
# error = tf.keras.metrics.Mean()
accuracy = tf.keras.metrics.CategoricalAccuracy()
gameLength = tf.keras.metrics.Mean()
model.compile(optimizer=optim,loss=f1_loss,metrics=[error,accuracy])
# summary_writer = tf.summary.create_file_writer('logs')
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1)
print(model.summary())
# model.load_weights('saved_model/checkpoints/cp')


# Globals
ct = time.time()
env = gym.make('battleship1-v1')

@tf.function # Decoration is 10 fold faster
def makeMove(obs,e):
	# print("Traced with " + str(e))
	if e > 0:
		r = tf.random.uniform(shape=[],dtype=tf.dtypes.float16)
		if r < EPSILON:
			return tf.random.uniform(shape=[],maxval=100,dtype=tf.dtypes.int64)
		else:
			preds = model(obs, training=False)
			return tf.argmax(preds, 1)[0]
	preds = model(obs, training=False)
	return tf.argmax(preds, 1)[0]

# @tf.function
def trainGrads(feature,expect):
	with tf.GradientTape() as tape:
		# predictions = self.model(features)
		preds = model(feature,training=True)
		# loss = customLoss(expect, preds[0])
		# loss = lossFunc(expect, predictions[0])
		loss = f1_loss(expect, preds[0])
	gradients = tape.gradient(loss, model.trainable_variables)
	optim.apply_gradients(zip(gradients, model.trainable_variables))
	error.update_state(expect, preds[0])
	accuracy.update_state(expect, preds)
	lossAvg.update_state(loss)
	return gradients

# def singleStepConv():
hits = 0
iterartions = 0
observations = []
expecteds = []
# sess = tf.Session()
for epoch in range(0,NUM_GAMES):
	prevObs, prevOut = env.reset()
	# prevObs = np.copy(prevObs)
	prevOut = np.copy(prevOut) # could get rid of copy in replace of tensor conversion
	prevObs = np.moveaxis(prevObs,0,-1) # CPU ONLY
	prevObs = [[[x.value[0] for x in y] for y in c] for c in prevObs] # redo timeit with numpy
	prevObs = tf.convert_to_tensor([prevObs])

	done = False

	slotsLeft = np.ones(shape=100,dtype=np.float32)
	while not done:
		# Could Accelerate this, however few tf methods, and a couple of outside methods
		move = makeMove(prevObs,EPSILON).numpy()
		# print(move.numpy())
		obs, reward, done, out = env.step(move)
		obs = [[[x.value[0] for x in y] for y in c] for c in obs] # numpy may be faster

		# out = tf.Variable(tf.zeros([100]))

		if reward:
			hits += 1
		prevOut = [t.value[0] for t in prevOut] # look at the last input

		observations.append(prevObs[0])
		# expecteds.append(tf.convert_to_tensor(slotsLeft))
		# slotsLeft[move] = 0
		expecteds.append(tf.cast(tf.convert_to_tensor(prevOut),dtype=tf.dtypes.float32))

		obs = np.moveaxis(obs,0,-1) # CPU ONLY
		prevObs = tf.convert_to_tensor([obs])
		prevOut = np.copy(out)
		iterartions += 1
		if done:
			gameLength.update_state(env.counter)

	# TRAINING

	# for i in range(len(observations)): # FOR DEBUGGING #
	# 	ret = trainGrads(tf.reshape(observations[i],shape=(1,10,10,6)),expecteds[i])
	# 	pass
	
	# if len(observations) > 128:
		# observations = tf.stack(observations)
		# expecteds = tf.stack(expecteds)
		
	# 	ret = model.train_on_batch(x=observations,y=expecteds,reset_metrics=False,return_dict=True)
	# 	lossAvg.update_state(ret['loss'])

	# 	observations = []
	# 	expecteds = []

	if (epoch+1) % (NUM_GAMES // 30) == 0:
		# with summary_writer.as_default():
		# 	tf.summary.scalar('Loss', lossAvg.result(), step=epoch+1)
		# 	tf.summary.scalar('Error', error.result(), step=epoch+1)
		# 	tf.summary.scalar('Accuracy', accuracy.result()*100, step=epoch+1)
		# 	tf.summary.scalar('Hits', 100*hits / iterartions, step=epoch+1)
		# 	tf.summary.scalar('Game Length', gameLength.result(), step=epoch+1)
		print(f"Completed {epoch+1} epochs at {round(EPSILON,7)} in {round(time.time() - ct, 3)}s. L={round(float(lossAvg.result().numpy()),6)} E={round(float(error.result().numpy()),6)} A={round(float(accuracy.result().numpy()),6)} H={round(hits / iterartions,6)} I={round(float(gameLength.result().numpy()),3)}")
		error.reset_states()
		accuracy.reset_states()
		gameLength.reset_states()
		lossAvg.reset_states()
		hits = 0
		iterartions = 0
		# if EPSILON > 0.06:
		# 	EPSILON -= 0.02
		# else:
		# 	EPSILON /= 1.75
		# model.save_weights('saved_model/checkpoints/cp')
		ct = time.time()
		# x = tf.function(makeMove)

observationStack = tf.stack(observations)
expectedStack = tf.stack(expecteds)
dataset = tf.data.Dataset.from_tensor_slices((observationStack, expectedStack))
# dataset = dataset.batch(32)
# dataset = dataset.shuffle(dataset.__len__(), reshuffle_each_iteration=True)
tf.data.experimental.save(dataset=dataset,path='saved_data')
with tf.device('/cpu:0'):
	model.fit(x=observationStack,y=expectedStack,epochs=10,verbose=2,callbacks=[tensorboard_callback],use_multiprocessing=True) # multiprocessing?
model.save('saved_model/my_model')
print("Model Saved")