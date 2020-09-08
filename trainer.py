# builtins.__dict__.update(locals())
# t = timeit.Timer('conved = list(map(lambda x: list(map(lambda y: y.value[0], x)), prevObs))')
# time = t.timeit(10000)
# print(time / 10000)
import faulthandler
import multiprocessing
logger = open('log.txt', 'w')
faulthandler.enable(file=logger)

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Add, Flatten, Dense

import numpy as np

import gym
import gym_battleship1

import builtins
# import timeit # DEBUG Only
import time
# from collections import deques

print(tf.__version__)

# MODEL TWEAKS
NUM_GAMES = 21000
FILTERS = 64 # 64 because its cool
EPSILON = 0.9 # Epsilon must start close to one or model training will scew incredibelly
LEARNING_RATE = 0.002
MOMENTUM = 0.9
CHANNEL_TYPE = "channels_last"

def convLayerCluster(inp):
	m = Conv2D(filters=FILTERS,kernel_size=(3,3),padding="same",use_bias=False,activation='linear',kernel_regularizer=reg,data_format=CHANNEL_TYPE)(inp)
	m = BatchNormalization(axis=1)(m)
	return LeakyReLU()(m)

def residualLayerCluster(inp):
	m = convLayerCluster(inp)
	m = Conv2D(filters=FILTERS,kernel_size=(3,3),padding="same",use_bias=False,activation='linear',kernel_regularizer=reg,data_format=CHANNEL_TYPE)(m)
	m = BatchNormalization(axis=1)(m)
	m = Add()([inp,m])
	return LeakyReLU()(m)

# def softmax_cross_entropy_with_logits(y_true, y_pred):

#BUILDING MODEL
reg = tf.keras.regularizers.L2(l2=0.0001)
def buildModel():
	inputLay = tf.keras.Input(shape=(10,10,6))#12

	m = Conv2D(filters=FILTERS,kernel_size=(3,3),padding="same",use_bias=False,activation='linear',kernel_regularizer=reg,data_format=CHANNEL_TYPE)(inputLay)
	m = BatchNormalization(axis=1)(m)
	m = LeakyReLU()(m)

	m = residualLayerCluster(m)
	m = residualLayerCluster(m)
	# m = residualLayerCluster(m) # removed on cluster for added simplicity

	m = Conv2D(filters=6,kernel_size=(1,1),padding="same",use_bias=False,activation='linear',kernel_regularizer=reg,data_format=CHANNEL_TYPE)(m) #12
	m = BatchNormalization(axis=1)(m)
	m = LeakyReLU()(m)

	m = Flatten()(m)
	m = Dense(100,activation='softmax')(m)

	return tf.keras.Model(inputs=inputLay, outputs=m)

model = buildModel()
# model = tf.keras.models.load_model('saved_model/my_model',compile=False)

lossFunc = tf.keras.losses.MeanSquaredLogarithmicError()
optim = tf.keras.optimizers.SGD(lr=LEARNING_RATE, momentum = MOMENTUM)
error = tf.keras.metrics.MeanAbsoluteError()
# error = tf.keras.metrics.Mean()
accuracy = tf.keras.metrics.CategoricalAccuracy()
gameLength = tf.keras.metrics.Mean()
model.compile(optimizer=optim,loss=lossFunc,loss_weights=[0.5],metrics=[error,accuracy])
print(model.summary())

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
			logits = model.predict_step(obs)
			return tf.argmax(logits, 1)[0]
	logits = model.predict_step(obs)
	return tf.argmax(logits, 1)[0]

@tf.function
def trainGrads(feature,expect):
	with tf.GradientTape() as tape:
		# predictions = self.model(features)
		predictions = model.predict_step(feature)
		loss = lossFunc(expect, predictions)
	gradients = tape.gradient(loss, model.trainable_variables)
	optim.apply_gradients(zip(gradients, model.trainable_variables))
	return loss

# def singleStepConv():
hits = 0
iterartions = 0
for epoch in range(0,NUM_GAMES):
	prevObs = env.reset()
	prevObs = [[[x.value[0] for x in y] for y in c] for c in prevObs] # redo timeit with numpy
	prevObs = tf.reshape(tf.transpose(tf.convert_to_tensor(prevObs)),shape=(1,10,10,6)) # ONLY NEEDED FOR CPUS
	# prevObs = tf.reshape(prevObs, (1,10,10,6))

	observations = [] # could also use deque
	expecteds = []
	done = False

	while not done:
		# Could Accelerate this, however few tf methods, and a couple of outside methods
		move = makeMove(prevObs,EPSILON).numpy()
		# print(move.numpy())
		obs, reward, done = env.step(move)
		obs = [[[x.value[0] for x in y] for y in c] for c in obs]

		out = tf.Variable(tf.zeros([100]))
		# out = np.zeros(100)
		if reward:
			hits += 1
			out[move].assign(1.)
			# print(out)
			observations.append(prevObs[0])
			expecteds.append(out)
		elif done:
			observations.append(prevObs[0])
			expecteds.append(out)

		# observations.append(prevObs[0])
		# expecteds.append(out)

		prevObs = tf.convert_to_tensor([obs])
		prevObs = tf.reshape(tf.transpose(tf.convert_to_tensor(prevObs)),shape=(1,10,10,6))
		iterartions += 1
		if done:
			gameLength.update_state(env.counter)
	
	observations = tf.stack(observations)
	expecteds = tf.stack(expecteds)
	
	model.train_on_batch(x=observations,y=expecteds,reset_metrics=False)

	if (epoch+1) % (NUM_GAMES // 30) == 0:
		print(f"Completed {epoch+1} epochs at {round(EPSILON,7)} in {round(time.time() - ct, 3)}s. E={round(float(error.result().numpy()),6)} A={round(float(accuracy.result().numpy()),6)} H={round(hits / iterartions,6)} L={round(float(gameLength.result().numpy()),3)}")
		error.reset_states()
		accuracy.reset_states()
		gameLength.reset_states()
		hits = 0
		iterartions = 0
		if EPSILON > 0.06:
			EPSILON -= 0.01
		else:
			EPSILON /= 1.75
		ct = time.time()
		# x = tf.function(makeMove)

model.save('saved_model/my_model')
print("Model Saved")