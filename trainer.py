# builtins.__dict__.update(locals())
# t = timeit.Timer('conved = list(map(lambda x: list(map(lambda y: y.value[0], x)), prevObs))')
# time = t.timeit(10000)
# print(time / 10000)

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Add, Flatten, Dense

import numpy as np

import gym
import gym_battleship1

import builtins
import timeit # DEBUG Only
import time
# from collections import deques

# MODEL TWEAKS
NUM_GAMES = 1000
FILTERS = 64 # 64 because its cool
EPSILON = 0.5
LEARNING_RATE = 0.1
MOMENTUM = 0.9

def convLayerCluster(inp):
	m = Conv2D(filters=FILTERS,kernel_size=(3,3),padding="same",use_bias=False,activation='linear',kernel_regularizer=reg)(inp)
	m = BatchNormalization(axis=1)(m)
	return LeakyReLU()(m)

def residualLayerCluster(inp):
	m = convLayerCluster(inp)
	m = Conv2D(filters=FILTERS,kernel_size=(3,3),padding="same",use_bias=False,activation='linear',kernel_regularizer=reg)(m)
	m = BatchNormalization(axis=1)(m)
	m = Add()([inp,m])
	return LeakyReLU()(m)

# def softmax_cross_entropy_with_logits(y_true, y_pred):

#BUILDING MODEL
reg = tf.keras.regularizers.L2(l2=0.0001)
inputLay = tf.keras.Input(shape=(10,10,1))#12

m = Conv2D(filters=FILTERS,kernel_size=(3,3),padding="same",use_bias=False,activation='linear',kernel_regularizer=reg,input_shape=(12,10,10))(inputLay)
m = BatchNormalization(axis=1)(m)
m = LeakyReLU()(m)

m = residualLayerCluster(m)
m = residualLayerCluster(m)
m = residualLayerCluster(m)

m = Conv2D(filters=1,kernel_size=(1,1),padding="same",use_bias=False,activation='linear',kernel_regularizer=reg)(m) #12
m = BatchNormalization(axis=1)(m)
m = LeakyReLU()(m)

m = Flatten()(m)
m = Dense(100,activation='sigmoid')(m)

model = tf.keras.Model(inputs=inputLay, outputs=m)

lossFunc = tf.nn.softmax_cross_entropy_with_logits
optim = tf.keras.optimizers.SGD(lr=LEARNING_RATE, momentum = MOMENTUM)
error = tf.keras.metrics.MeanAbsoluteError()
accuracy = tf.keras.metrics.CategoricalAccuracy()
model.compile(optimizer=optim,loss=lossFunc,loss_weights=[0.5],metrics=[error,accuracy])
# print(model.summary())

# Globals
ct = time.time()
env = gym.make('battleship1-v1')

@tf.function # Decoration is 10 fold faster
def makeMove(obs):
	if EPSILON > 0:
		r = tf.random.uniform(shape=[],dtype=tf.dtypes.float16)
		if r < EPSILON:
			return tf.random.uniform(shape=[],maxval=100,dtype=tf.dtypes.int64)
		else:
			logits = model.predict_step(obs)
			return tf.argmax(logits, 1)[0]
	logits = model.predict_step(obs)
	return tf.argmax(logits, 1)[0]

# def singleStepConv():
hits = 0
iterartions = 0
for epoch in range(0,NUM_GAMES):
	prevObs = env.reset()
	prevObs = [[y.value[0] for y in x] for x in prevObs]
	prevObs = tf.convert_to_tensor([prevObs])

	observations = [] # could also use deque
	expecteds = []
	done = False

	while not done:
		# Could Accelerate this, however few tf methods, and a couple of outside methods
		move = makeMove(prevObs).numpy()
		# print(move.numpy())
		obs, reward, done = env.step(move)
		obs = [[y.value[0] for y in x] for x in obs]

		out = tf.Variable(tf.zeros([100]))
		# out = np.zeros(100)
		if reward:
			hits += 1
			out[move].assign(1.)

		observations.append(prevObs[0])
		expecteds.append(out)

		prevObs = tf.convert_to_tensor([obs])
		iterartions += 1
	
	observations = tf.stack(observations)
	expecteds = tf.stack(expecteds)
	
	model.train_on_batch(x=observations,y=expecteds,reset_metrics=False)

	if (epoch+1) % (NUM_GAMES // 20) == 0:
		print(f"Completed {NUM_GAMES // 20} at e{EPSILON} epochs in {round(time.time() - ct, 3)}s. E={round(float(error.result().numpy()),6)} A={round(float(accuracy.result().numpy()),6)} H={round(hits / iterartions,6)}")
		ct = time.time()
		error.reset_states()
		accuracy.reset_states()
		hits = 0
		iterartions = 0

model.save('saved_model/my_model')