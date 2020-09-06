import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Add, Flatten, Dense

import gym
import gym_battleship1
import timeit

import time
from collections import deque

import builtins

NUM_GAMES = 100000
FILTERS = 64 # 64 because its cool
EPSILON = 1.0

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

#BUILDING MODEL
reg = tf.keras.regularizers.L2(l2=0.0001)
inputLay = tf.keras.Input(shape=(10,10,6))#12

m = Conv2D(filters=FILTERS,kernel_size=(3,3),padding="same",use_bias=False,activation='linear',kernel_regularizer=reg,input_shape=(12,10,10))(inputLay)
m = BatchNormalization(axis=1)(m)
m = LeakyReLU()(m)

m = residualLayerCluster(m)
m = residualLayerCluster(m)
m = residualLayerCluster(m)

m = Conv2D(filters=6,kernel_size=(1,1),padding="same",use_bias=False,activation='linear',kernel_regularizer=reg)(m) #12
m = BatchNormalization(axis=1)(m)
m = LeakyReLU()(m)

m = Flatten()(m)
m = Dense(100,activation='sigmoid')(m)

model = tf.keras.Model(inputs=inputLay, outputs=m)
# print(model.summary())

# Globals
ct = time.time()
env = gym.make('battleship1-v1')

# @tf.function
def makeMove(obs):
	if EPSILON > 0:
		r = env.np_random.uniform()
		if r < EPSILON:
			return env.np_random.randint(0,99)
	logits = model.predict(obs)
	return tf.argmax(logits, 1)

# def singleStepConv():

for epoch in range(0,NUM_GAMES):
	env.reset()

	moveTracker = deque()

	obsSamp = [env.observation_space.sample()]

	obsSamp = tf.convert_to_tensor(obsSamp)
	# obsSamp
	# print()

	done = False

	while not done:
		move = makeMove(obsSamp)
		prevObs, reward, done = env.step(move)
		builtins.__dict__.update(locals())

		t = timeit.Timer('conved = list(map(lambda x: list(map(lambda y: y.value[0], x)), prevObs))')
		time = t.timeit(10000)
		print(time / 10000)

		t = timeit.Timer('r = [[y.value[0] for y in x] for x in prevObs]')
		time = t.timeit(10000)
		print(time / 10000)
		
		t = timeit.Timer('r = [map(lambda y: y.value[0], x) for x in prevObs]')
		time = t.timeit(10000)
		print(time / 10000)

		if reward:
			pass
		

