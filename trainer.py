import multiprocessing

import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv2D, BatchNormalization, LeakyReLU, Add, Flatten, Dense, Concatenate, Dot, Reshape, Dropout
import tensorflow.keras.backend as K
print(tf.__version__)
# tf.autograph.set_verbosity(10,alsologtostdout=True)

import numpy as np

import gym
import gym_battleship1

import builtins
# import timeit # DEBUG Only
import time
from random import random, shuffle, randrange, choice
# from collections import deques

tf.keras.backend.set_image_data_format('channels_last')
from customs import customAccuracy, buildModel

from copy import copy

print(tf.__version__)

# MODEL TWEAKS
NUM_GAMES = 1000
EPSILON = 0.8
LEARNING_RATE = 0.0001
TOLERANCE = 0 # how many tries to permit
AXIS = 1 if CHANNEL_TYPE == "channels_first" else -1
# print(NUM_GAMES)
# CONFIGURING THE MODEL

model = buildModel()
# model = oldBuildModel()
# model = tf.keras.models.load_model('saved_model/my_model9.h5',compile=False,custom_objects={'customAccuracy':customAccuracy})

# for layer in model.layers:
# 	layer.trainable = False
# model.layers[-2].trainable = True # Switch between these two

# model.load_weights('saved_model/checkpoints/cp') #sometimes buggy with initializing the optimizer, perm fix

optim = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
error = tf.keras.metrics.MeanAbsoluteError()
lossAvg = tf.keras.metrics.Mean()
accuracy = tf.keras.metrics.Mean()
gameLength = tf.keras.metrics.Mean()
model.compile(optimizer=optim,loss='binary_crossentropy',metrics=[error,customAccuracy])
summary_writer = tf.summary.create_file_writer('logs')
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1)
print(model.summary())

# Globals
ct = time.time()
env = gym.make('battleship1-v1')

@tf.function() # Decoration is 10 fold faster, 
def makeMove(obs,f):
	preds = model(obs, training=False)
	if f == 1:
		return tf.argmax(preds, 1, output_type=tf.int32)[0]
	else:
		return tf.math.top_k(preds, k=f)[1][0][f-1]

def singleShipSight(e, match):
	if '2' in match.value[1]:
		return 1 if '2' in e.value[1] else 0
	if 'S' in match.value[1]:
		return 1 if 'S' in e.value[1] else 0
	if 'C' in match.value[1]:
		return 1 if 'C' in e.value[1] else 0
	if '4' in match.value[1]:
		return 1 if '4' in e.value[1] else 0
	if '5' in match.value[1]:
		return 1 if '5' in e.value[1] else 0
	return 0

hits = 0
iterartions = 0
observations = []
expecteds = []
vfunc = np.vectorize(lambda e: e.value[0])
vfuncSingleShip = np.vectorize(singleShipSight)
possMoves = list(range(100))
if EPSILON >= 1.0:
	fullyRandom = np.random.rand(NUM_GAMES,100).argsort(1)[:,:100]

for epoch in range(0,NUM_GAMES):
	prevObs, prevOut = env.reset()
	# prevObs = np.copy(prevObs)
	prevOut = np.copy(prevOut) # could get rid of copy in replace of tensor conversion
	# prevObs = np.moveaxis(prevObs,0,-1) # CPU ONLY
	prevObs = vfunc(prevObs) # redo timeit with numpy
	prevObs = tf.convert_to_tensor([prevObs])

	done = False
	slotsLeft = copy(possMoves)
	prevReward = False
	while not done:
		# Could Accelerate this, however few tf methods, and a couple of outside methods
		if EPSILON >= 1.0:
			move = fullyRandom[epoch, env.counter]
		elif random() < EPSILON:
			move = choice(slotsLeft)
		else:
			for f in range(TOLERANCE):
				move = makeMove(prevObs,f+1).numpy()
				if move in slotsLeft:
					break
		obs, reward, done, out = env.step(move)
		obs = vfunc(obs)

		if reward:
			hits += 1

# 		if prevReward:
# 			po = vfuncSingleShip(prevOut, env.state[move//10][move%10])
# 			if np.count_nonzero(po):
# 				observations.append(prevObs[0])
# 				expecteds.append(po)

		prevOut = vfunc(prevOut) #out = vfunc(out) #prevobs non zero should never have one in corresponding expected # non zero pre obs should never be greater than one
		prevReward = reward

# 		print(out[move]) #fullyRandom[epoch, abs(env.counter - 2)]

# 		if done: 
		observations.append(prevObs[0])
		expecteds.append(prevOut)

# 		zer = tf.math.count_nonzero(prevObs, axis=1, keepdims=True, dtype=tf.float32)
# 		sums = tf.add_n([tf.reshape(zer, [100]), tf.convert_to_tensor(prevOut)]) #tf.reshape(expBatch, [32,1,10,10])
# 		if 2 in sums.numpy():
# 			raise
		# obs = np.moveaxis(obs,0,-1) # CPU ONLY
		prevObs = tf.convert_to_tensor([obs])
		prevOut = np.copy(out)
		iterartions += 1
		if done:
			gameLength.update_state(env.counter)
		else:
			slotsLeft.remove(move)

	# TRAINING

	# for i in range(len(observations)): # FOR DEBUGGING #
	# 	ret = trainGrads(tf.reshape(observations[i],shape=(1,10,10,6)),expecteds[i])
	# 	pass

	if len(observations) > 1024:
		batch_size = 32

		pairing = list(zip(observations, expecteds))
		shuffle(pairing)
		observations, expecteds = list(zip(*pairing))

		observations = [observations[i:i + batch_size] for i in range(0, len(observations), batch_size)]
		expecteds = [expecteds[i:i + batch_size] for i in range(0, len(expecteds), batch_size)]
		for b in range(len(observations)):
			obsBatch = tf.stack(observations[b])
			expBatch = tf.stack(expecteds[b])

			ret = model.train_on_batch(x=obsBatch,y=expBatch,reset_metrics=False, return_dict=True)
			lossAvg.update_state(ret[0])
			accuracy.update_state(ret[2])

		observations = []
		expecteds = []

	if (epoch+1) % (NUM_GAMES // 30) == 0:
		# with summary_writer.as_default():
			# tf.summary.scalar('Loss', lossAvg.result(), step=epoch+1)
			# tf.summary.scalar('Error', error.result(), step=epoch+1)
			# tf.summary.scalar('Accuracy', accuracy.result()*100, step=epoch+1)
			# tf.summary.scalar('Hits', 100*hits / iterartions, step=epoch+1)
			# tf.summary.scalar('Game Length', gameLength.result(), step=epoch+1)
		print(f"Completed {epoch+1} epochs at {round(EPSILON,7)} in {round(time.time() - ct, 3)}s. L={round(float(lossAvg.result().numpy()),6)} E={round(float(error.result().numpy()),6)} A={round(float(accuracy.result().numpy()),6)} H={round(hits / iterartions,6)} I={round(float(gameLength.result().numpy()),3)}")
		error.reset_states()
		accuracy.reset_states()
		gameLength.reset_states()
		lossAvg.reset_states()
		hits = 0
		iterartions = 0
		if EPSILON > 0.06:
			EPSILON -= 0.02
		# else:
			# EPSILON /= 1.75

		if lossAvg.result() != 0.0:
			model.save_weights('saved_model/checkpoints/cp')
		ct = time.time()

model.save('saved_model/leviathan.h5')
print("Model Saved")