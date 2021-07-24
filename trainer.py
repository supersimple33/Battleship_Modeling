import multiprocessing

import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Conv2D, BatchNormalization, LeakyReLU, Add, Flatten, Dense, Concatenate, Dot, Reshape, Dropout
import tensorflow.keras.backend as K
print(tf.__version__)

import numpy as np

import gym
import gym_battleship1

import builtins
# import timeit # DEBUG Only
import time
from random import random, shuffle, randrange, choice
CHANNEL_TYPE = 'channels_last'

tf.keras.backend.set_image_data_format(CHANNEL_TYPE)
from customs import customAccuracy, buildModel

from copy import copy

print(tf.__version__)

# MODEL TWEAKS
NUM_GAMES = 1000
EPSILON = 0.8
LEARNING_RATE = 0.001
TOLERANCE = 100 # how many tries to permit
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

# GLOBALS
ct = time.time()
env = gym.make('battleship1-v1')

blankBoard = np.zeros((100,), np.float32)

# Takes a move and executes it on the environment
@tf.function() # Decoration is 10 fold faster, 
def makeMove(obs,f):
	preds = model(obs, training=False)
	if f == 1:
		return tf.argmax(preds, 1, output_type=tf.int32)[0]
	else:
		return tf.math.top_k(preds, k=f)[1][0][f-1]

# Converts regular spaces to what would be seen in a game 
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

# Recursion Variables and Stats
lengthLimit = 98
hits = 0
iterations = 0
observations = []
expecteds = []
vfunc = np.vectorize(lambda e: e.value[0])
vfuncSingleShip = np.vectorize(singleShipSight)
possMoves = list(range(100))

seed = 0
failed = 0

if EPSILON >= 1.0: # Time Savers
	fullyRandom = np.random.rand(NUM_GAMES,100).argsort(1)[:,:100]

epoch = 0
# Loop through for each epoch
while epoch < NUM_GAMES:
	if failed != 0:
		prevObs, prevOut = env.reset(seed=seed)
	else:
		prevObs, prevOut = env.reset()
		seed = env.seed

	# Get the first observation
	prevObs = np.moveaxis(prevObs,0,-1) # CPU ONLY
	prevObs = vfunc(prevObs) # redo timeit with numpy
	prevObs = tf.convert_to_tensor([prevObs])

	# Set up variables for this epoch
	done = False
	slotsLeft = copy(possMoves)
	prevReward = False

	gameObs = []
	gameExp = []

	# Loop through until game is over
	while not done: # Could Accelerate this, however few tf methods, and a couple of outside methods
		# Decide what move to make
		if EPSILON >= 1.0:
			move = fullyRandom[epoch, env.counter]
		elif random() < EPSILON:
			move = choice(slotsLeft)
		else:
			for f in range(TOLERANCE):
				move = makeMove(prevObs,f+1).numpy() # Could convert f to tensor for speed up
				if move in slotsLeft:
					break
		obs, reward, done, out = env.step(move)
		obs = vfunc(obs)

		if reward:
			hits += 1

		# prevOut = vfunc(prevOut) #out = vfunc(out) #prevobs non zero should never have one in corresponding expected # non zero pre obs should never be greater than one
		# prevReward = reward

# 		print(out[move]) #fullyRandom[epoch, abs(env.counter - 2)]
		expect = np.zeros((100,), np.float32)
		expect[move] = 1.0
		gameObs.append(prevObs[0])
		gameExp.append(expect)
		if done: 
			observations.extend(gameObs)
			expecteds.extend(gameExp)
			failed = 0
		elif env.counter >= lengthLimit: # or equal
			failed += 1
			if failed > 24:
				failed = 0 # dont want to get stuck in a loop and waste time
				print("SKIPPING SEED: " + str(seed))
				epoch += 1
			elif failed > 7:
				print("failed " + str(seed) + " for the " + str(failed) + " time this is epoch " + str(epoch)) # report failures
			epoch -= 1
			break

# 		zer = tf.math.count_nonzero(prevObs, axis=1, keepdims=True, dtype=tf.float32)
# 		sums = tf.add_n([tf.reshape(zer, [100]), tf.convert_to_tensor(prevOut)]) #tf.reshape(expBatch, [32,1,10,10])
# 		if 2 in sums.numpy():
# 			raise

		# prevOut = np.copy(out)
		iterations += 1
		if done:
			gameLength.update_state(env.counter)
		else:
			obs = np.moveaxis(obs,0,-1) # CPU ONLY
			prevObs = tf.convert_to_tensor([obs])
			slotsLeft.remove(move)

	# TRAINING THE MODEL
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
			lossAvg.update_state(ret['loss'])
			accuracy.update_state(ret['customAccuracy'])

		observations = []
		expecteds = []

	# Ocassionally update stats and save the model
	if (epoch+1) % (NUM_GAMES // 30) == 0:
		# with summary_writer.as_default():
			# tf.summary.scalar('Loss', lossAvg.result(), step=epoch+1)
			# tf.summary.scalar('Error', error.result(), step=epoch+1)
			# tf.summary.scalar('Accuracy', accuracy.result()*100, step=epoch+1)
			# tf.summary.scalar('Hits', 100*hits / iterations, step=epoch+1)
			# tf.summary.scalar('Game Length', gameLength.result(), step=epoch+1)
		iterations = 1
		print(f"Completed {epoch+1} epochs at {round(EPSILON,7)} in {round(time.time() - ct, 3)}s. L={round(float(lossAvg.result().numpy()),6)} E={round(float(error.result().numpy()),6)} A={round(float(accuracy.result().numpy()),6)} H={round(hits / iterations,6)} I={round(float(gameLength.result().numpy()),3)}")
		
		# BUG: We need to finish atleast 1 game before changing the length limit
		lengthLimit = int(round(float(gameLength.result().numpy()),0)) - 1 #new game target length
		
		error.reset_states()
		accuracy.reset_states()
		# gameLength.reset_states() # roll the avg
		lossAvg.reset_states()
		hits = 0
		iterations = 0
		# if EPSILON > 0.06:
		# 	EPSILON -= 0.02
		# else:
			# EPSILON /= 1.75

		if lossAvg.result() != 0.0:
			model.save_weights('saved_model/checkpoints/cp')
		ct = time.time()
	
	# Next Iteration
	epoch += 1

# Finally save
model.save('saved_model/foresithe.h5')
print("Model Saved")
