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
from random import random, shuffle, randrange
# from collections import deques

from customs import customAccuracy, buildModel

from copy import copy

print(tf.__version__)
tf.keras.backend.set_image_data_format('channels_first')

# MODEL TWEAKS
NUM_GAMES = 1200
EPSILON = 1.0
LEARNING_RATE = 0.0001
MOMENTUM = 0.05
CHANNEL_TYPE = "channels_last"
TOLERANCE = 1 # how many tries to permit
AXIS = 1 if CHANNEL_TYPE == "channels_first" else -1
# print(NUM_GAMES)
# CONFIGURING THE MODEL

model = buildModel()
# model = tf.keras.models.load_model('saved_model/my_model10.h5',compile=False,custom_objects={'customAccuracy':customAccuracy})
# for layer in model.layers:
# 	layer.trainable = True
# model.layers[-2].trainable = False # Switch between these two

optim = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
error = tf.keras.metrics.MeanAbsoluteError()
lossAvg = tf.keras.metrics.Mean()
accuracy = tf.keras.metrics.Mean()
gameLength = tf.keras.metrics.Mean()
model.compile(optimizer=optim,loss='binary_crossentropy',metrics=[error,customAccuracy])

summary_writer = tf.summary.create_file_writer('logs')
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1)
print(model.summary())
# model.load_weights('saved_model/checkpoints/cp')

# Globals
ct = time.time()
env = gym.make('battleship1-v1')

@tf.function() # Decoration is 10 fold faster, 
def makeMove(obs,f):
	print(e, f)
	if f == 1:
		preds = model(obs, training=False)
		return tf.argmax(preds, 1, output_type=tf.int32)[0]
	else:
		preds = model(obs, training=False)
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
# sess = tf.Session()
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
	slotsLeft = copy.copy(possMoves)
	prevReward = False
	while not done:
		# Could Accelerate this, however few tf methods, and a couple of outside methods
		if EPSILON >= 1.0:
			move = fullyRandom[epoch, env.counter]
		elif random() < EPSILON:
			pass
		else:
			for f in range(TOLERANCE):
				move = makeMove(prevObs,f+1).numpy()
				if move in slotsLeft:
					break
		obs, reward, done, out = env.step(move)
		obs = vfunc(obs)

# 		prevOut = vfunc(prevOut)
		if reward:
			hits += 1
# 		if prevReward:
# 			po = vfuncSingleShip(prevOut, env.state[move//10][move%10])
# 			if np.count_nonzero(prevOut):
		prevOut = vfunc(prevOut)
# 			observations.append(prevObs[0])
# 			expecteds.append(po)
# 		prevReward = reward

# 		if done:
		observations.append(prevObs[0])
		expecteds.append(prevOut)
# 		expecteds.append(tf.convert_to_tensor(slotsLeft))
# 		expecteds.append(tf.cast(tf.convert_to_tensor(prevOut),dtype=tf.dtypes.float32))

		# obs = np.moveaxis(obs,0,-1) # CPU ONLY
		prevObs = tf.convert_to_tensor([obs])
		prevOut = np.copy(out)
		iterartions += 1
		if done:
			gameLength.update_state(env.counter)
# 		else:
# 			slotsLeft.remove(move)

	# TRAINING

	# for i in range(len(observations)): # FOR DEBUGGING #
	# 	ret = trainGrads(tf.reshape(observations[i],shape=(1,10,10,6)),expecteds[i])
	# 	pass

	if len(observations) > 1024:
# 		rt = time.time()
		batch_size = 32
		shuffle(observations)
		shuffle(expecteds)
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
# 		print(round(time.time() - rt, 3))

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
# 		if EPSILON > 0.06:
# 			EPSILON -= 0.02
# 		else:
# 		EPSILON /= 1.75
		model.save_weights('saved_model/checkpoints/cp')
		ct = time.time()

# observationStack = tf.stack(observations)
# expectedStack = tf.stack(expecteds)
# dataset = tf.data.Dataset.from_tensor_slices((observationStack, expectedStack))

# observations = np.array(observations)
# expecteds = np.array(expecteds)
# print(observations)

# with open('data.npz', 'wb') as f:
# 	print('started save')
# 	np.savez_compressed(f, observations, expecteds)
# 	print('saved')
# with open('data.npz', 'rb') as f:
# 	l = np.load(f)
	# pass

# dataset = dataset.batch(32)
# dataset = dataset.shuffle(dataset.__len__(), reshuffle_each_iteration=True)
# tf.data.experimental.save(dataset=dataset,path='saved_data',compression='GZIP')
# with tf.device('/cpu:0'):
# 	model.fit(x=observationStack,y=expectezdStack,epochs=10,verbose=2,callbacks=[tensorboard_callback],use_multiprocessing=True) # multiprocessing?
model.save('saved_model/my_model10.h5')
print("Model Saved")