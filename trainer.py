import multiprocessing

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
from random import randint, shuffle
# from collections import deques

from customs import customAccuracy, buildModel0

print(tf.__version__)
tf.keras.backend.set_image_data_format('channels_last')

# MODEL TWEAKS
NUM_GAMES = 3000
EPSILON = 0.5
LEARNING_RATE = 0.01
MOMENTUM = 0.05
CHANNEL_TYPE = "channels_last"
TOLERANCE = 1 # how many tries to permit
AXIS = 1 if CHANNEL_TYPE == "channels_first" else -1
# print(NUM_GAMES)
# CONFIGURING THE MODEL

# model = buildModel0(None)
# model = oldBuildModel()
model = tf.keras.models.load_model('saved_model/my_model',compile=False,custom_objects={'customAccuracy':customAccuracy})

# lossFunc = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
# optim = tf.keras.optimizers.SGD(lr=LEARNING_RATE, momentum = MOMENTUM) # optim = tf.keras.optimizers.SGD(lr=LEARNING_RATE, momentum = MOMENTUM)
optim = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
error = tf.keras.metrics.MeanAbsoluteError()
lossAvg = tf.keras.metrics.Mean()
# error = tf.keras.metrics.Mean()
# accuracy = tf.keras.metrics.CategoricalAccuracy()
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

@tf.function(experimental_compile=True) # Decoration is 10 fold faster, 
def makeMove(obs,e,f):
	# print("Traced with " + str(e))
	if e > 0 and f == 1:
		r = tf.random.uniform(shape=[],dtype=tf.float32)
		if r < EPSILON:
			moveVals = K.flatten(tf.reduce_sum(tf.abs(obs),axis=1) * -1)
			mCount = 100 - tf.math.count_nonzero(moveVals,dtype=tf.int32) # could also use reduce sum use timeit to make a choice
			L = float(mCount) * (r / e) # skips having to do another round of random
			aMoves = tf.math.top_k(moveVals, k=mCount)[1]
			return aMoves[int(L)]
		else: #does not respect randomness
			preds = model(obs, training=False)
			return tf.math.top_k(preds, k=f)[1][0][f-1]
	else: #does not respect randomness
		preds = model(obs, training=False)
		return tf.math.top_k(preds, k=f)[1][0][f-1]

# @tf.function(experimental_compile=True)
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
vfunc = np.vectorize(lambda e: e.value[0])
# sess = tf.Session()
possMoves = np.arange(100)
for epoch in range(0,NUM_GAMES):
	prevObs, prevOut = env.reset()
	# prevObs = np.copy(prevObs)
	prevOut = np.copy(prevOut) # could get rid of copy in replace of tensor conversion
	# prevObs = np.moveaxis(prevObs,0,-1) # CPU ONLY
	prevObs = vfunc(prevObs) # redo timeit with numpy
	prevObs = tf.convert_to_tensor([prevObs])

	done = False
	pmov = list(possMoves)
	slotsLeft = list(possMoves) # could also use
	while not done:
		# Could Accelerate this, however few tf methods, and a couple of outside methods
		for f in range(TOLERANCE):
			move = makeMove(prevObs,EPSILON,f+1).numpy()
			if move in slotsLeft:
				break
		# move = randint(0,99)
		obs, reward, done, out = env.step(move)
		obs = vfunc(obs) # numpy may be faster

		if reward:
			hits += 1
		prevOut = vfunc(prevOut)

		observations.append(prevObs[0])
		# expecteds.append(tf.convert_to_tensor(slotsLeft))
		expecteds.append(prevOut.astype(float))
# 		expecteds.append(tf.cast(tf.convert_to_tensor(prevOut),dtype=tf.dtypes.float32))

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

	if len(observations) > 320:
		batch_size = 32
		shuffle(observations)
		shuffle(expecteds)
		observations = [observations[i:i + batch_size] for i in range(0, len(observations), batch_size)]
		expecteds = [expecteds[i:i + batch_size] for i in range(0, len(expecteds), batch_size)]
		for b in range(len(observations)):

			obsBatch = tf.stack(observations[b])
			expBatch = tf.stack(expecteds[b])

			ret = model.train_on_batch(x=obsBatch,y=expBatch,reset_metrics=False,return_dict=True)
			lossAvg.update_state(ret['loss'])
			accuracy.update_state(ret['customAccuracy'])

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
		else:
			EPSILON /= 1.75
		# model.save_weights('saved_model/checkpoints/cp')
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
# 	model.fit(x=observationStack,y=expectedStack,epochs=10,verbose=2,callbacks=[tensorboard_callback],use_multiprocessing=True) # multiprocessing?
model.save('saved_model/my_model')
print("Model Saved")