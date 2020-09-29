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

from customs import customAccuracy

print(tf.__version__)

# MODEL TWEAKS
NUM_GAMES = 30
FILTERS = 64 # 64 because its cool
EPSILON = 2.0 # Epsilon must start close to one or model training will scew incredibelly
LEARNING_RATE = 0.01
MOMENTUM = 0.05
CHANNEL_TYPE = "channels_last"
AXIS = 1 if CHANNEL_TYPE == "channels_first" else -1
# print(NUM_GAMES)
# CONFIGURING THE MODEL

# model = buildModel()
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

# @tf.function(experimental_compile=True) # Decoration is 10 fold faster
# def makeMove(obs,e):
# 	# print("Traced with " + str(e))
# 	if e > 0:
# 		r = tf.random.uniform(shape=[],dtype=tf.dtypes.float16)
# 		if r < EPSILON:
# 			return tf.random.uniform(shape=[],maxval=100,dtype=tf.dtypes.int64)
# 		else:
# 			preds = model(obs, training=False)
# 			return tf.argmax(preds, 1)[0]
# 	preds = model(obs, training=False)
# 	return tf.argmax(preds, 1)[0]

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
for epoch in range(0,NUM_GAMES):
	prevObs, prevOut = env.reset()
	# prevObs = np.copy(prevObs)
	prevOut = np.copy(prevOut) # could get rid of copy in replace of tensor conversion
	# prevObs = np.moveaxis(prevObs,0,-1) # CPU ONLY
	prevObs = vfunc(prevObs) # redo timeit with numpy
# 	prevObs = tf.convert_to_tensor([prevObs])

	done = False

	slotsLeft = np.ones(shape=100,dtype=np.float32)
	while not done:
		# Could Accelerate this, however few tf methods, and a couple of outside methods
		# move = makeMove(prevObs,EPSILON).numpy()
		move = randint(0,99)
		# print(move.numpy())
		obs, reward, done, out = env.step(move)
		obs = vfunc(obs) # numpy may be faster

		if reward:
			hits += 1
		prevOut = vfunc(prevOut)

		observations.append(prevObs)
		# expecteds.append(tf.convert_to_tensor(slotsLeft))
		# slotsLeft[move] = 0
		expecteds.append(prevOut.astype(float))
# 		expecteds.append(tf.cast(tf.convert_to_tensor(prevOut),dtype=tf.dtypes.float32))

		# obs = np.moveaxis(obs,0,-1) # CPU ONLY
# 		prevObs = tf.convert_to_tensor([obs])
		prevOut = np.copy(out)
		iterartions += 1
		if done:
			gameLength.update_state(env.counter)

	# TRAINING

	# for i in range(len(observations)): # FOR DEBUGGING #
	# 	ret = trainGrads(tf.reshape(observations[i],shape=(1,10,10,6)),expecteds[i])
	# 	pass

	n=32
	if len(observations) > 320:
		shuffle(observations)
		shuffle(expecteds)
		observations = [observations[i:i + n] for i in range(0, len(observations), n)]
		expecteds = [expecteds[i:i + n] for i in range(0, len(expecteds), n)]
		for b in range(len(observations)):

			observations = tf.stack(observations[b])
			expecteds = tf.stack(expecteds[b])

			ret = model.train_on_batch(x=observations,y=expecteds,reset_metrics=False,return_dict=True)
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
		# if EPSILON > 0.06:
		# 	EPSILON -= 0.02
		# else:
		# 	EPSILON /= 1.75
		# model.save_weights('saved_model/checkpoints/cp')
		ct = time.time()

# observationStack = tf.stack(observations)
# expectedStack = tf.stack(expecteds)
# dataset = tf.data.Dataset.from_tensor_slices((observationStack, expectedStack))

observations = np.array(observations)
expecteds = np.array(expecteds)
# print(observations)

with open('data.npz', 'wb') as f:
	print('started save')
	np.savez_compressed(f, observations, expecteds)
	print('saved')
with open('data.npz', 'rb') as f:
	l = np.load(f)
	pass

# dataset = dataset.batch(32)
# dataset = dataset.shuffle(dataset.__len__(), reshuffle_each_iteration=True)
# tf.data.experimental.save(dataset=dataset,path='saved_data',compression='GZIP')
# with tf.device('/cpu:0'):
# 	model.fit(x=observationStack,y=expectedStack,epochs=10,verbose=2,callbacks=[tensorboard_callback],use_multiprocessing=True) # multiprocessing?
# model.save('saved_model/my_model')
# print("Model Saved")