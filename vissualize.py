import gym
import gym_battleship1

import tensorflow as tf
import h5py
import numpy as np

import tensorflow.keras.losses
tensorflow.keras.losses.custom_loss = tf.nn.sigmoid_cross_entropy_with_logits

tf.keras.backend.set_image_data_format('channels_last')
from customs import customAccuracy, buildModel

from matplotlib import pyplot

env = gym.make('battleship1-v1')
env.reset()

# model.summary()
model = buildModel()
model.load_weights('saved_model/leviathan.h5', skip_mismatch=True, by_name=True) #unec?
weights = h5py.File('saved_model/leviathan.h5', 'r')
if tensorflow.keras.backend.image_data_format() == "channels_last":
	LC = weights['model_weights']['locally_connected2d']['locally_connected2d']['kernel:0'][()]
	LC = np.array(LC)
	LC = np.transpose(LC, [1,2,0,4,5,3])
	LCWeights = [LC, model.layers[-2].get_weights()[1]]
	model.layers[-2].set_weights(LCWeights)

def displayKernel(lay, ch=None):
	if model.layers[lay].use_bias:
		filters, biases = model.layers[lay].get_weights()
	else:
		filters = model.layers[lay].get_weights()[0]
	if ch is None:
		chRange = range(min([2,int(filters.shape[2])]))
	else:
		chRange = range(ch[0], ch[1])
	pyplot.close()
	# retrieve weights from the second hidden layer
	# normalize filter values to 0-1 so we can visualize them
	f_min, f_max = filters.min(), filters.max()
	filters = (filters - f_min) / (f_max - f_min)
	# plot first few filters
	n_filters, ix = 7, 1
	for i in range(n_filters):
		# get the filter
		f = filters[:, :, :, i]
		# plot each channel separately
		for j in chRange:
			# specify subplot and turn of axis
			ax = pyplot.subplot(n_filters, len(chRange), ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			pyplot.imshow(f[:, :, j], cmap='gray')
			ix += 1
	# show the figure
	pyplot.show()

# displayKernel(-2)
def scaler(x):
	if x.value[1] == "!M!":
		return -2
	elif '(' in x.value[1]:
		return 2
	return x.value[0]

vfunc = np.vectorize(scaler)
def heatMap(y_preds, state=None):
# 	pyplot.close()
	y_preds = np.reshape(y_preds, (10,10))
	ax = pyplot.subplot(1,2,1)
	pyplot.imshow(y_preds, cmap='gray')
	if state is not None:
		bx = pyplot.subplot(1,2,2)
		x = np.array(state)
		x = vfunc(x)
		pyplot.imshow(x, cmap='gray', vmin=-2, vmax=2)
	pyplot.show()

scores = []
choices = []
for each_game in range(1):
	score = 0
	prev_obs, _ = env.reset()
	prev_obs = [[[x.value[0] for x in y] for y in c] for c in prev_obs] # redo timeit with numpy
	prev_obs = tf.convert_to_tensor([prev_obs])
	if tensorflow.keras.backend.image_data_format() == "channels_last":
		prev_obs = tf.reshape(tf.transpose(prev_obs[0]),shape=(1,10,10,2)) # ONLY NEEDED FOR CPUS
	prev_action = -1
	for step_index in range(500):
		# env.render()

		# if len(prev_obs)==0:
		# 	action = 50
		# elif step_index < 50:
		# 	action = step_index

		logits = model(tf.cast(prev_obs, tf.float32), training=False)[0]
		heatMap(logits.numpy(), env.state)
# 		raise
		action = tf.argmax(logits,-1).numpy()
		print(action, logits[action].numpy(), tf.nn.softmax(logits)[action].numpy())
		
		# print(action, prev_obs)

		new_observation, reward, done, _ = env.step(action)
		print(reward)
		choices.append(1 if reward else 0)

		prev_obs = new_observation
		prev_obs = [[[x.value[0] for x in y] for y in c] for c in prev_obs] # redo timeit with numpy
		prev_obs = tf.convert_to_tensor([prev_obs])
		if tensorflow.keras.backend.image_data_format() == "channels_last":
			prev_obs = tf.reshape(tf.transpose(prev_obs[0]),shape=(1,10,10,2)) # ONLY NEEDED FOR CPUS

		prev_action = action
		if done:
			scores.append(env.counter)
			break
	scores.append(score)

print(scores)
print('Average Score:', sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))