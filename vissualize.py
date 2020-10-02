import gym
import gym_battleship1

import tensorflow as tf
import numpy as np

import tensorflow.keras.losses
tensorflow.keras.losses.custom_loss = tf.nn.sigmoid_cross_entropy_with_logits

from customs import customAccuracy

from matplotlib import pyplot

tf.keras.backend.set_image_data_format('channels_first')

env = gym.make('battleship1-v1')
env.reset()

trained_model = tf.keras.models.load_model('saved_model/my_model.h5',compile=False,custom_objects={'customAccuracy':customAccuracy})
trained_model.summary()

def displayKernel(lay, ch=None):
	if ch is None:
		chRange = range(6)
	else:
		chRange = range(ch[0], ch[1])
	pyplot.close()
	# retrieve weights from the second hidden layer
	filters, biases = trained_model.layers[lay].get_weights()
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
			ax = pyplot.subplot(n_filters, ch[1]-ch[0], ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			pyplot.imshow(f[:, :, j], cmap='gray')
			ix += 1
	# show the figure
	pyplot.show()

def heatMap(y_preds, state=None):
# 	pyplot.close()
	y_preds = np.reshape(y_preds, (10,10))
	ax = pyplot.subplot(1,2,1)
	pyplot.imshow(y_preds, cmap='gray')
	if state is not None:
		bx = pyplot.subplot(1,2,2)
		x = [[(x.value[0] if x.value[1] != "!M!" else -2) for x in y] for y in state]
		pyplot.imshow(x, cmap='gray')
	pyplot.show()

scores = []
choices = []
for each_game in range(1):
	score = 0
	prev_obs, _ = env.reset()
	prev_obs = [[[x.value[0] for x in y] for y in c] for c in prev_obs] # redo timeit with numpy
	prev_obs = tf.convert_to_tensor([prev_obs])
	prev_obs = tf.reshape(tf.transpose(prev_obs[0]),shape=(1,10,10,6)) # ONLY NEEDED FOR CPUS
	prev_action = -1
	for step_index in range(500):
		# env.render()

		# if len(prev_obs)==0:
		# 	action = 50
		# elif step_index < 50:
		# 	action = step_index

		logits = trained_model(tf.cast(prev_obs, tf.float32), training=False)[0]
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
		prev_obs = tf.reshape(tf.transpose(prev_obs[0]),shape=(1,10,10,6)) # ONLY NEEDED FOR CPUS

		prev_action = action
		if done:
			scores.append(env.counter)
			break
	scores.append(score)

print(scores)
print('Average Score:', sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))