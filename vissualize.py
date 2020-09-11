import gym
import gym_battleship1

import tensorflow as tf
import numpy as np

import tensorflow.keras.losses
tensorflow.keras.losses.custom_loss = tf.nn.sigmoid_cross_entropy_with_logits

env = gym.make('battleship1-v1')
env.reset()

trained_model = tf.keras.models.load_model('saved_model/my_model',compile=False)

scores = []
choices = []

for each_game in range(1):
	score = 0
	prev_obs = env.reset()
	prev_obs = [[[x.value[0] for x in y] for y in c] for c in prev_obs] # redo timeit with numpy
	prev_obs = tf.reshape(tf.transpose(tf.convert_to_tensor(prev_obs)),shape=(1,10,10,6)) # ONLY NEEDED FOR CPUS

	prev_action = -1
	for step_index in range(500):
		# env.render()

		# if len(prev_obs)==0:
		# 	action = 50
		# elif step_index < 50:
		# 	action = step_index
		if False:
			pass
		else:
			logits = trained_model.predict_step(prev_obs)[0]
			print(logits)
			action = tf.argmax(logits,-1).numpy()
			print(action, logits[action].numpy(), tf.nn.softmax(logits)[action].numpy())
		
		# print(action, prev_obs)

		choices.append(action)
		new_observation, reward, done, _ = env.step(action)

		prev_obs = new_observation
		prev_obs = [[[x.value[0] for x in y] for y in c] for c in prev_obs] # redo timeit with numpy
		prev_obs = tf.reshape(tf.transpose(tf.convert_to_tensor(prev_obs)),shape=(1,10,10,6)) # ONLY NEEDED FOR CPUS

		prev_action = action
		if done:
			scores.append(env.counter)
			break

	env.reset()
	scores.append(score)

print(scores)
print('Average Score:', sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
