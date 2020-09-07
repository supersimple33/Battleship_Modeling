import gym
import gym_battleship1

import tensorflow as tf
import numpy as np

import tensorflow.keras.losses
tensorflow.keras.losses.custom_loss = tf.nn.softmax_cross_entropy_with_logits

env = gym.make('battleship1-v1')
env.reset()

trained_model = tf.keras.models.load_model('saved_model/my_model',compile=False)

scores = []
choices = []

for each_game in range(10):
	score = 0
	prev_obs = env.reset()
	prev_obs = [[y.value[0] for y in x] for x in prev_obs]
	prev_obs = tf.convert_to_tensor([prev_obs])

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

			action = tf.argmax(logits,-1)[0].numpy()
		
		# print(action, prev_obs)

		choices.append(action)
		new_observation, reward, done = env.step(action)

		prev_obs = new_observation
		prev_obs = [[y.value[0] for y in x] for x in prev_obs]
		prev_obs = tf.convert_to_tensor([prev_obs])

		prev_action = action
		if done:
			scores.append(env.counter)
			break

	env.reset()
	scores.append(score)

print(scores)
print('Average Score:', sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
