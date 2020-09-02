import gym
import gym_battleship1

import tensorflow as tf
import numpy as np

env = gym.make('battleship1-v1')
env.reset()

trained_model = tf.keras.models.load_model('saved_model/my_model.h5')

scores = []
choices = []

for each_game in range(10):
	score = 0
	prev_obs = []
	prev_action = -1
	for step_index in range(500):
		# env.render()

		if len(prev_obs)==0:
			action = 50
		elif step_index < 50:
			action = step_index
		else:
			action = np.argmax(trained_model.predict([prev_obs])[0])
			# suggestion = trained_model.predict([prev_obs])[0]
			# while True:
				# action = np.argmax(suggestion)
			#	 if action != prev_action:
			#		 break
			#	 else:
			#		 suggestion[action] = 0.0
		
		print(action, prev_obs)

		choices.append(action)
		new_observation, reward, done, succ, hit = env.step(action)
		prev_obs = new_observation
		prev_action = action
		if done:
			score.append(env.counter)
			break

	env.reset()
	scores.append(score)

print(scores)
print('Average Score:', sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
