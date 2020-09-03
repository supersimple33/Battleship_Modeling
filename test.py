import gym
import gym_battleship1

import tensorflow as tf
import numpy as np
import timeit

env = gym.make('battleship1-v1')
env.reset()

scores = []
choices = []

for each_game in range(1):
	score = 0
	prev_obs = []
	prev_action = -1
	for step_index in range(500):
		env.render()

		action = int(input("Move# "))
		
		# print(action, prev_obs)

		choices.append(action)
		new_observation, reward, done, info = env.step(action)
		succ, hit = info

		prev_obs = new_observation
		prev_action = action
		if done:
			scores.append(env.counter)
			break

	env.reset()
	scores.append(score)

print(scores)
print('Average Score:', sum(scores)/len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
