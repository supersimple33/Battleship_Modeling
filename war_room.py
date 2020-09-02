import gym
import gym_battleship1

# from keras.models	 import Sequential
# from keras.layers	 import Dense
# from keras.optimizers import Adam

import numpy as np
from random import randint # sources: https://blog.tanka.la/2018/10/19/build-your-first-ai-game-bot-using-openai-gym-keras-tensorflow-in-python/, https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa
import sys
import json

env = gym.make('battleship1-v1')
env.reset()
goal_steps = 500
score_requirement = 5 #tweak
intial_games = 50000

def playRandomGameFirst():
	for step_index in range(goal_steps):
		action = randomSpace()
		observation, reward, done, successStep = env.step(action)
		print("Step {}:".format(step_index))
		print("action: {}".format(action))
		# print("observation: {}".format(observation))
		env.render()
		print("reward: {}".format(reward))
		print("done: {}".format(done))
		# print("info: {}".format(info))
		if done:
			break
	env.reset()

def randomSpace():
	return randint(0, 9) + (randint(0,9) * 10)

# playRandomGameFirst() # demo run

def modelDataPrep():
	trainingData = []
	accepted_scores = 0
	totalMovesMade = 0
	for game_index in range(intial_games):
		print("Starting Simulation #" + str(game_index))
		previous_observation = [0]*100
		for step_index in range(goal_steps):
			action = randomSpace()
			observation, reward, done, succ, hit = env.step(action)
			if not succ:
				continue
			
			if hit:
				ops = [0]*100
				ops[action] = 1
				trainingData.append([previous_observation, ops])
				accepted_scores += 1

			previous_observation = observation
			if done:
				totalMovesMade += env.counter
				break
		env.reset()

	# print(accepted_scores)
	print("Model will be trained based on " + str(accepted_scores) + " boards.")
	
	return trainingData, totalMovesMade

print("Simulating")
training_data, totalMovesMade = modelDataPrep()
print("Data Collected")

print("Average Game Length = " + str(totalMovesMade / intial_games))

with open("data.json", "w") as f:
	json.dump(training_data, f, ensure_ascii=False, indent=4)
	print("Done")
	f.close()
	print("Closed")
