import gym
import gym_battleship1

import multiprocessing
from os import getpid

import enum
class Space(enum.Enum):
	Empty = 0,"|-|" #_

import numpy as np
import random # sources: https://blog.tanka.la/2018/10/19/build-your-first-ai-game-bot-using-openai-gym-keras-tensorflow-in-python/, https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa
import sys
# import json
import pandas as pd

intial_games = 350

# def playRandomGameFirst():
# 	for step_index in range(goal_steps):
# 		action = randomSpace()
# 		observation, reward, done = env.step(action)
# 		print("Step {}:".format(step_index))
# 		print("action: {}".format(action))
# 		# print("observation: {}".format(observation))
# 		env.render()
# 		print("reward: {}".format(reward))
# 		print("done: {}".format(done))
# 		# print("info: {}".format(info))
# 		if done:
# 			break
# 	env.reset()

# playRandomGameFirst() # demo run
def worker(procnum):
	seeded = random.Random()
	print("Running on " + str(getpid()))
	vfunc = np.vectorize(lambda e: e.value[0])
	possMovesNumpy = np.arange(100)
	env = gym.make('battleship1-v1')
	env.reset()
	trainingData = []
	accepted_scores = 0
	totalMovesMade = 0
	for game_index in range(intial_games):
		if game_index % 50 == 0:
			print("Starting Simulation #" + str(game_index) + " for process #" + str(procnum))
		previous_observation = np.full(shape=(6,10,10),fill_value=Space.Empty)
		possMoves = possMovesNumpy.tolist()
		done = False
		while not done:
			action = seeded.choice(possMoves)
			observation, reward, done, _ = env.step(action)
			if reward:
				v = np.transpose(vfunc(previous_observation), (2,0,1))
				trainingData.append(np.append(v,[action]))
				# numpy.append(previous_observation)
				accepted_scores += 1

			previous_observation = observation
			if done:
				totalMovesMade += env.counter
				break
			possMoves.remove(action)
		env.reset()

	# print(accepted_scores)
	print("Model will be trained based on " + str(accepted_scores) + " boards. proc #" + str(procnum))
	
	return trainingData, totalMovesMade

if __name__ == '__main__':
	pool = multiprocessing.Pool(processes=6)
	ret = pool.map(worker,range(6))

	fullGames = []
	for i in range(5):
		fullGames += ret[i][0]
	print("started saving")
	df = pd.DataFrame(fullGames,dtype=np.int32)
	# df.astype('int64',copy=False)
	df.to_csv('shortdata.csv',index=False,chunksize=1000)
	print("saved csv")
