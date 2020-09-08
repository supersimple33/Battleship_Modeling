import gym
from gym import error, spaces, utils

import enum
import numpy as np
# from random import randint seeding replaces

# indList = ("|-|","!M!","(2)","(S)","(C)","(4)","(5)",3"x2x","xSx","xCx","x4x","x5x","|2|","|S|","|C|","|4|","|5|","HiddenCruiser")
class Space(enum.Enum):
	Empty = 0,"|-|" #_

	Miss = 1,"!M!" #m
	
	HitPTwo = 1,"(2)"
	HitPSub = 1,"(S)"
	HitPCruiser = 1,"(C)"
	HitPFour = 1,"(4)"
	HitPFive = 1,"(5)"

	SunkTwo = -1,"x2x"
	SunkSub = -1,"xSx"
	SunkCruiser = -1,"xCx"
	SunkFour = -1,"x4x"
	SunkFive = -1,"x5x"

	HiddenTwo = 12,"|2|" # Need to update these values likely 1
	HiddenSub = 13,"|S|"
	HiddenCruiser = 14,"|C|"
	HiddenFour = 15,"|4|"
	HiddenFive = 16,"|5|"

def addShip(state, ship, len_, x, y, d):
		r = range(0, len_)
		# print(self.boardRep())
		if d == 0:
			for j in r:
				if state[y + j][x] != Space.Empty: # loop run twice in order to make sure all spaces are clear before making modifications
					return False
			for j in r:
				state[y + j][x] = ship
		elif d == 1:
			for j in r:
				if state[y][x + j] != Space.Empty:
					return False
			for j in r:
				state[y][x + j] = ship
		elif d == 2:
			for j in r:
				if state[y - j][x] != Space.Empty:
					return False
			for j in r:
				state[y - j][x] = ship
		elif d == 3:
			for j in r:
				if state[y][x - j] != Space.Empty:
					return False
			for j in r:
				state[y][x - j] = ship
		return True

def setupShips(np_random):
		i = 0
		# state = [[Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10]
		state = np.full(shape=(10,10),fill_value=Space.Empty)
		# shipIds = [0,1,2,3,4]
		while (i < 5): #refactor outf
			# s = np_random.choice(shipIds)
			len_ = [5, 4, 3, 3, 2][i]#s
			ship = [Space.HiddenFive, Space.HiddenFour, Space.HiddenCruiser, Space.HiddenSub, Space.HiddenTwo][i]#s
			x = np_random.randint(0,9)
			y = np_random.randint(0,9)
			d = np_random.randint(0,3)
			
			if ((d % 2 == 1) and ((x - len_ < -1) or (x + len_) > 10)) or ((d % 2 == 0) and ((y - len_ < -1) or (y + len_) > 10)):
				continue
			elif (not addShip(state, ship, len_, x, y, d)): # could we add the ship, if not try again with new random coordinate
				continue
			# shipIds.remove(len_)
			i += 1
		return state

#game code
class Battleship1(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		self.action_space = spaces.Discrete(100)
		missesChannel = spaces.Tuple([
    		spaces.MultiDiscrete([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]) for _ in range(10)
		])
		regChannel = spaces.Tuple([
    		spaces.MultiDiscrete([3, 3, 3, 3, 3, 3, 3, 3, 3, 3]) for _ in range(10)
		])
		self.observation_space = spaces.Tuple((missesChannel,regChannel,regChannel,regChannel,regChannel,regChannel))
		self.seed()

		self.reset()

		# Action and observations spaces

	def _searchAndReplace(self, x, y, len_, search, replace,c):
		self.state[y][x] = replace
		direc = (-1,1)
		for d in direc:
			if (y+d) > 9 or (y+d) < 0:
				continue
			if len_ == 0:
				continue
			if self.state[y + d][x] == search:
				self.state[y + d][x] = replace
				self.hidState[c][y + d][x] = replace
				len_ -= 1
				for eLow in range(1,len_):
					e = eLow + 1
					if self.state[y + (d * e)][x] == search:
						self.state[y + (d * e)][x] = replace
						self.hidState[c][y + (d * e)][x] = replace
						len_ -= 1
					else:
						break
		for d in direc:
			if (x+d) > 9 or (x+d) < 0:
				continue
			if len_ == 0:
				continue
			if self.state[y][x + d] == search:
				self.state[y][x + d] = replace
				self.hidState[c][y][x + d] = replace
				len_ -= 1
				for eLow in range(1,len_):
					e = eLow + 1
					if self.state[y][x + (d*e)] == search:
						self.state[y][x + (d*e)] = replace
						self.hidState[c][y][x + (d*e)] = replace
						len_ -= 1
					else:
						break

	def step(self, target):
		x = target % 10
		y = target // 10
		targetSpace = self.state[y][x]
		self.reward = False
		# hit = False

		if self.done == True:
			print("Game Over")
			return [self.hidState, self.reward, self.done] #check return
		else:
			if targetSpace == Space.Empty:
				self.state[y][x] = Space.Miss
				self.hidState[0][y][x] = Space.Miss
			elif targetSpace == Space.HiddenTwo:
				self.reward = True
				self.state[y][x] = Space.HitPTwo
				self.hidState[1][y][x] = Space.HitPTwo
				self.hitsOnShips[4] = self.hitsOnShips[4] + 1
				if self.hitsOnShips[4] == 2:
					self._searchAndReplace(x, y, self.hitsOnShips[4], Space.HitPTwo, Space.SunkTwo, 1)
			elif targetSpace == Space.HiddenSub:
				self.reward = True
				self.state[y][x] = Space.HitPSub
				self.hidState[2][y][x] = Space.HitPSub
				self.hitsOnShips[3] = self.hitsOnShips[3] + 1
				if self.hitsOnShips[3] == 3:
					self._searchAndReplace(x, y, self.hitsOnShips[3], Space.HitPSub, Space.SunkSub, 2)
			elif targetSpace == Space.HiddenCruiser:
				self.reward = True
				self.state[y][x] = Space.HitPCruiser
				self.hidState[3][y][x] = Space.HitPCruiser
				self.hitsOnShips[2] = self.hitsOnShips[2] + 1
				if self.hitsOnShips[2] == 3:
					self._searchAndReplace(x, y, self.hitsOnShips[2], Space.HitPCruiser, Space.SunkCruiser, 3)
			elif targetSpace == Space.HiddenFour:
				self.reward = True
				self.state[y][x] = Space.HitPFour
				self.hidState[4][y][x] = Space.HitPFour
				self.hitsOnShips[1] = self.hitsOnShips[1] + 1
				if self.hitsOnShips[1] == 4:
					self._searchAndReplace(x, y, self.hitsOnShips[1], Space.HitPFour, Space.SunkFour, 4)
			elif targetSpace == Space.HiddenFive:
				self.reward = True
				self.state[y][x] = Space.HitPFive
				self.hidState[5][y][x] = Space.HitPFive
				self.hitsOnShips[0] = self.hitsOnShips[0] + 1
				if self.hitsOnShips[0] == 5:
					self._searchAndReplace(x, y, self.hitsOnShips[0], Space.HitPFive, Space.SunkFive, 5)
			else:
				# print("Misfire")
				self.done = True
				self.reward = False
				return [self.hidState, self.reward, self.done]
			self.counter += 1
		win = self.hitsOnShips == [5,4,3,3,2]
		if win:
			self.done = True
			print("Game over: ", self.counter, " moves.", sep = "", end = "\n")
			self.reward = 100 - self.counter
		return [self.hidState, self.reward, self.done]

	def reset(self):
		self.state = setupShips(self.np_random)
		self.hidState = np.full(shape=(6,10,10),fill_value=Space.Empty)
		
		self.hitsOnShips = [0, 0, 0, 0, 0]

		self.counter = 0
		self.done = False
		# self.add = [0, 0]
		self.reward = None
		return self.hidState
	
	def render(self, mode='human', close=False):
		ret = "0 "
		print("   0   1   2   3   4   5   6   7   8   9 ")
		i =  0
		for row in self.state:
			for slot in row:
				ret += slot.value[1] + " "
			print(ret)
			i += 1
			ret = str(i) + " "
		print()

	def seed(self, seed=None):
		self.np_random, seed = utils.seeding.np_random()
		return [seed]

env = Battleship1()
for i in range(20):
	env.step(i)
print(env.hidState)