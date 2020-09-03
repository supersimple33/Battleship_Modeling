import gym
from gym import error, spaces, utils

import enum
# from random import randint seeding replaces

class Space(enum.Enum):
	Empty = "|-|" #_

	Miss = "!M!" #m
	
	HitPTwo = "(2)"
	HitPSub = "(S)"
	HitPCruiser = "(C)"
	HitPFour = "(4)"
	HitPFive = "(5)"

	SunkTwo = "x2x"
	SunkSub = "xSx"
	SunkCruiser = "xCx"
	SunkFour = "x4x"
	SunkFive = "x5x"

	HiddenTwo = "|2|"
	HiddenSub = "|S|"
	HiddenCruiser = "|C|"
	HiddenFour = "|4|"
	HiddenFive = "|5|"

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
		state = [[Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10]
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
		self.observation_space = spaces.Tuple([
    		spaces.MultiDiscrete([12, 12, 12, 12, 12, 12, 12, 12, 12, 12]) for _ in range(10)
		])
		self.seed()

		# self.state = setupShips(self.np_random)
		# self.hidState = [[Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10]

		# self.counter = 0
		# self.done = 0
		# # self.add = [0, 0]
		# self.reward = 0
		self.reset()

		# Action and observations spaces

	def _searchAndReplace(self, x, y, len_, search, replace):
		self.state[y][x] = replace
		direc = (-1,1)
		for d in direc:
			if len_ == 0:
				continue
			if self.state[y + d][x] == search:
				self.state[y + d][x] = replace
				self.hidState[y + d][x] = replace
				len_ -= 1
				for eLow in range(1,len_):
					e = eLow + 1
					if self.state[y + (d * e)][x] == search:
						self.state[y + (d * e)][x] = replace
						self.hidState[y + (d * e)][x] = replace
						len_ -= 1
					else:
						break
		for d in direc:
			if len_ == 0:
				continue
			if self.state[y][x + d] == search:
				self.state[y][x + d] = replace
				self.hidState[y][x + d] = replace
				len_ -= 1
				for eLow in range(1,len_):
					e = eLow + 1
					if self.state[y][x + (d*e)] == search:
						self.state[y][x + (d*e)] = replace
						self.hidState[y][x + (d*e)] = replace
						len_ -= 1
					else:
						break

	def step(self, target):
		x = target % 10
		y = target // 10
		targetSpace = self.state[y][x]
		hit = False

		if self.done == 1:
			print("Game Over")
			return [self.hidState, self.reward, self.done, (True, hit)] #check return
		else:
			if targetSpace == Space.Empty:
				self.state[y][x] = Space.Miss
				self.hidState[y][x] = Space.Miss
			elif targetSpace == Space.HiddenTwo:
				hit = True
				self.state[y][x] = Space.HitPTwo
				self.hidState[y][x] = Space.HitPTwo
				self.hitsOnShips[4] = self.hitsOnShips[4] + 1
				if self.hitsOnShips[4] == 2:
					self._searchAndReplace(x, y, self.hitsOnShips[4], Space.HitPTwo, Space.SunkTwo)
			elif targetSpace == Space.HiddenSub:
				hit = True
				self.state[y][x] = Space.HitPSub
				self.hidState[y][x] = Space.HitPSub
				self.hitsOnShips[3] = self.hitsOnShips[3] + 1
				if self.hitsOnShips[3] == 3:
					self._searchAndReplace(x, y, self.hitsOnShips[3], Space.HitPSub, Space.SunkSub)
			elif targetSpace == Space.HiddenCruiser:
				hit = True
				self.state[y][x] = Space.HitPCruiser
				self.hidState[y][x] = Space.HitPCruiser
				self.hitsOnShips[2] = self.hitsOnShips[2] + 1
				if self.hitsOnShips[2] == 3:
					self._searchAndReplace(x, y, self.hitsOnShips[2], Space.HitPCruiser, Space.SunkCruiser)
			elif targetSpace == Space.HiddenFour:
				hit = True
				self.state[y][x] = Space.HitPFour
				self.hitsOnShips[1] = self.hitsOnShips[1] + 1
				if self.hitsOnShips[1] == 4:
					self._searchAndReplace(x, y, self.hitsOnShips[1], Space.HitPFour, Space.SunkFour)
			elif targetSpace == Space.HiddenFive:
				hit = True
				self.state[y][x] = Space.HitPFive
				self.hitsOnShips[0] = self.hitsOnShips[0] + 1
				if self.hitsOnShips[0] == 5:
					self._searchAndReplace(x, y, self.hitsOnShips[0], Space.HitPFive, Space.SunkFive)
			else:
				# print("Misfire")
				self.done = 1
				self.reward = 0
				return [self.hidState, self.reward, self.done, (False, hit)]
			self.counter += 1
		win = self.hitsOnShips == [5,4,3,3,2]
		if win:
			self.done = 1
			print("Game over: ", self.counter, " moves.", sep = "", end = "\n")
			self.reward = 100 - self.counter
		return [self.hidState, self.reward, self.done, (True, hit)]

	def reset(self):
		self.state = setupShips(self.np_random)
		self.hidState = [[Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10]
		
		self.hitsOnShips = [0, 0, 0, 0, 0]

		self.counter = 0
		self.done = 0
		# self.add = [0, 0]
		self.reward = None
		return self.hidState
	
	def render(self, mode='human', close=False):
		ret = "0 "
		print("   0   1   2   3   4   5   6   7   8   9 ")
		i =  0
		for row in self.state:
			for slot in row:
				ret += slot.value + " "
			print(ret)
			i += 1
			ret = str(i) + " "
		print()

	def seed(self, seed=None):
		self.np_random, seed = utils.seeding.np_random()
		return [seed]