import gym
from gym import error, spaces, utils
from gym.utils import seeding

import enum
from random import randint

class Space(enum.Enum):
	Empty = "|-|" #_
	Miss = "!M!" #m
	
	HiddenTwo = "|2|"
	HiddenSub = "|S|"
	HiddenCruiser = "|C|"
	HiddenFour = "|4|"
	HiddenFive = "|5|"
	
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

#game code
class Battleship1(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		self.state = [[Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10]
		i = 0
		while (i < 5):
			len = [5, 4, 3, 3, 2][i]
			ship = [Space.HiddenFive, Space.HiddenFour, Space.HiddenCruiser, Space.HiddenSub, Space.HiddenTwo][i]
			x = randint(0,9)
			y = randint(0,9)
			#d = randint(0,3)
			d = 0
			if ((d % 2 == 1) and ((x - len < -1) or (x + len) > 10)) or ((d % 2 == 0) and ((y - len < -1) or (y + len) > 10)):
				continue
			elif (not self.addShip(ship, len, x, y, d)):
				continue
			i += 1

		self.counter = 0
		self.done = 0
		# self.add = [0, 0]
		self.reward = 0
	
	def check(self):
		if self.counter < 16:
			return 0
		elif not (self.checkForSpace(Space.HiddenTwo) or self.checkForSpace(Space.HiddenSub) or self.checkForSpace(Space.HiddenCruiser) or self.checkForSpace(Space.HiddenFour) or self.checkForSpace(Space.HiddenFive)):
			return 1
		return 0

	def checkForSpace(self, sp):
		for row in self.state:
			if sp in row:
				return True
		return False

	def step(self, target):
		x = target % 10
		y = int(target / 10)
		targetSpace = self.state[y][x]
		hit = False

		if self.done == 1:
			print("Game Over")
			return [self.hidState(), self.reward, self.done, True, hit] #check return
		else:
			if targetSpace == Space.Empty:
				self.state[y][x] = Space.Miss
			elif targetSpace == Space.HiddenTwo:
				hit = True
				self.state[y][x] = Space.HitPTwo
				if not self.checkForSpace(Space.HiddenTwo):
					for row in range(10):
						for col in range(10):
							if self.state[row][col] == Space.HitPTwo:
								self.state[row][col] = Space.SunkTwo
			elif targetSpace == Space.HiddenSub:
				hit = True
				self.state[y][x] = Space.HitPSub
				if not self.checkForSpace(Space.HiddenSub):
					for row in range(10):
						for col in range(10):
							if self.state[row][col] == Space.HitPSub:
								self.state[row][col] = Space.SunkSub
			elif targetSpace == Space.HiddenCruiser:
				hit = True
				self.state[y][x] = Space.HitPCruiser
				if not self.checkForSpace(Space.HiddenCruiser):
					for row in range(10):
						for col in range(10):
							if self.state[row][col] == Space.HitPCruiser:
								self.state[row][col] = Space.SunkCruiser
			elif targetSpace == Space.HiddenFour:
				hit = True
				self.state[y][x] = Space.HitPFour
				if not self.checkForSpace(Space.HiddenFour):
					for row in range(10):
						for col in range(10):
							if self.state[row][col] == Space.HitPFour:
								self.state[row][col] = Space.SunkFour
			elif targetSpace == Space.HiddenFive:
				hit = True
				self.state[y][x] = Space.HitPFive
				if not self.checkForSpace(Space.HiddenFive):
					for row in range(10):
						for col in range(10):
							if self.state[row][col] == Space.HitPFive:
								self.state[row][col] = Space.SunkFive
			else:
				# print("Misfire")
				return [self.hidState(), self.reward, self.done, False, hit]
			self.counter += 1
		win = self.check()
		if win:
			self.done = 1
			print("Game over: ", self.counter, " moves.", sep = "", end = "\n")
			self.reward = 100 - self.counter
		return [self.hidState(), self.reward, self.done, True, hit]

	def reset(self):
		self.state = [[Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10]
		i = 0
		while (i < 5):
			len = [5, 4, 3, 3, 2][i]
			ship = [Space.HiddenFive, Space.HiddenFour, Space.HiddenCruiser, Space.HiddenSub, Space.HiddenTwo][i]
			x = randint(0,9)
			y = randint(0,9)
			#d = randint(0,3)
			d = 0
			if ((d % 2 == 1) and ((x - len < -1) or (x + len) > 10)) or ((d % 2 == 0) and ((y - len < -1) or (y + len) > 10)):
				continue
			elif (not self.addShip(ship, len, x, y, d)):
				continue
			i += 1
		
		self.counter = 0
		self.done = 0
		# self.add = [0, 0]
		self.reward = 0
		return self.hidState()
	
	def render(self, mode='human', close=False):
		ret = ""
		for row in self.state:
			for slot in row:
				ret += slot.value + " "
			print(ret)
			ret = ""
		print()
	
	
	def hidState(self): #map without hidden
		ret = [None]*100
		for y in range(10):
			for x in range(10):
				elem = self.state[y][x]
				if ((elem == Space.HiddenTwo) or (elem == Space.HiddenSub) or (elem == Space.HiddenCruiser) or (elem == Space.HiddenFour) or (elem == Space.HiddenFive) or (elem == Space.Empty)):
					ret[(y * 10) + x] = 0
				elif elem == Space.Miss:
					ret[(y * 10) + x] = -1
				elif ((elem == Space.HitPTwo) or (elem == Space.HitPSub) or (elem == Space.HitPCruiser) or (elem == Space.HitPFour) or (elem == Space.HitPFive)):
					ret[(y * 10) + x] = 1
				elif elem == Space.SunkTwo:
					ret[(y * 10) + x] = 11
				elif elem == Space.SunkSub:
					ret[(y * 10) + x] = 22
				elif elem == Space.SunkCruiser:
					ret[(y * 10) + x] = 33
				elif elem == Space.SunkFour:
					ret[(y * 10) + x] = 44
				elif elem == Space.SunkFive:
					ret[(y * 10) + x] = 55
		return ret

	def addShip(self, ship, len, x, y, d):
		r = range(0, len)
		# print(self.boardRep())
		if d == 0:
			for j in r:
				if self.state[y + j][x] != Space.Empty: # loop run twice in order to make sure all spaces are clear before making modifications
					return False
			for j in r:
				self.state[y + j][x] = ship
		elif d == 1:
			for j in r:
				if self.state[y][x + j] != Space.Empty:
					return False
			for j in r:
				self.state[y][x + j] = ship
		elif d == 2:
			for j in r:
				if self.state[y - j][x] != Space.Empty:
					return False
			for j in r:
				self.state[y - j][x] = ship
		elif d == 3:
			for j in r:
				if self.state[y][x - j] != Space.Empty:
					return False
			for j in r:
				self.state[y][x - j] = ship
		return True

