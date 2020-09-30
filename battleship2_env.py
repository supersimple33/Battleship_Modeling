from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.environments import utils

import enum
import numpy as np
# from random import randint seeding replaces

# indList = ("|-|","!M!","(2)","(S)","(C)","(4)","(5)",3"x2x","xSx","xCx","x4x","x5x","|2|","|S|","|C|","|4|","|5|","HiddenCruiser")
class Space(enum.Enum):
	Empty = -1,"|-|" #_

	Miss = 1,"!M!" #m
	
	HitPTwo = 1,"(2)"
	HitPSub = 1,"(S)"
	HitPCruiser = 1,"(C)"
	HitPFour = 1,"(4)"
	HitPFive = 1,"(5)"

	SunkTwo = 2,"x2x"
	SunkSub = 2,"xSx"
	SunkCruiser = 2,"xCx"
	SunkFour = 2,"x4x"
	SunkFive = 2,"x5x"

	HiddenTwo = 1,"|2|" # Need to update these values likely 1
	HiddenSub = 1,"|S|"
	HiddenCruiser = 1,"|C|"
	HiddenFour = 1,"|4|"
	HiddenFive = 1,"|5|"

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

def setupShips():
		i = 0
		# state = [[Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10]
		state = np.full(shape=(10,10),fill_value=Space.Empty)
		# shipIds = [0,1,2,3,4]
		while (i < 5): #refactor outf
			# s = np_random.choice(shipIds)
			len_ = [5, 4, 3, 3, 2][i]#s
			ship = [Space.HiddenFive, Space.HiddenFour, Space.HiddenCruiser, Space.HiddenSub, Space.HiddenTwo][i]#s
			x = np.random.randint(0,9)
			y = np.random.randint(0,9)
			d = np.random.randint(0,3)
			
			if ((d % 2 == 1) and ((x - len_ < -1) or (x + len_) > 10)) or ((d % 2 == 0) and ((y - len_ < -1) or (y + len_) > 10)):
				continue
			elif (not addShip(state, ship, len_, x, y, d)): # could we add the ship, if not try again with new random coordinate
				continue
			# shipIds.remove(len_)
			i += 1
		return state

#game code
class Battleship2(py_environment.PyEnvironment):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.float32, minimum=0, maximum=99, name='action')
		self._observation_spec = array_spec.BoundedArraySpec(shape=(10,10,6), dtype=np.float32, minimum=0, maximum=2, name='observation')
		# self.seed()

		self.reset()

	def action_spec(self): # Unused methods?
		return self._action_spec
	def observation_spec(self):
		return self._observation_spec

	def _searchAndReplace(self, x, y, len_, search, replace,c):
		self._state[y][x] = replace
		direc = (-1,1)
		for d in direc:
			if (y+d) > 9 or (y+d) < 0:
				continue
			if len_ == 0:
				continue
			if self._state[y + d][x] == search:
				self._state[y + d][x] = replace
				self.hidState[c][y + d][x] = replace.value[0]
				len_ -= 1
				for eLow in range(1,len_):
					e = eLow + 1
					if self._state[y + (d * e)][x] == search:
						self._state[y + (d * e)][x] = replace
						self.hidState[c][y + (d * e)][x] = replace.value[0]
						len_ -= 1
					else:
						break
		for d in direc:
			if (x+d) > 9 or (x+d) < 0:
				continue
			if len_ == 0:
				continue
			if self._state[y][x + d] == search:
				self._state[y][x + d] = replace
				self.hidState[c][y][x + d] = replace.value[0]
				len_ -= 1
				for eLow in range(1,len_):
					e = eLow + 1
					if self._state[y][x + (d*e)] == search:
						self._state[y][x + (d*e)] = replace
						self.hidState[c][y][x + (d*e)] = replace.value[0]
						len_ -= 1
					else:
						break

	def _step(self, target):
		target = int(target)
		x = target % 10
		y = target // 10
		targetSpace = self._state[y][x]
		hitCheck = False
		# hit = False

		if self._episode_ended == True:
			# print("Game Over")
			return self.reset()
		else:
			if targetSpace == Space.Empty:
				self._state[y][x] = Space.Miss
				self.hidState[0][y][x] = Space.Miss.value[0]
			elif targetSpace == Space.HiddenTwo:
				hitCheck = True
				self.expectedShots[target] = Space.Empty
				self._state[y][x] = Space.HitPTwo
				self.hidState[1][y][x] = Space.HitPTwo.value[0]
				self.hitsOnShips[4] = self.hitsOnShips[4] + 1
				if self.hitsOnShips[4] == 2:
					self._searchAndReplace(x, y, self.hitsOnShips[4], Space.HitPTwo, Space.SunkTwo, 1)
			elif targetSpace == Space.HiddenSub:
				hitCheck = True
				self.expectedShots[target] = Space.Empty
				self._state[y][x] = Space.HitPSub
				self.hidState[2][y][x] = Space.HitPSub.value[0]
				self.hitsOnShips[3] = self.hitsOnShips[3] + 1
				if self.hitsOnShips[3] == 3:
					self._searchAndReplace(x, y, self.hitsOnShips[3], Space.HitPSub, Space.SunkSub, 2)
			elif targetSpace == Space.HiddenCruiser:
				hitCheck = True
				self.expectedShots[target] = Space.Empty
				self._state[y][x] = Space.HitPCruiser
				self.hidState[3][y][x] = Space.HitPCruiser.value[0]
				self.hitsOnShips[2] = self.hitsOnShips[2] + 1
				if self.hitsOnShips[2] == 3:
					self._searchAndReplace(x, y, self.hitsOnShips[2], Space.HitPCruiser, Space.SunkCruiser, 3)
			elif targetSpace == Space.HiddenFour:
				hitCheck = True
				self.expectedShots[target] = Space.Empty
				self._state[y][x] = Space.HitPFour
				self.hidState[4][y][x] = Space.HitPFour.value[0]
				self.hitsOnShips[1] = self.hitsOnShips[1] + 1
				if self.hitsOnShips[1] == 4:
					self._searchAndReplace(x, y, self.hitsOnShips[1], Space.HitPFour, Space.SunkFour, 4)
			elif targetSpace == Space.HiddenFive:
				hitCheck = True
				self.expectedShots[target] = Space.Empty
				self._state[y][x] = Space.HitPFive
				self.hidState[5][y][x] = Space.HitPFive.value[0]
				self.hitsOnShips[0] = self.hitsOnShips[0] + 1
				if self.hitsOnShips[0] == 5:
					self._searchAndReplace(x, y, self.hitsOnShips[0], Space.HitPFive, Space.SunkFive, 5)
			else:
				# print("Misfire")
				self._episode_ended = True
				hitCheck = False
				return ts.termination(np.transpose(self.hidState), reward=-1.0) # change misfire reward
			self._counter += 1
		win = self.hitsOnShips == [5,4,3,3,2]
		if win:
			self._episode_ended = True
			# print("Game over: ", self._counter, " moves.", sep = "", end = "\n")
			# hitCheck = 100 - self._counter
		rew = 0.0 if hitCheck else -1.0
		if self._episode_ended:
			return ts.termination(np.transpose(self.hidState), reward=rew)
		else:
			return ts.transition(np.transpose(self.hidState), reward=rew, discount=1.0)
		# return [self.hidState, hitCheck, self._episode_ended, self.expectedShots]

	def _reset(self):
		self._state = setupShips()
		self.hidState = np.full(shape=(6,10,10),fill_value=0,dtype=np.float32)
		self.expectedShots = np.reshape(self._state, (100))
		
		self.hitsOnShips = [0, 0, 0, 0, 0]

		self._counter = 0
		self._episode_ended = False
		# self.add = [0, 0]
		return ts.restart(np.transpose(self.hidState))
	
	def _render(self, mode='human', close=False):
		ret = "0 "
		print("   0   1   2   3   4   5   6   7   8   9 ")
		i =  0
		for row in self._state:
			for slot in row:
				ret += slot.value[1] + " "
			print(ret)
			i += 1
			ret = str(i) + " "
		print()