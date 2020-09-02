import enum
from random import randint

class Space(enum.Enum):
	Empty = "_" #_
	Miss = "m" #m
	ShipHidden = "h" #h
	ShipHit = "x" #x

class GameBoard:
	board = [[Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10, [Space.Empty]*10]

	def __init__(self):
		i = 0
		while (i < 5):
			len = [5, 4, 3, 3, 2][i]
			x = randint(0,9)
			y = randint(0,9)
			#d = randint(0,3)
			d = 0
			if ((d % 2 == 1) and ((x - len < -1) or (x + len) > 10)) or ((d % 2 == 0) and ((y - len < -1) or (y + len) > 10)):
				continue
			elif (not self.addShip(len, x, y, d)):
				continue
			i += 1

	def addShip(self, len, x, y, d):
		r = range(0, len)
		print(self.boardRep())
		if d == 0:
			for j in r:
				if self.board[y + j][x] != Space.Empty: # loop run twice in order to make sure all spaces are clear before making modifications
					return False
			for j in r:
				self.board[y + j][x] = Space.ShipHidden
		elif d == 1:
			for j in r:
				if self.board[y][x + j] != Space.Empty:
					return False
			for j in r:
				self.board[y][x + j] = Space.ShipHidden
		elif d == 2:
			for j in r:
				if self.board[y - j][x] != Space.Empty:
					return False
			for j in r:
				self.board[y - j][x] = Space.ShipHidden
		elif d == 3:
			for j in r:
				if self.board[y][x - j] != Space.Empty:
					return False
			for j in r:
				self.board[y][x - j] = Space.ShipHidden
		return True

	def boardRep(self):
		ret = ""
		for row in self.board:
			for slot in row:
				ret += slot.value + " "
			ret += "\n"
		return ret
