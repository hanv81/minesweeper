import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Discrete,Tuple,Box,MultiDiscrete
import numpy as np
import random
import sys

class Minesweeper(gym.Env):
	metadata = {'render.modes': ['human']}
	UNKNOWN = -1
	MINE = -99

	def __init__(self, rows=10, cols=10, mines=10):
		
		self.action_space = Tuple((Discrete(rows),Discrete(cols)))
		self.observation_space = Box(low= -1, high=8, shape=(rows, cols), dtype=np.uint8)
		self.rows = rows
		self.cols = cols
		self.mines = mines
		self.nonMines = rows*cols - mines
		self.clickedCoords = set()
		self.letter_Axis = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']
		self.chosenCoords = []
		self.state = np.full([self.rows, self.cols], Minesweeper.UNKNOWN)
		self.info = dict()
		self.coordcount = dict()

	def conCoord(self, userInput):
		# rows x cols
		self.cc = userInput
		firstVal = self.letter_Axis[self.cc[0]]
		x = str(firstVal)
		y = str(self.cc[1])
		xy = x + y
		return xy

	def scanCoord(self, coord):
		list = []
		neighboring_mines = 0
		for r in range(coord[0]-1, coord[0]+2):
		    for c in range(coord[1]-1, coord[1]+2):
		        if (r,c) in self.mine_coords:
		            neighboring_mines += 1
		self.state[coord] = neighboring_mines

		if neighboring_mines == 0:
			for r in range(coord[0]-1, coord[0]+2):
				for c in range(coord[1]-1, coord[1]+2):
					if r != coord[0] or c != coord[1]:
						if 0<=r<self.rows and 0<=c<self.cols and self.state[(r,c)] == -1:
							self.scanCoord((r,c))

	def step(self, coord):
		done = False
		reward = 0
		self.coord = coord
		if coord in self.clickedCoords:
		    reward -=  1

		elif coord in self.mine_coords:
		    # Clicked on a mine!
		    self.state[coord] = Minesweeper.MINE
		    self.clickedCoords.add(coord)
		    done = True
		else:
			reward += 1
			self.scanCoord(coord)
			self.coords_to_clear -= 1
			if self.coords_to_clear == 0:
				# Win !!!
				done = True
			else:
				self.clickedCoords.add(coord)

		return (self.state, reward, done, self.info)

	def reset(self):
		# Internal state: where are all the mines?
		self.mine_coords = set()
		mines_to_place = self.mines
		while mines_to_place > 0:
		    r = random.randrange(self.rows)
		    c = random.randrange(self.cols)
		    if (r, c) not in self.mine_coords:  # new coord
		        self.mine_coords.add((r, c))
		        mines_to_place -= 1
		print("MINE locations:", self.mine_coords)
		self.state = np.full([self.rows, self.cols], Minesweeper.UNKNOWN)
		self.coords_to_clear = self.rows * self.cols - self.mines
		self.clickedCoords = set()
		return self.state

	def render(self):
		for x in range(self.rows):
			sys.stdout.write(self.letter_Axis[x])
			for y in range(self.cols):
				if self.state[x][y] == Minesweeper.MINE:
					sys.stdout.write(' x')
				elif self.state[x][y] == -1:
					sys.stdout.write(' .')
				elif self.state[x][y] == 0:
					sys.stdout.write('  ')
				else:
					sys.stdout.write(' %s' % int(self.state[x][y]))
				if y != self.cols-1:
					sys.stdout.write(' ')
					if y == (self.cols - 1):
						sys.stdout.write('\n')
			sys.stdout.write('\n')
		sys.stdout.write(' ')
		for k in range(self.cols):
			sys.stdout.write(' %s ' % k)
		print ("")
