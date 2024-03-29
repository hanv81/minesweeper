import gym
from gym.spaces import Discrete, Box
import numpy as np
import random
import sys

UNKNOWN = -1
MINE = -99

class Minesweeper(gym.Env):

	def __init__(self, rows=10, cols=10, mines=10):
		
		self.action_space = Discrete(rows*cols)
		self.observation_space = Box(low= -1, high=8, shape=(rows, cols), dtype=np.uint8)
		self.rows = rows
		self.cols = cols
		self.mines = mines
		self.clickedCoords = set()
		self.letter_Axis = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']
		self.state = np.full([self.rows, self.cols], UNKNOWN)
		self.info = dict()

	def scanCoord(self, coord):
		cells = [coord]
		neighboring_mines = 0
		for r in range(coord[0]-1, coord[0]+2):
			for c in range(coord[1]-1, coord[1]+2):
				if (r,c) in self.mine_coords:
					neighboring_mines += 1
		self.state[coord] = neighboring_mines
		self.coords_to_clear -= 1

		if neighboring_mines == 0:
			r,c = coord[0], coord[1]
			rc = ((r, c+1), (r, c-1), (r+1, c), (r+1, c+1), (r+1, c-1), (r-1, c), (r-1, c+1), (r-1, c-1))
			for r,c in rc:
				if 0 <= r < self.rows and 0 <= c < self.cols and self.state[r, c] == UNKNOWN:
					cells.extend(self.scanCoord((r, c)))
		return cells

	def step(self, action):
		coord = (action // self.cols, action % self.cols)
		done = False
		reward = 0
		if coord in self.mine_coords:
			reward = -10
			self.state[coord] = MINE
			self.clickedCoords.add(coord)
			done = True
		elif coord not in self.clickedCoords:
			reward = 1
			self.info['coord'] = self.scanCoord(coord)
			if self.coords_to_clear == 0:
				# Win !!!
				done = True
			else:
				self.clickedCoords.add(coord)

		return np.copy(self.state), reward, done, self.info

	def reset(self):
		# Internal state: where are all the mines?
		self.mine_coords = set()
		mines = random.sample(range(self.rows*self.cols), self.mines)
		for m in mines:
			self.mine_coords.add((m // self.cols, m % self.cols))

		# print("MINE locations:", self.mine_coords)
		self.state = np.full([self.rows, self.cols], UNKNOWN)
		self.coords_to_clear = self.rows * self.cols - self.mines
		self.clickedCoords = set()
		return np.copy(self.state)

	def render(self):
		for x in range(self.rows):
			sys.stdout.write(self.letter_Axis[x])
			for y in range(self.cols):
				if self.state[x,y] == MINE:
					sys.stdout.write(' x')
				elif self.state[x,y] == UNKNOWN:
					sys.stdout.write(' .')
				elif self.state[x,y] == 0:
					sys.stdout.write('  ')
				else:
					sys.stdout.write(' %s' % int(self.state[x,y]))
				if y != self.cols-1:
					sys.stdout.write(' ')
					if y == (self.cols - 1):
						sys.stdout.write('\n')
			sys.stdout.write('\n')
		sys.stdout.write(' ')
		for k in range(self.cols):
			sys.stdout.write(' %s ' % k)
		print ("")
