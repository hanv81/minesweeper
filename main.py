import gym
import gym_minesweeper
from agent import *

rows = 10
cols = 10
mines = 10
env = gym.make('minesweeper-v0', rows=rows, cols=cols, mines=mines)
agent = Agent(rows, cols)