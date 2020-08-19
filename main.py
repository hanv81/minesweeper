import gym
import gym_minesweeper
from agent import *

ROWS = 10
COLS = 10
MINES = 10
EPISODES = 100
env = gym.make('minesweeper-v0', rows=ROWS, cols=COLS, mines=MINES)
agent = Agent(ROWS, COLS)

total_point = 0
for episode in range(EPISODES):
    state = env.reset()
    point = 0
    while True:
        action = env.action_space.sample()
        (next_state, reward, done, info) = env.step(action)
        point += reward
        agent.update(state, action, reward, next_state, done)
        if done:
            break
    print('episode ', episode, point)
    total_point += point
print('avg point', total_point/episode)
