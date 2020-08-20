import gym
import gym_minesweeper
from agent import *
import numpy as np

ROWS = 10
COLS = 10
MINES = 10
EPISODES = 100

EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

env = gym.make('minesweeper-v0', rows=ROWS, cols=COLS, mines=MINES)

def train():
    epsilon = 1
    agent = Agent(ROWS, COLS)

    total_point = 0
    for episode in range(EPISODES):
        state = env.reset()
        point = 0
        while True:
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_q(state))
                r = action // ROWS
                c = action % COLS
            else:
                (r,c) = env.action_space.sample()
                action = r*c
            (next_state, reward, done, info) = env.step((r,c))
            point += reward
            agent.update(state, action, reward, next_state, done)
            if done:
                total_point += point
                env.render()
                print('episode ', episode+1, point)
                print('avg point', total_point/(episode+1))
                break

        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

def main():
    train()

if __name__ == "__main__":
    main()