import gym
import gym_minesweeper
from agent import *
import numpy as np
import matplotlib.pyplot as plt

ROWS = 10
COLS = 10
MINES = 10
EPISODES = 1000

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

def play_random():
    total_point = 0
    x = []
    y = []
    max = 0
    max_avg = 0
    for episode in range(EPISODES):
        state = env.reset()
        point = 0
        while True:
            action = env.action_space.sample()
            (next_state, reward, done, info) = env.step(action)
            point += reward
            if done:
                if point > max:
                    max = point
                total_point += point
                avg = total_point/(episode+1)
                if avg > max_avg:
                    max_avg = avg
                x.append(episode+1)
                y.append(avg)
                print("episode %d %d"%(episode+1, point))
                break
    print('max', max)
    print('avg %1.2f', avg)
    print('max avg %1.2f', max_avg)
    plt.figure(figsize=(10,5))
    plt.scatter(x, y)
    plt.xlabel('Episode')
    plt.ylabel('Point')
    plt.title('Random')
    plt.show()

def main():
    play_random()

if __name__ == "__main__":
    main()