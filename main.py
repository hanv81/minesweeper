import gym
import gym_minesweeper
from agent import *
import numpy as np
import matplotlib.pyplot as plt

ROWS = 10
COLS = 10
MINES = 10
EPISODES = 500

EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

env = gym.make('minesweeper-v0', rows=ROWS, cols=COLS, mines=MINES)

def train():
    epsilon = 1
    agent = Agent(ROWS, COLS)

    total_point = 0
    x = []
    y = []
    max_point = 0
    max_avg = 0
    for episode in range(EPISODES):
        state = env.reset()
        point = 0
        done = False
        while not done:
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_q(state))
                r = action // COLS
                c = action % COLS
            else:
                (r,c) = env.action_space.sample()
                action = r*COLS + c
            next_state, reward, done = env.step((r,c))
            point += reward
            agent.update_replay_memory((state, action, reward, next_state, done))
            agent.train()

        agent.update_critic()
        if point > max_point:
            max_point = point
        total_point += point
        avg = total_point/(episode+1)
        if avg > max_avg:
            max_avg = avg
        x.append(episode+1)
        y.append(avg)
        print("episode %d %d %1.2f"%(episode+1, point, avg))

        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    plot(x, y, 'Train', 'train.png')
    return max_point, avg, max_avg

def play_random():
    total_point = 0
    x = []
    y = []
    max = 0
    max_avg = 0
    for episode in range(EPISODES):
        state = env.reset()
        point = 0
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done = env.step(action)
            point += reward

        if point > max:
            max = point
        total_point += point
        avg = total_point/(episode+1)
        if avg > max_avg:
            max_avg = avg
        x.append(episode+1)
        y.append(avg)
        print("episode %d %d"%(episode+1, point))

    plot(x, y, 'Random', 'random.png')
    return max, avg, max_avg

def plot(x, y, title, filename):
    plt.figure(figsize=(10,5))
    plt.scatter(x, y)
    plt.xlabel('Episode')
    plt.ylabel('Point')
    plt.title(title)
    plt.savefig(filename)

def main():
    max1, avg1, max_avg1 = play_random()
    max2, avg2, max_avg2 = train()
    print('Random %d %1.2f %1.2f'%( max1, avg1, max_avg1))
    print('Train %d %1.2f %1.2f'%( max2, avg2, max_avg2))

if __name__ == "__main__":
    main()