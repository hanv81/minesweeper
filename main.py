import gym
import gym_minesweeper
from agent import *
import numpy as np
import matplotlib.pyplot as plt

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

    y = []
    p = []
    for episode in range(EPISODES):
        state = env.reset()
        point = 0
        done = False
        clicked_cells = []
        cells_to_click = [x for x in range(0, ROWS * COLS)]
        while not done:
            if np.random.random() > epsilon:
                qs = agent.get_q(state)
                for cell in clicked_cells:
                    qs[0, cell] = np.min(qs)
                action = np.argmax(qs)
            else:
                action = random.sample(cells_to_click, 1)[0]
            r = action // COLS
            c = action % COLS
            clicked_cells.append(action)
            cells_to_click.remove(action)
            next_state, reward, done = env.step((r,c))
            point += reward
            agent.update_replay_memory((state, action, reward, next_state, done))
            agent.train()

        agent.update_critic()

        p.append(point)
        avg = sum(p)/(episode+1)
        y.append(avg)
        
        if (episode + 1) % 10 == 0:
            print("episode %d %d %1.2f"%(episode+1, point, avg))

        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    agent.save_model()
    return p, y

def play_random():
    y = []
    p = []
    for episode in range(EPISODES):
        state = env.reset()
        point = 0
        done = False
        cells_to_click = [x for x in range(0, ROWS * COLS)]
        while not done:
            action = random.sample(cells_to_click, 1)[0]
            r = action // COLS
            c = action % COLS
            next_state, reward, done = env.step((r,c))
            cells_to_click.remove(action)
            point += reward

        p.append(point)
        avg = sum(p)/(episode+1)
        y.append(avg)

        if (episode + 1) % 100 == 0:
            print("episode %d %d"%(episode+1, point))

    return p, y

def plot(random_p, random_avg, train_p, train_avg):
    x = [xi for xi in range(1, EPISODES+1)]
    plt.figure(figsize=(15,10))
    plt.xlabel('Episode')
    plt.ylabel('Point')
    plt.plot(x, random_avg)
    plt.plot(x, train_avg)
    text = 'Max random ' + str(max(random_p)) + '\nMax train ' + str(max(train_p)) + '\nAvg random ' + str(random_avg[-1]) + '\nAvg train ' + str(train_avg[-1])
    plt.text(1, 10, text)
    plt.legend(['Random','Train'])
    plt.title('Average Point')
    plt.savefig('minesweeper')

def main():
    p1, avg1 = play_random()
    p2, avg2 = train()
    print('Random %d %1.2f %1.2f'%( max(p1), avg1[-1], max(avg1)))
    print('Train %d %1.2f %1.2f'%( max(p2), avg2[-1], max(avg2)))
    plot(p1, avg1, p2, avg2)

if __name__ == "__main__":
    main()