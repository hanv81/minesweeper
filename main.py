import gym
import gym_minesweeper
from agent import Agent
import numpy as np
import random
import matplotlib.pyplot as plt

ROWS = 10
COLS = 10
MINES = 10
EPISODES = 100
EPISODES_TEST = 1000

EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

env = gym.make('minesweeper-v0', rows=ROWS, cols=COLS, mines=MINES)

def train(cnn=False):
    epsilon = 1
    agent = Agent()
    agent.create_model(ROWS, COLS, cnn)

    y = []
    p = []
    for episode in range(EPISODES):
        state = env.reset()
        point = 0
        done = False
        clicked_cells = []
        cells_to_click = [x for x in range(0, ROWS * COLS)]
        while not done:
            action = action_policy(agent, state, point, cells_to_click, clicked_cells, epsilon)
            r = action // COLS
            c = action % COLS
            next_state, reward, done, info = env.step((r,c))
            agent.step((state, action, reward, next_state, done))
            if reward > 0:
                point += reward
                for (r,c) in info:
                    action = r * COLS + c
                    clicked_cells.append(action)
                    cells_to_click.remove(action)
            state = next_state

        p.append(point)
        avg = sum(p)/(episode+1)
        y.append(avg)
        
        if (episode + 1) % 10 == 0:
            print("episode %d %d %1.2f"%(episode+1, point, avg))

        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    save_model(agent, cnn)

    return p, y

def action_policy(agent, state, point, cells_to_click, clicked_cells, epsilon):
    if random.random() <= epsilon or point == 0: # first cell -> just random
        return random.sample(cells_to_click, 1)[0]
    else:
        qs = agent.predict(state)[0]
        for cell in clicked_cells:
            qs[cell] = np.min(qs)
        if np.max(qs) > np.min(qs):
            return np.argmax(qs)

    return random.sample(cells_to_click, 1)[0]    # no max action, just random

def save_model(agent, cnn):
    filename = 'model_' + str(ROWS) + '_' + str(COLS) + '_' + str(MINES) + ('_cnn.h5' if cnn else '.h5')
    agent.save_model(filename)

def play_random(episodes):
    y = []
    p = []
    for episode in range(episodes):
        env.reset()
        point = 0
        done = False
        cells_to_click = [x for x in range(0, ROWS * COLS)]
        while not done:
            action = random.sample(cells_to_click, 1)[0]
            r = action // COLS
            c = action % COLS
            next_state, reward, done, info = env.step((r,c))
            if reward > 0:
                point += reward
                for (r,c) in info:
                    action = r * COLS + c
                    cells_to_click.remove(action)

        p.append(point)
        avg = sum(p)/(episode+1)
        y.append(avg)

        if (episode + 1) % 100 == 0:
            print("episode %d %d"%(episode+1, point))

    return p, y

def plot(random_avg, train_avg, train_cnn_avg):
    x = [xi for xi in range(1, EPISODES+1)]
    plt.figure(figsize=(15,10))
    plt.xlabel('Episode')
    plt.ylabel('Point')
    plt.plot(x, random_avg)
    plt.plot(x, train_avg)
    plt.plot(x, train_cnn_avg)
    plt.legend(['Random','DQN','DQCNN'])
    plt.title('Average Point')
    plt.savefig('train')

def plot_test(random_avg, dnn_no_heu_avg, dnn_heu_avg, cnn_no_heu_avg, cnn_heu_avg):
    x = [xi for xi in range(1, EPISODES_TEST+1)]
    plt.figure(figsize=(15,10))
    plt.xlabel('Episode')
    plt.ylabel('Point')
    plt.plot(x, random_avg)
    plt.plot(x, dnn_no_heu_avg)
    plt.plot(x, dnn_heu_avg)
    plt.plot(x, cnn_no_heu_avg)
    plt.plot(x, cnn_heu_avg)
    plt.legend(['Random','DQN','DQN Heuristic', 'DQCNN','DQCNN Heuristic'])
    plt.title('Average Point')
    plt.savefig('test')

def main():
    p1, avg1 = play_random(EPISODES)
    p2, avg2 = train(cnn=False)
    p3, avg3 = train(cnn=True)
    print('------------------ SUMMARY ------------------')
    print('RANDOM:  max %d avg %1.2f max_avg %1.2f'%( max(p1), avg1[-1], max(avg1)))
    print('DQN:     max %d avg %1.2f max_avg %1.2f'%( max(p2), avg2[-1], max(avg2)))
    print('DQCNN:   max %d avg %1.2f max_avg %1.2f'%( max(p3), avg3[-1], max(avg3)))
    plot(avg1, avg2, avg3)

def test_agent():
    p, avg = play_random(EPISODES_TEST)
    agent = Agent()
    agent_cnn = Agent()
    agent.load_model('./minesweeper/model/model_' + str(ROWS) + '_' + str(COLS) + '_' + str(MINES) + '.h5')
    agent_cnn.load_model('./minesweeper/model/model_' + str(ROWS) + '_' + str(COLS) + '_' + str(MINES) + '_cnn.h5')
    p1, avg1, win1 = test(agent)
    p2, avg2, win2 = test(agent, heuristic=True)
    p3, avg3, win3 = test(agent_cnn)
    p4, avg4, win4 = test(agent_cnn, heuristic=True)
    print('------------------ SUMMARY ------------------')
    print('RANDOM:              max %d avg %1.2f max_avg %1.2f' % (max(p), avg[-1], max(avg)))
    print('DNN NO HEURISTIC:    max %d avg %1.2f max_avg %1.2f win %d' % (max(p1), avg1[-1], max(avg1), win1))
    print('DNN HEURISTIC:       max %d avg %1.2f max_avg %1.2f win %d' % (max(p2), avg2[-1], max(avg2), win2))
    print('CNN NO HEURISTIC:    max %d avg %1.2f max_avg %1.2f win %d' % (max(p3), avg3[-1], max(avg3), win3))
    print('CNN HEURISTIC:       max %d avg %1.2f max_avg %1.2f win %d' % (max(p4), avg4[-1], max(avg4), win4))
    plot_test(avg, avg1, avg2, avg3, avg4)

def test(agent, heuristic=False):
    y = []
    p = []
    win = 0
    for episode in range(EPISODES_TEST):
        state = env.reset()
        point = 0
        done = False
        clicked_cells = []
        cells_to_click = [x for x in range(0, ROWS * COLS)]
        while not done:
            if point == 0: # first cell -> just random
                action = random.randint(0, ROWS * COLS - 1)
            else:
                mine_cells = []
                if heuristic:
                    for i in range(ROWS):
                        for j in range(COLS):
                            if state[i,j] > 0:
                                neibors = []
                                neibors_flag = []
                                for r in range(i-1, i+2):
                                    for c in range(j-1, j+2):
                                        if 0<=r<ROWS and 0<=c<COLS:
                                            if state[r,c] < 0:
                                                pos = r * COLS + c
                                                if pos in mine_cells:
                                                    neibors_flag.append(pos)
                                                else:
                                                    neibors.append(pos)
                                if state[i,j] == len(neibors) + len(neibors_flag):
                                    for n in neibors:
                                        mine_cells.append(n)
                qs = agent.predict(state)[0]
                for cell in range(ROWS*COLS):
                    if cell in clicked_cells or cell in mine_cells:
                        qs[cell] = np.min(qs)
                if np.max(qs) > np.min(qs):
                    action = np.argmax(qs)
                else:
                    action = random.sample(cells_to_click, 1)[0]
            r = action // COLS
            c = action % COLS
            next_state, reward, done, info = env.step((r,c))
            if reward > 0:
                if done:
                    win += 1
                point += reward
                for (r,c) in info:
                    action = r * COLS + c
                    clicked_cells.append(action)
                    cells_to_click.remove(action)
            state = next_state

        p.append(point)
        avg = sum(p)/(episode+1)
        y.append(avg)
        
        if (episode + 1) % 100 == 0:
            print("episode %d %d %1.2f"%(episode+1, point, avg))

    return p, y, win

if __name__ == "__main__":
    main()
    # test_agent()