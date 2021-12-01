import gym
import gym_minesweeper  # must import for create env
from agent.DQN import DQN
from agent.PG import PG
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse

ROWS = 10
COLS = 10
MINES = 10

EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

env = gym.make('minesweeper-v0', rows=ROWS, cols=COLS, mines=MINES)

def trainPG(episodes, cnn=False):
    agent = PG()
    agent.create_model(ROWS, COLS, cnn)
    p, avg = agent.train(episodes, env)
    p1, avg1 = play_random(episodes)
    print('------------------ SUMMARY ------------------')
    print('RANDOM:  max %d avg %1.2f max_avg %1.2f'%( max(p1), avg1[-1], max(avg1)))
    print('PG:      max %d avg %1.2f max_avg %1.2f'%( max(p), avg[-1], max(avg)))
    plot(avg1, avg)

def train(episodes, cnn=False):
    epsilon = 1
    agent = DQN()
    agent.create_model(ROWS, COLS, cnn)

    avg = []
    pts = []
    for episode in range(episodes):
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

        agent.update_target()
        pts.append(point)
        avg.append(np.mean(pts))
        
        if (episode + 1) % 100 == 0:
            print("episode %d %1.2f"%(episode+1, avg[-1]))

        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

    agent.save_model()

    return pts, avg

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

def play_random(episodes):
    y = []
    p = []
    print("Random playing ...")
    for episode in range(episodes):
        env.reset()
        point = 0
        done = False
        cells_to_click = [x for x in range(0, ROWS * COLS)]
        while not done:
            action = random.sample(cells_to_click, 1)[0]
            r = action // COLS
            c = action % COLS
            _, reward, done, info = env.step((r,c))
            if reward > 0:
                point += reward
                for (r,c) in info:
                    action = r * COLS + c
                    cells_to_click.remove(action)

        p.append(point)
        avg = sum(p)/(episode+1)
        y.append(avg)

    return p, y

def plot(random_avg, train_avg):
    x = [xi for xi in range(1, len(train_avg)+1)]
    plt.figure(figsize=(15,10))
    plt.xlabel('Episode')
    plt.ylabel('Point')
    plt.plot(x, random_avg)
    plt.plot(x, train_avg)
    plt.legend(['Random','DQN'])
    plt.title('Average Point')
    plt.savefig('train')

def plot_test(random_avg, dnn_no_heu_avg, dnn_heu_avg, cnn_no_heu_avg, cnn_heu_avg):
    x = [xi for xi in range(1, len(random_avg)+1)]
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

def main(args):
    p1, avg1 = play_random(args.episodes)
    p2, avg2 = train(args.episodes)
    print('------------------ SUMMARY ------------------')
    print('RANDOM:  max %d avg %1.2f max_avg %1.2f'%( max(p1), avg1[-1], max(avg1)))
    print('DQN:     max %d avg %1.2f max_avg %1.2f'%( max(p2), avg2[-1], max(avg2)))
    plot(avg1, avg2)

def test_agent(args):
    p, avg = play_random(args.episodes)
    agent = DQN()
    agent_cnn = DQN()
    agent.load_model('./minesweeper/model/model_' + str(ROWS) + '_' + str(COLS) + '_' + str(MINES) + '.h5')
    agent_cnn.load_model('./minesweeper/model/model_' + str(ROWS) + '_' + str(COLS) + '_' + str(MINES) + '_cnn.h5')
    p1, avg1, win1 = test(agent, args.episodes)
    p2, avg2, win2 = test(agent, args.episodes, heuristic=True)
    p3, avg3, win3 = test(agent_cnn, args.episodes)
    p4, avg4, win4 = test(agent_cnn, args.episodes, heuristic=True)
    print('------------------ SUMMARY ------------------')
    print('RANDOM:              max %d avg %1.2f max_avg %1.2f' % (max(p), avg[-1], max(avg)))
    print('DQN:                 max %d avg %1.2f max_avg %1.2f win %d' % (max(p1), avg1[-1], max(avg1), win1))
    print('DQN HEURISTIC:       max %d avg %1.2f max_avg %1.2f win %d' % (max(p2), avg2[-1], max(avg2), win2))
    print('DQCNN:               max %d avg %1.2f max_avg %1.2f win %d' % (max(p3), avg3[-1], max(avg3), win3))
    print('DQCNN HEURISTIC:     max %d avg %1.2f max_avg %1.2f win %d' % (max(p4), avg4[-1], max(avg4), win4))
    plot_test(avg, avg1, avg2, avg3, avg4)

def test(agent, episodes, heuristic=False):
    y = []
    p = []
    win = 0
    for episode in range(episodes):
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

def parseArgs():
    ''' Reads command line arguments. '''
    parser = argparse.ArgumentParser(description = 'An AI Agent for Minesweeper.', formatter_class = argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode', type = str, default = 'train')
    parser.add_argument('--episodes', type = int, default = '100')
    args = parser.parse_known_args()[0]
    return args

if __name__ == "__main__":
    args = parseArgs()
    if args.mode == 'train':
        main(args)
    else:
        test_agent(args)