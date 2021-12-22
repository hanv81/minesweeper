import gym
import gym_minesweeper  # must import for create env
from agent.DQN import DQN
from agent.DDQN import DDQN
from agent.DoubleDQN import DoubleDQN
from agent.PG import PG
from agent.A2C import A2C
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse

def play_random(env, episodes, rows, cols):
    avg = []
    pts = []
    print("Random playing ...")
    for _ in range(episodes):
        env.reset()
        point = 0
        done = False
        cells_to_click = [x for x in range(0, rows * cols)]
        while not done:
            action = random.sample(cells_to_click, 1)[0]
            _, reward, done, info = env.step(action)
            if reward > 0:
                point += reward
                for (r,c) in info:
                    action = r * cols + c
                    cells_to_click.remove(action)

        pts.append(point)
        avg.append(np.mean(pts))

    return pts, avg

def show_training_summary(algo, pts_train, avg_train, pts_ran, avg_ran):
    print('------------------', algo.upper(), 'SUMMARY ------------------')
    print('RANDOM:  max %d avg %1.2f' % (max(pts_ran), avg_ran[-1]))
    print('TRAIN:   max %d avg %1.2f' % (max(pts_train), avg_train[-1]))
    plot(avg_ran, avg_train)

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

def test_agent(args):
    env = gym.make('minesweeper-v0', rows=args.rows, cols=args.cols, mines=args.mines)
    p, avg = play_random(env, args.episodes, args.rows, args.cols)
    agent = DQN()
    agent_cnn = DQN()
    agent.load_model('./minesweeper/model/model_' + str(args.rows) + '_' + str(args.cols) + '_' + str(args.mines) + '.h5')
    agent_cnn.load_model('./minesweeper/model/model_' + str(args.rows) + '_' + str(args.cols) + '_' + str(args.mines) + '_cnn.h5')
    p1, avg1, win1 = test(agent, env, args.episodes, args.rows, args.cols)
    p2, avg2, win2 = test(agent, env, args.episodes, args.rows, args.cols, heuristic=True)
    p3, avg3, win3 = test(agent_cnn, env, args.episodes, args.rows, args.cols)
    p4, avg4, win4 = test(agent_cnn, env, args.episodes, args.rows, args.cols, heuristic=True)
    print('------------------ SUMMARY ------------------')
    print('RANDOM:              max %d avg %1.2f max_avg %1.2f' % (max(p), avg[-1], max(avg)))
    print('DQN:                 max %d avg %1.2f max_avg %1.2f win %d' % (max(p1), avg1[-1], max(avg1), win1))
    print('DQN HEURISTIC:       max %d avg %1.2f max_avg %1.2f win %d' % (max(p2), avg2[-1], max(avg2), win2))
    print('DQCNN:               max %d avg %1.2f max_avg %1.2f win %d' % (max(p3), avg3[-1], max(avg3), win3))
    print('DQCNN HEURISTIC:     max %d avg %1.2f max_avg %1.2f win %d' % (max(p4), avg4[-1], max(avg4), win4))
    plot_test(avg, avg1, avg2, avg3, avg4)

def test(agent, env, episodes, rows, cols, heuristic=False):
    y = []
    p = []
    win = 0
    for episode in range(episodes):
        state = env.reset()
        point = 0
        done = False
        clicked_cells = []
        cells_to_click = [x for x in range(0, rows * cols)]
        while not done:
            if point == 0: # first cell -> just random
                action = random.randint(0, rows * cols - 1)
            else:
                mine_cells = []
                if heuristic:
                    for i in range(rows):
                        for j in range(cols):
                            if state[i,j] > 0:
                                neibors = []
                                neibors_flag = []
                                for r in range(i-1, i+2):
                                    for c in range(j-1, j+2):
                                        if 0<=r<rows and 0<=c<cols:
                                            if state[r,c] < 0:
                                                pos = r * cols + c
                                                if pos in mine_cells:
                                                    neibors_flag.append(pos)
                                                else:
                                                    neibors.append(pos)
                                if state[i,j] == len(neibors) + len(neibors_flag):
                                    for n in neibors:
                                        mine_cells.append(n)
                qs = agent.predict(state)[0]
                for cell in range(rows*cols):
                    if cell in clicked_cells or cell in mine_cells:
                        qs[cell] = np.min(qs)
                if np.max(qs) > np.min(qs):
                    action = np.argmax(qs)
                else:
                    action = random.sample(cells_to_click, 1)[0]

            next_state, reward, done, info = env.step(action)
            if reward > 0:
                if done:
                    win += 1
                point += reward
                for (r,c) in info:
                    action = r * cols + c
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
    parser.add_argument('--algo', type = str, default = 'dqn')
    parser.add_argument('--episodes', type = int, default = 100)
    parser.add_argument('--rows', type = int, default = 10)
    parser.add_argument('--cols', type = int, default = 10)
    parser.add_argument('--mines', type = int, default = 10)
    parser.add_argument('--cnn', type = bool, default = False)
    args = parser.parse_known_args()[0]
    return args

def main(args):
    env = gym.make('minesweeper-v0', rows=args.rows, cols=args.cols, mines=args.mines)
    pts_ran, avg_ran = play_random(env, args.episodes, args.rows, args.cols)
    if args.mode == 'train':
        if args.algo == 'dqn':
            agent = DQN()
        elif args.algo == 'doubledqn':
            agent = DoubleDQN()
        elif args.algo == 'ddqn':
            agent = DDQN()
        elif args.algo == 'a2c':
            agent = A2C()
        else:
            agent = PG()

        agent.create_model(args.rows, args.cols, args.cnn)
        pts, avg = agent.train(args.episodes, env)
        show_training_summary(args.algo, pts, avg, pts_ran, avg_ran)
    else:
        test_agent(args)

if __name__ == "__main__":
    main(parseArgs())