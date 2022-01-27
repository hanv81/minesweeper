import gym
import gym_minesweeper  # must import for create env
from agent.DQN import DQN
from agent.DDQN import DDQN
from agent.DoubleDQN import DoubleDQN
from agent.PG import PG
from agent.A2C import A2C
from agent.DQN_Torch import DQNTorch
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse

def play_random(env, episodes, rows, cols):
    pts = []
    avg = []
    print("Random playing ...")
    for _ in tqdm(range(episodes)):
        env.reset()
        point = 0
        done = False
        cells_to_click = [x for x in range(0, rows * cols)]
        while not done:
            action = random.sample(cells_to_click, 1)[0]
            _, reward, done, info = env.step(action)
            if reward > 0:
                point += reward
                for (r,c) in info['coord']:
                    action = r * cols + c
                    cells_to_click.remove(action)

        pts.append(point)
        avg.append(np.mean(pts))

    return pts, avg

def show_training_summary(algo, pts_train, avg_train, pts_ran, avg_ran):
    print('------------------', algo.upper(), 'SUMMARY ------------------')
    print('RANDOM:  max %d avg %1.2f' % (max(pts_ran), avg_ran[-1]))
    print('TRAIN:   max %d avg %1.2f' % (max(pts_train), avg_train[-1]))
    plot(algo, avg_ran, avg_train)

def plot(algo, random_avg, train_avg):
    x = [xi for xi in range(1, len(train_avg)+1)]
    plt.figure(figsize=(15,10))
    plt.xlabel('Episode')
    plt.ylabel('Point')
    plt.plot(x, random_avg)
    plt.plot(x, train_avg)
    plt.legend(['Random', algo.upper()])
    plt.title('Average Point')
    plt.savefig('train')

def plot_test(random_pts, dqn_pts, ddqn_pts, double_dqn_pts, pg_pts, a2c_pts):
    labels = ['Random', 'DQN', 'DDQN', 'Double DQN', 'PG', 'A2C']
    avg_pts = [np.mean(random_pts), np.mean(dqn_pts), np.mean(ddqn_pts), np.mean(double_dqn_pts), np.mean(pg_pts), np.mean(a2c_pts)]
    max_pts = [max(random_pts), max(dqn_pts), max(ddqn_pts), max(double_dqn_pts), max(pg_pts), max(a2c_pts)]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, avg_pts, width, label='Average Point')
    rects2 = ax.bar(x + width/2, max_pts, width, label='Maximum Point')

    ax.set_ylabel('Points')
    ax.set_title('Test Result')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    plt.savefig('test')

def test(args):
    env = gym.make('minesweeper-v0', rows=args.rows, cols=args.cols, mines=args.mines)
    dqn = DQN()
    ddqn = DDQN()
    double_dqn = DoubleDQN()
    pg = PG()
    a2c = A2C()
    dqn.load_model('./minesweeper/model/dqn.h5')
    ddqn.load_model('./minesweeper/model/ddqn.h5')
    double_dqn.load_model('./minesweeper/model/double_dqn.h5')
    pg.load_model('./minesweeper/model/pg.h5')
    a2c.load_model('./minesweeper/model/a2c.h5')
    pg.action_size = args.rows * args.cols
    a2c.action_size = args.rows * args.cols
    p, avg = play_random(env, args.episodes, args.rows, args.cols)
    print('\nTesting DQN ...')
    p1, win1 = dqn.test(env, args.episodes, args.rows, args.cols)
    print('\nTesting DDQN ...')
    p2, win2 = ddqn.test(env, args.episodes, args.rows, args.cols)
    print('\nTesting Double DQN ...')
    p3, win3 = double_dqn.test(env, args.episodes, args.rows, args.cols)
    print('\nTesting PG ...')
    p4, win4 = pg.test(env, args.episodes, args.rows, args.cols)
    print('\nTesting A2C ...')
    p5, win5 = a2c.test(env, args.episodes, args.rows, args.cols)

    print('\n------------------ SUMMARY ------------------')
    print('RANDOM:      max %d avg %1.2f' % (max(p), avg[-1]))
    print('DQN:         max %d avg %1.2f win %d' % (max(p1), np.mean(p1), win1))
    print('DDQN:        max %d avg %1.2f win %d' % (max(p2), np.mean(p2), win2))
    print('Double DQN:  max %d avg %1.2f win %d' % (max(p3), np.mean(p3), win3))
    print('PG:          max %d avg %1.2f win %d' % (max(p4), np.mean(p4), win4))
    print('A2C:         max %d avg %1.2f win %d' % (max(p5), np.mean(p5), win5))

    plot_test(p, p1, p2, p3, p4, p5)

def train(args):
    if args.algo == 'dqn':
        agent = DQN()
    elif args.algo == 'dqntorch':
        agent = DQNTorch()
    elif args.algo == 'doubledqn':
        agent = DoubleDQN()
    elif args.algo == 'ddqn':
        agent = DDQN()
    elif args.algo == 'a2c':
        agent = A2C()
    else:
        agent = PG()

    env = gym.make('minesweeper-v0', rows=args.rows, cols=args.cols, mines=args.mines)
    agent.create_model(args.rows, args.cols, args.cnn)
    pts_ran, avg_ran = play_random(env, args.episodes, args.rows, args.cols)
    pts, avg = agent.train(args.episodes, env)
    show_training_summary(args.algo, pts, avg, pts_ran, avg_ran)

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
    if args.mode == 'train':
        train(args)
    else:
        test(args)

if __name__ == "__main__":
    main(parseArgs())