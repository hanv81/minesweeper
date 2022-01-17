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
                for (r,c) in info['coord']:
                    action = r * cols + c
                    cells_to_click.remove(action)

        pts.append(point)

    return pts

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

def plot_test(random_avg, random_max, dqn_avg, dqn_max, ddqn_avg, ddqn_max, double_dqn_avg, double_dqn_max, pg_avg, pg_max):
    labels = ['Random', 'DQN', 'DDQN', 'Double DQN', 'PG']
    avg_pts = [random_avg, dqn_avg, ddqn_avg, double_dqn_avg, pg_avg]
    max_pts = [random_max, dqn_max, ddqn_max, double_dqn_max, pg_max]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, avg_pts, width, label='Average Point')
    rects2 = ax.bar(x + width/2, max_pts, width, label='Maximum Point')

    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
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
    p = play_random(env, args.episodes, args.rows, args.cols)
    dqn = DQN()
    ddqn = DDQN()
    double_dqn = DoubleDQN()
    pg = PG()
    dqn.load_model('./minesweeper/model/dqn.h5')
    ddqn.load_model('./minesweeper/model/dueling_double_dqn.h5')
    double_dqn.load_model('./minesweeper/model/double_dqn.h5')
    pg.load_model('./minesweeper/model/pg.h5')
    pg.action_size = args.rows * args.cols
    print('\nTesting DQN ...')
    p1, win1 = dqn.test(env, args.episodes, args.rows, args.cols)
    print('\nTesting DDQN ...')
    p2, win2 = ddqn.test(env, args.episodes, args.rows, args.cols)
    print('\nTesting Double DQN ...')
    p3, win3 = double_dqn.test(env, args.episodes, args.rows, args.cols)
    print('\nTesting PG ...')
    p4, win4 = pg.test(env, args.episodes, args.rows, args.cols)

    print('------------------ SUMMARY ------------------')
    print('RANDOM:      max %d avg %1.2f' % (max(p), np.mean(p)))
    print('DQN:         max %d avg %1.2f win %d' % (max(p1), np.mean(p1), win1))
    print('DDQN:        max %d avg %1.2f win %d' % (max(p2), np.mean(p2), win2))
    print('Double DQN:  max %d avg %1.2f win %d' % (max(p3), np.mean(p3), win3))
    print('PG:          max %d avg %1.2f win %d' % (max(p4), np.mean(p4), win4))

    plot_test(np.mean(p), max(p), np.mean(p1), max(p1), np.mean(p2), max(p2), np.mean(p3), max(p3), np.mean(p4), max(p4))

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

        env = gym.make('minesweeper-v0', rows=args.rows, cols=args.cols, mines=args.mines)
        agent.create_model(args.rows, args.cols, args.cnn)
        pts_ran, avg_ran = play_random(env, args.episodes, args.rows, args.cols)
        pts, avg = agent.train(args.episodes, env)
        show_training_summary(args.algo, pts, avg, pts_ran, avg_ran)
    else:
        test(args)

if __name__ == "__main__":
    main(parseArgs())