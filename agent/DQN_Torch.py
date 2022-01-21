import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import numpy as np

class Net(nn.Module):
    def __init__(self, rows, cols, cnn):
        super(Net, self).__init__()
        self.rows = rows
        self.cols = cols
        self.cnn = cnn
        self.fc1 = nn.Linear(rows*cols, 512)
        self.fc2 = nn.Linear(512, rows*cols)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DQN:
    REPLAY_MEMORY_SIZE = 50000
    MIN_REPLAY_MEMORY_SIZE = 1000
    BATCH_SIZE = 64
    GAMMA = 0.99
    EPSILON_DECAY = 0.99975
    MIN_EPSILON = 0.001
    epsilon = 1

    def __init__(self):
        self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)

    def create_model(self, rows, cols, cnn=False):
        self.rows = rows
        self.cols = cols
        self.model = Net(rows, cols, cnn)
        self.target = Net(rows, cols, cnn)

    def save_model(self):
        torch.save(self.model.state_dict(), 'dqn_torch.h5')

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def predict(self, state):
        return self.model(state)[0]

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def step(self, transition):
        self.replay_memory.append(transition)
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        batch = random.sample(self.replay_memory, self.BATCH_SIZE)
        states = np.array([transition[0] for transition in batch])
        qs = self.model(states)

        next_states = np.array([transition[3] for transition in batch])
        next_qs = self.target(next_states)

        X = []
        y = []

        for index, (state, action, reward, _, done) in enumerate(batch):
            qs[index][action] = reward + (1 - int(done)) * self.GAMMA * np.max(next_qs[index])
            X.append(state)
            y.append(qs[index])

        self.model.fit(np.array(X), np.array(y), batch_size=self.BATCH_SIZE, verbose=0)

    def act(self, state, cells_to_click, clicked_cells):
        if random.random() > self.epsilon:
            qs = self.predict(state)[0]
        for cell in clicked_cells:
            qs[cell] = np.min(qs)
        if np.max(qs) > np.min(qs): # if max = min -> random
            return np.argmax(qs)

        return random.sample(cells_to_click, 1)[0]

    def train(self, episodes, env):
        avg = []
        pts = []
        for episode in range(episodes):
            state = env.reset()
            point = 0
            done = False
            clicked_cells = []
            cells_to_click = [x for x in range(0, self.rows * self.cols)]
            while not done:
                action = self.act(state, cells_to_click, clicked_cells)
                next_state, reward, done, info = env.step(action)
                self.step((state, action, reward, next_state, done))
                if reward > 0:
                    point += reward
                for (r,c) in info['coord']:
                    action = r * self.cols + c
                    clicked_cells.append(action)
                    cells_to_click.remove(action)
                state = next_state

            self.update_target()
            pts.append(point)
            avg.append(np.mean(pts))
            
            if (episode + 1) % 100 == 0:
                print("episode %d %1.2f"%(episode+1, avg[-1]))

            if len(self.replay_memory) >= self.MIN_REPLAY_MEMORY_SIZE:
                self.epsilon = max(self.MIN_EPSILON, self.epsilon*self.EPSILON_DECAY)

            self.save_model()

        return pts, avg