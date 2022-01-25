import torch
import torch.nn as nn
import torch.optim as optim
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

class DQNTorch:
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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = Net(rows, cols, cnn).to(self.device)
        self.target = Net(rows, cols, cnn).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-6)
        self.criterion = nn.MSELoss()

    def save_model(self):
        torch.save(self.model.state_dict(), 'dqn_torch.h5')

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()

    def predict(self, state):
        return self.model(state)

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())

    def step(self, transition):
        self.replay_memory.append(transition)
        if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
            return

        batch = random.sample(self.replay_memory, self.BATCH_SIZE)
        states = torch.cat(tuple(transition[0] for trasition in batch))
        actions = torch.cat(tuple(transition[1] for trasition in batch))
        rewards = torch.cat(tuple(transition[2] for trasition in batch))
        next_states = torch.cat(tuple(transition[3] for trasition in batch))
        dones = torch.cat(tuple(transition[4] for trasition in batch))
        qs = torch.sum(self.model(states) * actions, dim=1)
        next_qs = self.target(next_states)

        qs_target = torch.cat(tuple(rewards[i] + (1-int(dones[i])) * self.GAMMA * torch.max(next_qs[i]) for i in range(len(batch))))
        self.optimizer.zero_grad()
        qs_target = qs_target.detach()
        loss = self.criterion(qs, qs_target)
        loss.backward()
        self.optimizer.step()

    def act(self, state, cells_to_click, clicked_cells):
        if random.random() > self.epsilon:
            state_ts = torch.from_numpy(state).type(torch.FloatTensor).unsqueeze(0)
            qs = self.predict(state_ts)[0]
            for cell in clicked_cells:
                qs[cell] = torch.min(qs)
            if torch.max(qs) > torch.min(qs): # if max = min -> random
                return torch.argmax(qs).item()

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
                action = random.randint(0, self.rows*self.cols-1) if point == 0 else self.act(state, cells_to_click, clicked_cells)
                next_state, reward, done, info = env.step(action)

                if point > 0: # point = 0 -> first cell, just random, nothing to learn
                    action_ts = torch.zeros([self.rows*self.cols], dtype=torch.float32)
                    action_ts[action] = 1
                    action_ts = action_ts.unsqueeze(0)
                    state_ts = torch.from_numpy(state).type(torch.FloatTensor).unsqueeze(0)
                    next_state_ts = torch.from_numpy(next_state).type(torch.FloatTensor).unsqueeze(0)
                    reward_ts = torch.Tensor([reward]).unsqueeze(0).to(self.device)
                    done_ts = torch.Tensor([done]).unsqueeze(0).to(self.device)
                    self.step((state_ts, action_ts, reward_ts, next_state_ts, done_ts))

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