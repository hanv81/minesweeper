from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D
from tensorflow import keras
from collections import deque
import random
import numpy as np

class DQN:
  REPLAY_MEMORY_SIZE = 50000
  MIN_REPLAY_MEMORY_SIZE = 1000
  BATCH_SIZE = 64
  GAMMA = 0.99
  EPSILON_DECAY = 0.99975
  MIN_EPSILON = 0.001

  def __init__(self):
    self.replay_memory = deque(maxlen=self.REPLAY_MEMORY_SIZE)

  def create_model(self, rows, cols, cnn=False):
    self.rows = rows
    self.cols = cols
    input = Input(shape=(rows, cols, 1))
    if cnn:
      self.cnn = True
      layer = Conv2D(filters=8, kernel_size=2, padding='same', activation='relu')(input)
      layer = MaxPooling2D()(layer)
      layer = Flatten()(layer)
    else:
      self.cnn = False
      layer = Flatten()(input)
    layer = Dense(512, activation='relu')(layer)
    output = Dense(rows * cols, activation='linear')(layer)
    self.model = Model(input, output)
    self.model.compile(loss='mse', optimizer='adam', metrics='accuracy')
    self.target = keras.models.clone_model(self.model)

  def save_model(self):
    self.model.save('dqn.h5')

  def load_model(self, path):
    self.model = keras.models.load_model(path)

  def predict(self, state):
    return self.model.predict(state[None, ...])

  def update_target(self):
    self.target.set_weights(self.model.get_weights()) 

  def step(self, transition):
    self.replay_memory.append(transition)
    if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
      return

    batch = random.sample(self.replay_memory, self.BATCH_SIZE)
    states = np.array([transition[0] for transition in batch])
    qs = self.model.predict(states)

    next_states = np.array([transition[3] for transition in batch])
    next_qs = self.target.predict(next_states)

    X = []
    y = []

    for index, (state, action, reward, _, done) in enumerate(batch):
      qs[index][action] = reward + (1 - int(done)) * self.GAMMA * np.max(next_qs[index])
      X.append(state)
      y.append(qs[index])

    self.model.fit(np.array(X), np.array(y), batch_size=self.BATCH_SIZE, verbose=0)

  def act(self, state, cells_to_click, clicked_cells, epsilon):
    if random.random() <= epsilon:
        return random.sample(cells_to_click, 1)[0]
    else:
        qs = self.predict(state)[0]
        for cell in clicked_cells:
            qs[cell] = np.min(qs)
        if np.max(qs) > np.min(qs):
            return np.argmax(qs)

    return random.sample(cells_to_click, 1)[0]    # no max action, just random

  def train(self, episodes, env):
    epsilon = 1
    
    avg = []
    pts = []
    for episode in range(episodes):
        state = env.reset()
        point = 0
        done = False
        clicked_cells = []
        cells_to_click = [x for x in range(0, self.rows * self.cols)]
        while not done:
            action = self.act(state, cells_to_click, clicked_cells, epsilon)
            r = action // self.cols
            c = action % self.cols
            next_state, reward, done, info = env.step((r,c))
            self.step((state, action, reward, next_state, done))
            if reward > 0:
                point += reward
                for (r,c) in info:
                    action = r * self.cols + c
                    clicked_cells.append(action)
                    cells_to_click.remove(action)
            state = next_state

        self.update_target()
        pts.append(point)
        avg.append(np.mean(pts))
        
        if (episode + 1) % 100 == 0:
            print("episode %d %1.2f"%(episode+1, avg[-1]))

        epsilon *= self.EPSILON_DECAY
        epsilon = max(self.MIN_EPSILON, epsilon)

    self.save_model()

    return pts, avg