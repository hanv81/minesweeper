from tensorflow.keras.models import Model, clone_model, load_model
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D
from collections import deque
from tqdm import tqdm
import random
import numpy as np

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
    input = Input(shape=(rows, cols, 1))
    if cnn:
      self.cnn = True
      layer = Conv2D(filters=8, kernel_size=3, padding='same', activation='relu')(input)
      layer = MaxPooling2D()(layer)
      layer = Flatten()(layer)
    else:
      self.cnn = False
      layer = Flatten()(input)
    layer = Dense(512, activation='relu')(layer)
    output = Dense(rows * cols, activation='linear')(layer)
    self.model = Model(input, output)
    self.model.compile(loss='mse', optimizer='adam')
    self.target = clone_model(self.model)

  def save_model(self):
    self.model.save('dqn.h5')

  def load_model(self, path):
    self.model = load_model(path)

  def is_value_based(self):
    return True

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

  def act(self, state, cells_to_click, clicked_cells):
    if random.random() >= self.epsilon:
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
        action = random.randint(0, self.rows*self.cols-1) if point == 0 else self.act(state, cells_to_click, clicked_cells)
        next_state, reward, done, info = env.step(action)
        if point > 0: # point = 0 -> first cell, just random, nothing to learn
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

  def test(self, env, episodes, rows, cols, heuristic=False):
    pts = []
    win = 0
    self.epsilon = 0 # exploit only
    for _ in tqdm(range(episodes), desc='Testing...'):
      state = env.reset()
      point = 0
      done = False
      clicked_cells = []
      cells_to_click = [x for x in range(0, rows * cols)]
      while not done:
        action = random.randint(0, rows * cols - 1) if point == 0 else self.act(state, cells_to_click, clicked_cells)
        next_state, reward, done, info = env.step(action)
        if reward > 0:
          if done:
            win += 1
          point += reward
          for (r,c) in info['coord']:
            action = r * cols + c
            clicked_cells.append(action)
            cells_to_click.remove(action)
        state = next_state

      pts.append(point)

    print('max %d avg %1.2f win %d' % (max(pts), np.mean(pts), win))
    return pts, win