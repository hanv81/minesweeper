from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow import keras
from collections import deque
import random
import numpy as np

REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
BATCH_SIZE = 64
GAMMA = 0.99

class Agent:
  def create_model(self, rows, cols):
    input = Input(shape=(rows, cols))
    layer = Flatten()(input)
    layer = Dense(512, activation='relu')(layer)
    output = Dense(rows * cols, activation='linear')(layer)
    model = Model(input, output)
    model.compile(loss='mse', optimizer='adam', metrics='accuracy')
    model.summary()
    return model

  def save_model(self):
    self.model.save('model.h5')

  def load_model(self):
    self.model = keras.models.load_model('model.h5')

  def __init__(self, rows, cols):
    self.model = self.create_model(rows, cols)
    self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

  def predict(self, state):
    return self.model.predict(np.array(state[None, ...]))

  def update_replay_memory(self, transition):
    self.replay_memory.append(transition)

  def train(self):
    if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
      return
    batch = random.sample(self.replay_memory, BATCH_SIZE)
    states = np.array([transition[0] for transition in batch])
    qs_list = self.model.predict(states)

    next_states = np.array([transition[3] for transition in batch])
    next_qs = self.model.predict(next_states)

    X = []
    y = []

    for index, (state, action, reward, next_state, done) in enumerate(batch):
      qs = qs_list[index]
      qs[action] = reward
      if not done:
        qs[action] += GAMMA * np.max(next_qs[index])

      X.append(state)
      y.append(qs)

    self.model.fit(np.array(X), np.array(y), batch_size=BATCH_SIZE, verbose=0)
