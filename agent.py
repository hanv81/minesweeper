from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from collections import deque

REPLAY_MEMORY_SIZE = 10000

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

  def __init__(self, rows, cols):
    self.actor = self.create_model(rows, cols)
    self.critic = self.create_model(rows, cols)
    self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

  def update(self, state, action, reward, next_state, done):
    self.replay_memory.append((state, action, reward, next_state, done))