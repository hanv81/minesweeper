from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from collections import deque
import random
import numpy as np

REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
BATCH_SIZE = 64
UPDATE_CRITIC_EVERY = 5
GAMMA = 0.9
ALPHA = 0.1

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
    self.critic_update_counter = 0

  def get_q(self, state):
    return self.actor.predict(np.array(state[None, ...]))

  def update_critic(self):
    self.critic_update_counter += 1
    if self.critic_update_counter % UPDATE_CRITIC_EVERY == 0:
      self.critic.set_weights(self.actor.get_weights())

  def update_replay_memory(self, transition):
    self.replay_memory.append(transition)

  def train(self):
    if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
      return
    batch = random.sample(self.replay_memory, BATCH_SIZE)
    states = np.array([transition[0] for transition in batch])
    qs_list = self.actor.predict(states)

    next_states = np.array([transition[3] for transition in batch])
    next_qs = self.actor.predict(next_states)

    X = []
    y = []

    for index, (state, action, reward, next_state, done) in enumerate(batch):
      qs = qs_list[index]
      qs[action] = reward
      if not done:
        qs[action] += GAMMA * np.max(next_qs[index])

      X.append(state)
      y.append(qs)

    self.actor.fit(np.array(X), np.array(y), batch_size=BATCH_SIZE, verbose=0)
