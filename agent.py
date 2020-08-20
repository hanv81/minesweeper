from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from collections import deque
import random
import numpy as np

REPLAY_MEMORY_SIZE = 50000
MIN_REPLAY_MEMORY_SIZE = 1000
MINIBATCH_SIZE = 64
UPDATE_CRITIC_EVERY = 5
DISCOUNT = 0.9

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

  def update(self, state, action, reward, next_state, done):
    self.replay_memory.append((state, action, reward, next_state, done))
    if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
      return
    batch = random.sample(self.replay_memory, MINIBATCH_SIZE)
    states = np.array([transition[0] for transition in batch])
    current_qs_list = self.actor.predict(states)

    next_states = np.array([transition[3] for transition in batch])
    next_qs_list = self.critic.predict(next_states)

    X = []
    y = []

    for index, (current_state, action, reward, next_current_state, done) in enumerate(batch):
      new_q = reward
      if not done:
        max_next_q = np.max(next_qs_list[index])
        new_q += DISCOUNT * max_next_q

      current_qs = current_qs_list[index]
      current_qs[action] = new_q

      X.append(current_state)
      y.append(current_qs)

    self.actor.fit(np.array(X), np.array(y), batch_size=MINIBATCH_SIZE, verbose=0)
    if done:
      self.critic_update_counter += 1
      if self.critic_update_counter == UPDATE_CRITIC_EVERY:
        self.critic.set_weights(self.actor.get_weights())