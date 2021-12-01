from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras
import numpy as np

GAMMA = 0.99
LR = 0.000025

class PG:
  def __init__(self):
    self.states, self.actions, self.rewards = [], [], []

  def create_model(self, rows, cols, cnn=False):
    self.rows = rows
    self.cols = cols
    self.action_size = rows * cols
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
    output = Dense(rows * cols, activation='softmax', kernel_initializer='he_uniform')(layer)
    self.model = Model(input, output)
    self.model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=LR))

  def save_model(self):
    self.model.save('pg.h5')

  def load_model(self, path):
    self.model = keras.models.load_model(path)

  def predict(self, state):
    return self.model.predict(state[None, ...])

  def act(self, state, cells_to_click):
    if len(cells_to_click) == self.action_size:
      return np.random.randint(0, self.action_size) # first cell -> random

    prediction = self.predict(state)[0]
    while True:
      action = np.random.choice(self.action_size, p=prediction)
      if action in cells_to_click:
        return action

  def remember(self, state, action, reward):
    self.states.append(state)
    action_onehot = np.zeros([self.action_size])
    action_onehot[action] = 1
    self.actions.append(action_onehot)
    self.rewards.append(float(reward))

  def discount_rewards(self, reward):
    running_add = 0
    discounted_r = np.zeros_like(reward)
    for i in reversed(range(0,len(reward))):
      running_add = running_add * GAMMA + reward[i]
      discounted_r[i] = running_add

    discounted_r -= np.mean(discounted_r) # normalizing the result
    discounted_r /= np.std(discounted_r) # divide by standard deviation
    return discounted_r    

  def replay(self):
    if len(self.states) == 1: # bomb on first cell, nothing to train
      return

    states = np.array(self.states)
    actions = np.array(self.actions)

    # Compute discounted rewards
    discounted_r = self.discount_rewards(self.rewards)

    # training PG network
    self.model.fit(states, actions, sample_weight=discounted_r, epochs=1, verbose=0)

  def train(self, episodes, env):
    avg = []
    pts = []
    for episode in range(episodes):
      state = env.reset()
      point = 0
      done = False
      cells_to_click = [x for x in range(0, self.action_size)]
      while not done:
        action = self.act(state, cells_to_click)
        r = action // self.cols
        c = action % self.cols
        next_state, reward, done, info = env.step((r,c))
        self.remember(state, action, reward)
        state = next_state
        if reward > 0:
          point += reward
          for (r,c) in info:
            action = r * self.cols + c
            cells_to_click.remove(action)

      self.replay()
      # reset training memory
      self.states, self.actions, self.rewards = [], [], []

      pts.append(point)
      avg.append(np.mean(pts))
      
      if (episode + 1) % 100 == 0:
          print("episode %d %1.2f"%(episode+1, avg[-1]))

    env.close()
    self.save_model()

    return pts, avg