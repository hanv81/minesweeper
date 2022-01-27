from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras
from tqdm import tqdm
import random
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

  def act(self, state):
    prediction = self.model.predict(state[None, ...])[0]
    return np.random.choice(self.action_size, p=prediction)

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
      step = point = 0
      done = False
      while not done and step < 30:
        action = np.random.randint(0, self.action_size) if step == 0 else self.act(state)
        next_state, reward, done, _ = env.step(action)
        self.remember(state, action, reward)
        state = next_state
        step += 1
        if reward > 0:
          point += reward

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

  def test(self, env, episodes, rows, cols, heuristic=False):
    pts = []
    win = 0
    for _ in tqdm(range(episodes)):
      state = env.reset()
      point = 0
      done = False
      while not done:
        action = random.randint(0, rows * cols - 1) if point == 0 else self.act(state)
        next_state, reward, done, _ = env.step(action)
        if reward > 0:
          point += reward
          if done:
            win += 1

        state = next_state

      pts.append(point)

    print('max %d avg %1.2f win %d' % (max(pts), np.mean(pts), win))
    return pts, win