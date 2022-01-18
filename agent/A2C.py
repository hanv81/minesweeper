from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow import keras
import numpy as np
from agent.PG import PG

GAMMA = 0.99
LR = 0.000025

class A2C(PG):
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
    self.Actor = Model(input, output)
    self.Actor.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=LR))

    value = Dense(1, kernel_initializer='he_uniform')(layer)
    self.Critic = Model(input, value)
    self.Critic.compile(loss='mse', optimizer=RMSprop(learning_rate=LR))

  def save_model(self):
    self.Actor.save('a2c.h5')

  def load_model(self, path):
    self.Actor = keras.models.load_model(path)

  def act(self, state):
    prediction = self.Actor.predict(state[None, ...])[0]
    return np.random.choice(self.action_size, p=prediction)

  def replay(self):
    if len(self.states) == 1: # bomb on first cell, nothing to train
      return

    states = np.array(self.states)
    actions = np.array(self.actions)

    # Compute discounted rewards
    discounted_r = self.discount_rewards(self.rewards)

    # Get Critic network predictions
    values = self.Critic.predict(states)[:, 0]

    # Compute advantages
    advantages = discounted_r - values

    self.Actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
    self.Critic.fit(states, discounted_r, epochs=1, verbose=0)