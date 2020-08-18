from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

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