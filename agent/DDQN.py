from tensorflow.keras.models import Model, clone_model
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, MaxPooling2D, Add
from tensorflow import keras
from agent.DoubleDQN import DoubleDQN

class DDQN(DoubleDQN):
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
    value = Dense(512, activation='relu')(layer)
    value = Dense(1, activation='linear')(value)
    advantage = Dense(512, activation='relu')(layer)
    advantage = Dense(rows * cols)(advantage)
    output = Add()([value, advantage])
    self.model = Model(input, output)
    self.model.compile(loss='mse', optimizer='adam')
    clone_model(self.model)