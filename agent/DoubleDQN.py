import random
import numpy as np
from agent.DQN import DQN

class DoubleDQN(DQN):
  def save_model(self):
    self.model.save('double_dqn.h5')

  def step(self, transition):
    self.replay_memory.append(transition)
    if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
      return

    batch = random.sample(self.replay_memory, self.BATCH_SIZE)
    states = np.array([transition[0] for transition in batch])
    qs = self.model.predict(states)

    next_states = np.array([transition[3] for transition in batch])
    next_qs = self.model.predict(next_states)
    next_qs_target = self.target.predict(next_states)

    X = []
    y = []

    for index, (state, action, reward, _, done) in enumerate(batch):
      next_action = np.argmax(next_qs[index])
      qs[index][action] = reward + (1 - int(done)) * self.GAMMA * next_qs_target[index][next_action]
      X.append(state)
      y.append(qs[index])

    self.model.fit(np.array(X), np.array(y), batch_size=self.BATCH_SIZE, verbose=0)
