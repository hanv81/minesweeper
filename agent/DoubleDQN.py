import random
import numpy as np
from agent.DQN import DQN

class DoubleDQN(DQN):
  def step(self, transition):
    self.replay_memory.append(transition)
    if len(self.replay_memory) < self.MIN_REPLAY_MEMORY_SIZE:
      return

    batch = random.sample(self.replay_memory, self.BATCH_SIZE)
    states = np.array([state for state,_,_,_,_ in batch])
    qs = self.model.predict(states, verbose=0)

    next_states = np.array([next_state for _,_,_,next_state,_ in batch])
    next_qs = self.model.predict(next_states)
    next_qs_target = self.target.predict(next_states, verbose=0)

    for index, (_, action, reward, _, done) in enumerate(batch):
      next_action = next_qs[index].argmax()
      qs[index][action] = reward + (1 - done) * self.GAMMA * next_qs_target[index][next_action]

    self.model.fit(states, qs, batch_size=self.BATCH_SIZE, verbose=0)
