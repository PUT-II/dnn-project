import numpy as np

from udrl.util import make_episode


class ReplayBuffer:
    def __init__(self, size=0):
        self.size = size
        self.buffer = []

    def add(self, episode):
        self.buffer.append(episode)

    def get(self, num):
        return self.buffer[-num:]

    def random_batch(self, batch_size):
        indexes = np.random.randint(0, len(self), batch_size)
        return [self.buffer[index] for index in indexes]

    def sort(self):
        self.buffer = sorted(self.buffer, key=lambda episode: episode.total_return)[-self.size:]

    def save(self, filename):
        np.save(filename, self.buffer)

    def load(self, filename):
        raw_buffer: np.array = np.load(filename)
        self.size = len(raw_buffer)
        # e stands for episode
        self.buffer = [make_episode(e[0], e[1], e[2], e[3], e[4], e[5]) for e in raw_buffer]

    def __len__(self):
        return len(self.buffer)
