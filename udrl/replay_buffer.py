import numpy as np

from udrl.util import make_episode


class ReplayBuffer:
    def __init__(self, init_array: list = None):
        if init_array is None:
            self.__array = []
        else:
            self.__array = init_array

    def get_n_first(self, num):
        return ReplayBuffer(init_array=self.__array[:num])

    def get_n_last(self, num):
        return ReplayBuffer(init_array=self.__array[-num:])

    def random_batch(self, batch_size) -> list:
        indexes = np.random.randint(0, len(self.__array), batch_size)
        return [self.__array[index] for index in indexes]

    def append(self, item):
        self.__array.append(item)

    def sort(self):
        self.__array.sort(key=lambda episode: episode.total_return)

    def save(self, filename):
        np.save(filename, np.array(self.__array, dtype=object))

    def load(self, filename):
        raw_buffer: np.array = np.load(filename)
        # e stands for episode
        self.__array = [make_episode(e[0], e[1], e[2], e[3], e[4], e[5], e[6]) for e in raw_buffer]

    def __len__(self) -> int:
        return len(self.__array)

    def __getitem__(self, item):
        return self.__array[item]
