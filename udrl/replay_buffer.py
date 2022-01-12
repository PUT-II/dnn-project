from collections import namedtuple
from typing import List

import numpy as np

EpisodeTuple = namedtuple(
    typename='Episode',
    field_names=[
        'states',
        'actions',
        'infos',
        'rewards',
        'init_command',
        'total_return',
        'length',
    ]
)


class ReplayBuffer:
    __array: List[EpisodeTuple]

    def __init__(self, init_array: List[EpisodeTuple] = None):
        if init_array is None:
            self.__array = []
        else:
            self.__array = init_array

    def __len__(self) -> int:
        return len(self.__array)

    def __getitem__(self, item):
        if type(item) is slice:
            return ReplayBuffer(self.__array[item])
        else:
            return self.__array[item]

    def __iter__(self):
        for item in self.__array:
            yield item

    def append(self, item):
        self.__array.append(item)

    def sort(self):
        self.__array.sort(key=lambda episode: episode.total_return)

    def random_batch(self, batch_size) -> List[EpisodeTuple]:
        indexes = np.random.randint(0, len(self.__array), batch_size)
        return [self.__array[index] for index in indexes]

    def sample_command(self, last_few: int):
        if len(self.__array) == 0:
            return [1, 1]

        commands = self.__array[-last_few:]

        lengths = [command.length for command in commands]
        returns = [command.total_return for command in commands]
        mean_return, std_return = np.mean(returns), np.std(returns)

        desired_horizon = np.round(np.mean(lengths))
        desired_return = np.random.uniform(mean_return, mean_return + std_return)
        return [desired_return, desired_horizon]

    def save(self, filename):
        np.save(filename, np.array(self.__array, dtype=object))

    def load(self, filename):
        raw_buffer: np.array = np.load(filename, allow_pickle=True)
        # e stands for episode
        self.__array = [EpisodeTuple(e[0], e[1], e[2], e[3], e[4], e[5], e[6]) for e in raw_buffer]
