from time import time

import numpy as np
import torch
import torch.nn.functional as nn_functional

from udrl.behavior import Behavior
from udrl.replay_buffer import ReplayBuffer, EpisodeTuple
from udrl.train_params import TrainParams
from udrl.util import preprocess_state, preprocess_info, get_state_size


class UdrlTrainer:
    def __init__(self, env, device, params: TrainParams = None):
        self.env = env
        self.device = device

        self.state_size = self.env.observation_space.shape[0]
        # self.state_size = 10
        self.action_size = self.env.action_space.n
        self.info_size = 3

        if params is None:
            self.params = TrainParams()
        else:
            self.params = params

    def get_action(self, policy, state: np.ndarray, command, info) -> int:
        state_input = torch.FloatTensor(np.ascontiguousarray(np.expand_dims(state, axis=(0, 1)))).to(self.device)
        command_input = torch.FloatTensor(command).to(self.device)
        info_input = torch.FloatTensor(info).to(self.device)

        action = policy(state_input, command_input, info_input)
        return action

    def step(self, action: int):
        state, reward, done, info = self.env.step(action)
        state = preprocess_state(state)
        info = preprocess_info(info)
        return state, reward, done, info

    def train(self, buffer=None, behavior=None, learning_history: list = None):
        if learning_history is None:
            learning_history = []

        if buffer is None:
            buffer = self.__initialize_replay_buffer()

        if behavior is None:
            behavior = self.__initialize_behavior_function()

        for i in range(1, self.params.n_main_iter + 1):
            time_start = time()
            mean_loss = self.__train_behavior(behavior, buffer)

            print(f"Iter: {i}, Loss: {mean_loss:.4f}, ", end="")

            # Sample exploratory commands and generate episodes
            buffer = self.__generate_episodes(behavior, buffer)

            print(f"Took: {time() - time_start:.4f}s")

            if i % self.params.evaluate_every == 0:
                command = buffer.sample_command(self.params.last_few)
                mean_return = self.__evaluate_agent(behavior, command)

                learning_history.append({
                    'training_loss': mean_loss,
                    'desired_return': command[0],
                    'desired_horizon': command[1],
                    'actual_return': mean_return,
                })

                if self.params.save_on_eval:
                    behavior.save('behavior_latest.pth')
                    buffer.save('buffer_latest.npy')
                    np.save('history_latest.npy', np.array(learning_history, dtype=object))

                if self.params.stop_on_solved and mean_return >= self.params.target_return:
                    break

        return behavior, buffer, learning_history

    def __evaluate_agent(self, behavior, command, render=False):
        behavior.eval()

        print('\nEvaluation.', end=' ')

        desired_return = command[0]
        desired_horizon = command[1]

        print(f'Desired return: {desired_return:.2f}, Desired horizon: {desired_horizon:.2f}.', end=' ')

        all_rewards = []

        for e in range(self.params.n_evals):
            done = False
            total_reward = 0
            state = preprocess_state(self.env.reset())
            info = [0.0, 0.0, 0.0]

            while not done:
                if render:
                    self.env.render()

                action = self.get_action(behavior.greedy_action, state, command, info)
                next_state, reward, done, info = self.step(action)

                total_reward += reward
                state = next_state

                desired_return = min(desired_return - reward, self.params.max_reward)
                desired_horizon = max(desired_horizon - 1, 1)

                command = [desired_return, desired_horizon]

            if render:
                self.env.close()

            all_rewards.append(total_reward)

        mean_return = np.mean(all_rewards)
        print(f'Reward achieved: {mean_return:.2f}')

        behavior.train()

        return mean_return

    def __generate_episode(self, policy, init_command: list = None) -> EpisodeTuple:
        if init_command is None:
            init_command = [1, 1]

        command = init_command.copy()
        desired_return = command[0]
        desired_horizon = command[1]

        states = []
        actions = []
        infos = []
        rewards = []

        time_steps = 0
        done = False
        state = preprocess_state(self.env.reset())
        info = [0.0, 0.0, 0.0]

        while not done:
            action = self.get_action(policy, state, command, info)
            next_state, reward, done, next_info = self.step(action)

            if not done and time_steps % self.params.skip_every_n_observations != 0:
                time_steps += 1
                continue

            if not done and time_steps > self.params.max_steps:
                done = True
                reward = self.params.max_steps_reward

            states.append(state)
            infos.append(info)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            info = next_info

            # Clipped such that it's upper-bounded by the maximum return achievable in the env
            desired_return = min(desired_return - reward, self.params.max_reward)

            # Make sure it's always a valid horizon
            desired_horizon = max(desired_horizon - 1, 1)

            command = [desired_return, desired_horizon]
            time_steps += 1

        return EpisodeTuple(states, actions, infos, rewards, init_command, sum(rewards), len(states))

    def __initialize_replay_buffer(self) -> ReplayBuffer:
        def random_policy(*_):
            return np.random.randint(self.env.action_space.n)

        buffer = ReplayBuffer()

        for i in range(self.params.n_warm_up_episodes):
            command = buffer.sample_command(self.params.last_few)
            episode = self.__generate_episode(random_policy, command)  # See Algorithm 2
            buffer.append(episode)

        buffer.sort()
        return buffer[:self.params.replay_size]

    def __initialize_behavior_function(self) -> Behavior:
        behavior = Behavior(
            action_size=self.action_size,
            info_size=self.info_size,
            device=self.device,
            command_scale=[self.params.return_scale, self.params.horizon_scale]
        )

        behavior.init_optimizer(lr=self.params.learning_rate)

        return behavior

    def __generate_episodes(self, behavior, buffer: ReplayBuffer):
        for i in range(self.params.n_episodes_per_iter):
            command = buffer.sample_command(self.params.last_few)
            episode = self.__generate_episode(behavior.action, command)  # See Algorithm 2
            buffer.append(episode)

        buffer.sort()
        return buffer[:self.params.replay_size]

    def __train_behavior(self, behavior: Behavior, buffer: ReplayBuffer) -> np.float64:
        all_loss = []
        for update in range(self.params.n_updates_per_iter):
            episodes = buffer.random_batch(self.params.batch_size)

            batch_states = np.ndarray(shape=(self.params.batch_size, 1) + get_state_size())
            batch_commands = np.ndarray(shape=(self.params.batch_size, 2), dtype=np.int32)
            batch_actions = np.ndarray(shape=(self.params.batch_size,), dtype=np.uint8)
            batch_info = np.ndarray(shape=(self.params.batch_size, 3), dtype=np.float32)

            for i, episode in enumerate(episodes):
                # noinspection PyPep8Naming
                T = episode.length
                t1 = np.random.randint(0, T)
                t2 = np.random.randint(t1 + 1, T + 1)
                dr = sum(episode.rewards[t1:t2])
                dh = t2 - t1

                batch_states[i][0] = episode.states[t1]
                batch_actions[i] = episode.actions[t1]
                batch_commands[i] = np.array([dr, dh], dtype=np.int32)
                batch_info[i] = episode.infos[t1]

            batch_states_in = torch.FloatTensor(np.array(batch_states)).to(self.device)
            batch_commands_in = torch.FloatTensor(batch_commands).to(self.device)
            batch_info_in = torch.FloatTensor(batch_info).to(self.device)

            batch_actions_in = torch.LongTensor(batch_actions).to(self.device)

            pred = behavior.forward(batch_states_in, batch_commands_in, batch_info_in)

            loss = nn_functional.cross_entropy(pred, batch_actions_in)

            behavior.optim.zero_grad()
            loss.backward()
            behavior.optim.step()

            all_loss.append(loss.item())

        # noinspection PyTypeChecker
        return np.mean(all_loss)
