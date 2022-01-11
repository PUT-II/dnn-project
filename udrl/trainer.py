from time import time

import numpy as np
import torch
import torch.nn.functional as nn_functional

from udrl.behavior import Behavior
from udrl.replay_buffer import ReplayBuffer, EpisodeTuple
from udrl.train_params import TrainParams
from udrl.util import preprocess_state, preprocess_info, get_state_size, clip_reward, get_state_channels


class UdrlTrainer:
    def __init__(self, envs, device, params: TrainParams = None):
        self.envs = envs
        self.device = device

        self.state_size = self.envs[0].observation_space.shape[0]
        # self.state_size = 10
        self.action_size = self.envs[0].action_space.n
        self.info_size = 3

        if params is None:
            self.params = TrainParams()
        else:
            self.params = params

    def get_action(self, policy, state: np.ndarray, command, info) -> int:
        if get_state_channels() == 1:
            expanded_state = np.ascontiguousarray(np.expand_dims(state, axis=(0, 1)))
        else:
            expanded_state = np.ascontiguousarray(np.expand_dims(state, axis=(0,)))

        state_input = torch.FloatTensor(expanded_state).to(self.device)
        command_input = torch.FloatTensor(command).to(self.device)
        info_input = torch.FloatTensor(info).to(self.device)

        action = policy(state_input, command_input, info_input)
        return action

    @staticmethod
    def step(env, action: int):
        state, reward, done, info = env.step(action)
        state = preprocess_state(state)
        info = preprocess_info(info)
        return state, reward, done, info

    def train(self, buffer=None, behavior=None, learning_history: list = None):
        if learning_history is None:
            start_iter = 1
            learning_history = []
        else:
            start_iter = max(learning_history, key=lambda e: e['iter'])['iter']
            start_iter += 1

        if buffer is None:
            buffer = self.__initialize_replay_buffer()

        if behavior is None:
            behavior = self.initialize_behavior_function()

        for i in range(start_iter, self.params.n_main_iter + 1):
            time_start = time()
            print(f"Iter: {i}, ", end="")
            mean_loss = self.__train_behavior(behavior, buffer)

            print(f"Loss: {mean_loss:.4f}, ", end="")

            # Sample exploratory commands and generate episodes
            buffer = self.__generate_episodes(behavior, buffer)

            print(f"Took: {time() - time_start:.4f}s")

            if i % self.params.evaluate_every == 0:
                command = buffer.sample_command(self.params.last_few)
                mean_return = self.__evaluate_behavior(behavior, command)

                learning_history.append({
                    'iter': i,
                    'training_loss': mean_loss,
                    'desired_return': command[0],
                    'desired_horizon': command[1],
                    'actual_return': mean_return,
                })

                if self.params.save_on_eval:
                    behavior.save(f'behavior_{i}.pth')
                    behavior.save('behavior_latest.pth')
                    buffer.save('buffer_latest.npy')
                    np.save('history_latest.npy', np.array(learning_history, dtype=object))

                if self.params.stop_on_solved and mean_return >= self.params.target_return:
                    break

        return behavior, buffer, learning_history

    def initialize_behavior_function(self) -> Behavior:
        behavior = Behavior(
            action_size=self.action_size,
            info_size=self.info_size,
            device=self.device,
            state_channels=get_state_channels(),
            command_scale=[self.params.return_scale, self.params.horizon_scale]
        )

        behavior.init_optimizer(lr=self.params.learning_rate)

        return behavior

    def __evaluate_behavior(self, behavior, command, render=False):
        behavior.eval()

        print('\nEvaluation.', end=' ')

        desired_return = command[0]
        desired_horizon = command[1]

        print(f'Desired return: {desired_return:.2f}, Desired horizon: {desired_horizon:.2f}.', end=' ')

        env = self.envs[0]
        all_rewards = []

        for e in range(self.params.n_evals):
            done = False
            total_reward = 0
            state = preprocess_state(env.reset())
            info = [0.0, 0.0, 0.0]

            while not done:
                if render:
                    env.render()

                action = self.get_action(behavior.action, state, command, info)
                next_state, reward, done, info = self.step(env, action)

                total_reward += reward
                state = next_state

                desired_horizon = max(desired_horizon - 1, 1)
                min_reward = self.params.min_reward * desired_horizon
                max_reward = self.params.max_reward * desired_horizon
                desired_return = clip_reward(desired_return - reward, min_reward, max_reward)

                command = [desired_return, desired_horizon]

            if render:
                env.close()

            all_rewards.append(total_reward)

        mean_return = np.mean(all_rewards)
        print(f'Reward achieved: {mean_return:.2f}')

        behavior.train()

        return mean_return

    def __generate_episode(self, env, policy, init_command: list) -> EpisodeTuple:
        command = init_command.copy()
        desired_return = command[0]
        desired_horizon = command[1]

        states = []
        actions = []
        infos = []
        rewards = []

        time_steps = 0
        done = False
        state = preprocess_state(env.reset())
        info = [0.0, 0.0, 0.0]

        while not done:
            action = self.get_action(policy, state, command, info)
            next_state, reward, done, next_info = self.step(env, action)

            if not done and time_steps % self.params.skip_every_n_observations != 0:
                time_steps += 1
                continue

            if not done and time_steps > self.params.max_steps:
                done = True
                reward = self.params.max_steps_penalty

            states.append(state)
            infos.append(info)
            actions.append(action)
            rewards.append(reward)

            state = next_state
            info = next_info

            # Make sure it's always a valid horizon
            desired_horizon = max(desired_horizon - 1, 1)

            # Clipped such that it's upper-bounded by the maximum return achievable in the env
            desired_horizon = max(desired_horizon - 1, 1)
            min_reward = self.params.min_reward * desired_horizon
            max_reward = self.params.max_reward * desired_horizon
            desired_return = clip_reward(desired_return - reward, min_reward, max_reward)

            command = [desired_return, desired_horizon]
            time_steps += 1

        return EpisodeTuple(states, actions, infos, rewards, init_command, sum(rewards), len(states))

    def __initialize_replay_buffer(self) -> ReplayBuffer:
        def random_policy(*_):
            return np.random.randint(self.envs[0].action_space.n)

        buffer = ReplayBuffer()

        for i in range(self.params.n_warm_up_episodes):
            command = buffer.sample_command(self.params.last_few)
            episode = self.__generate_episode(np.random.choice(self.envs), random_policy, command)  # See Algorithm 2
            buffer.append(episode)

        buffer.sort()
        return buffer[:self.params.replay_size]

    def __generate_episodes(self, behavior, buffer: ReplayBuffer):
        for i in range(self.params.n_episodes_per_iter):
            env = np.random.choice(self.envs)
            command = buffer.sample_command(self.params.last_few)
            episode = self.__generate_episode(env, behavior.action, command)  # See Algorithm 2
            buffer.append(episode)

        buffer.sort()
        return buffer[:self.params.replay_size]

    def __train_behavior(self, behavior: Behavior, buffer: ReplayBuffer) -> np.float64:
        all_loss = []
        for update in range(self.params.n_updates_per_iter):
            episodes = buffer.random_batch(self.params.batch_size)

            channels = get_state_channels()
            batch_states = np.ndarray(shape=(self.params.batch_size, channels) + get_state_size())
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

                if channels == 1:
                    batch_states[i][0] = episode.states[t1]
                else:
                    batch_states[i] = episode.states[t1]
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
