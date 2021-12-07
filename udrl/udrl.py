import numpy as np
import torch
import torch.nn.functional as nn_functional

from udrl.behavior import Behavior
from udrl.replay_buffer import ReplayBuffer
from udrl.train_params import TrainParams
from udrl.util import make_episode


class UDRL:

    def __init__(self, env, device, params: TrainParams = None):
        self.env = env
        self.device = device

        # TODO: Add size from info
        # self.state_size = self.env.observation_space.shape[0] + 8
        self.state_size = 10
        self.action_size = self.env.action_space.n

        if params is None:
            self.params = TrainParams()
        else:
            self.params = params

    def get_action(self, policy, state, command) -> int:
        state_input = torch.FloatTensor(state).to(self.device)
        command_input = torch.FloatTensor(command).to(self.device)

        action = policy(state_input, command_input)
        return action

    def step(self, action: int):
        state, reward, done, info = self.env.step(action)
        state = self.__preprocess_state(state, info)
        return state, reward, done

    def train(self, buffer=None, behavior=None, learning_history: list = None):
        if learning_history is None:
            learning_history = []

        if buffer is None:
            buffer = self.__initialize_replay_buffer()

        if behavior is None:
            behavior = self.__initialize_behavior_function()

        for i in range(1, self.params.n_main_iter + 1):
            mean_loss = self.__train_behavior(behavior, buffer)

            print(f"Iter: {i}, Loss: {mean_loss:.4f}")

            # Sample exploratory commands and generate episodes
            self.__generate_episodes(behavior, buffer)

            if i % self.params.evaluate_every == 0:
                command = self.__sample_command(buffer)
                mean_return = self.__evaluate_agent(behavior, command)

                learning_history.append({
                    'training_loss': mean_loss,
                    'desired_return': command[0],
                    'desired_horizon': command[1],
                    'actual_return': mean_return,
                })

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
            self.env.reset()
            state, reward, _ = self.step(0)  # NOOP

            while not done:
                if render:
                    self.env.render()

                state_input = torch.FloatTensor(state).to(self.device)
                command_input = torch.FloatTensor(command).to(self.device)

                action = behavior.greedy_action(state_input, command_input)
                next_state, reward, done = self.step(action)

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

    def __generate_episode(self, policy, init_command: list = None):
        if init_command is None:
            init_command = [1, 1]

        command = init_command.copy()
        desired_return = command[0]
        desired_horizon = command[1]

        states = []
        actions = []
        rewards = []

        time_steps = 0
        done = False
        self.env.reset()
        state, reward, _ = self.step(0)  # NOOP

        while not done:
            action = self.get_action(policy, state, command)
            next_state, reward, done = self.step(action)

            if not done and time_steps > self.params.max_steps:
                done = True
                reward = self.params.max_steps_reward

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = next_state

            # Clipped such that it's upper-bounded by the maximum return achievable in the env
            desired_return = min(desired_return - reward, self.params.max_reward)

            # Make sure it's always a valid horizon
            desired_horizon = max(desired_horizon - 1, 1)

            command = [desired_return, desired_horizon]
            time_steps += 1

        return make_episode(states, actions, rewards, init_command, sum(rewards), time_steps)

    def __initialize_replay_buffer(self):
        def random_policy(_1, _2):
            return np.random.randint(self.env.action_space.n)

        buffer = ReplayBuffer(self.params.replay_size)

        for i in range(self.params.n_warm_up_episodes):
            command = self.__sample_command(buffer)
            episode = self.__generate_episode(random_policy, command)  # See Algorithm 2
            buffer.add(episode)

        buffer.sort()
        return buffer

    def __initialize_behavior_function(self):
        behavior = Behavior(self.state_size,
                            self.action_size,
                            self.device,
                            [self.params.return_scale, self.params.horizon_scale])

        behavior.init_optimizer(lr=self.params.learning_rate)

        return behavior

    def __generate_episodes(self, behavior, buffer):
        for i in range(self.params.n_episodes_per_iter):
            command = self.__sample_command(buffer)
            episode = self.__generate_episode(behavior.action, command)  # See Algorithm 2
            buffer.add(episode)

        buffer.sort()

    def __train_behavior(self, behavior, buffer):
        all_loss = []
        for update in range(self.params.n_updates_per_iter):
            episodes = buffer.random_batch(self.params.batch_size)

            batch_states = []
            batch_commands = []
            batch_actions = []

            for episode in episodes:
                # noinspection PyPep8Naming
                T = episode.length
                t1 = np.random.randint(0, T)
                t2 = np.random.randint(t1 + 1, T + 1)
                dr = sum(episode.rewards[t1:t2])
                dh = t2 - t1

                st1 = episode.states[t1]
                at1 = episode.actions[t1]

                batch_states.append(st1)
                batch_actions.append(at1)
                batch_commands.append([dr, dh])

            batch_states = torch.FloatTensor(batch_states).to(self.device)
            batch_commands = torch.FloatTensor(batch_commands).to(self.device)
            batch_actions = torch.LongTensor(batch_actions).to(self.device)

            pred = behavior(batch_states, batch_commands)

            loss = nn_functional.cross_entropy(pred, batch_actions)

            behavior.optim.zero_grad()
            loss.backward()
            behavior.optim.step()

            all_loss.append(loss.item())

        return np.mean(all_loss)

    def __sample_command(self, buffer):
        if len(buffer) == 0:
            return [1, 1]

        commands = buffer.get(self.params.last_few)

        lengths = [command.length for command in commands]
        desired_horizon = np.round(np.mean(lengths))

        returns = [command.total_return for command in commands]
        mean_return, std_return = np.mean(returns), np.std(returns)
        desired_return = np.random.uniform(mean_return, mean_return + std_return)

        return [desired_return, desired_horizon]

    @staticmethod
    def __preprocess_state(state: np.ndarray, info: dict):
        # TODO: Find a way to join state and info

        info["flag_get"] = float(info["flag_get"])
        if info["status"] == "small":
            info["status"] = 0.25
        elif info["status"] == "tall":
            info["status"] = 0.75
        elif info["status"] == "fireball":
            info["status"] = 1.0
        else:
            info["status"] = 0.0
        return list(info.values())
