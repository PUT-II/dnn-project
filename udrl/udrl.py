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

        if params is None:
            self.params = TrainParams()
        else:
            self.params = params

    @staticmethod
    def preprocess_state(state):
        state["flag_get"] = float(state["flag_get"])
        if state["status"] == "small":
            state["status"] = 0.25
        elif state["status"] == "tall":
            state["status"] = 0.75
        elif state["status"] == "fireball":
            state["status"] = 1.0
        else:
            state["status"] = 0.0
        return list(state.values())

    def generate_episode(self, policy, init_command: list = None):
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
        _, reward, _, state = self.env.step(0)  # NOOP
        state = self.preprocess_state(state)

        while not done:
            state_input = torch.FloatTensor(state).to(self.device)
            command_input = torch.FloatTensor(command).to(self.device)
            action = policy(state_input, command_input)
            _, reward, done, next_state = self.env.step(action)

            if not done and time_steps > self.params.max_steps:
                done = True
                reward = self.params.max_steps_reward

            states.append(state)
            actions.append(action)
            rewards.append(reward)

            state = self.preprocess_state(next_state)

            # Clipped such that it's upper-bounded by the maximum return achievable in the env
            desired_return = min(desired_return - reward, self.params.max_reward)

            # Make sure it's always a valid horizon
            desired_horizon = max(desired_horizon - 1, 1)

            command = [desired_return, desired_horizon]
            time_steps += 1

        return make_episode(states, actions, rewards, init_command, sum(rewards), time_steps)

    @staticmethod
    def sample_command(buffer, last_few):
        if len(buffer) == 0:
            return [1, 1]

        commands = buffer.get(last_few)

        lengths = [command.length for command in commands]
        desired_horizon = np.round(np.mean(lengths))

        returns = [command.total_return for command in commands]
        mean_return, std_return = np.mean(returns), np.std(returns)
        desired_return = np.random.uniform(mean_return, mean_return + std_return)

        return [desired_return, desired_horizon]

    def initialize_replay_buffer(self, replay_size, n_episodes, last_few):
        def random_policy(_1, _2):
            return np.random.randint(self.env.action_space.n)

        buffer = ReplayBuffer(replay_size)

        for i in range(n_episodes):
            command = self.sample_command(buffer, last_few)
            episode = self.generate_episode(random_policy, command)  # See Algorithm 2
            buffer.add(episode)

        buffer.sort()
        return buffer

    def initialize_behavior_function(
            self,
            state_size,
            action_size,
            learning_rate,
            command_scale
    ):
        behavior = Behavior(state_size,
                            action_size,
                            self.device,
                            command_scale)

        behavior.init_optimizer(lr=learning_rate)

        return behavior

    def generate_episodes(self, behavior, buffer, last_few):
        def stochastic_policy(state_, command_):
            behavior.action(state_, command_)

        for i in range(self.params.n_episodes_per_iter):
            command = self.sample_command(buffer, last_few)
            episode = self.generate_episode(stochastic_policy, command)  # See Algorithm 2
            buffer.add(episode)

        buffer.sort()

    def evaluate_agent(self, behavior, command, render=False):
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
            _, reward, _, state = self.env.step(0)  # NOOP
            state = self.preprocess_state(state)

            while not done:
                if render:
                    self.env.render()

                state_input = torch.FloatTensor(state).to(self.device)
                command_input = torch.FloatTensor(command).to(self.device)

                action = behavior.greedy_action(state_input, command_input)
                _, reward, done, next_state = self.env.step(action)

                total_reward += reward
                state = self.preprocess_state(next_state)

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

    def train(self, buffer=None, behavior=None, learning_history: list = None):
        if learning_history is None:
            learning_history = []

        if buffer is None:
            buffer = self.initialize_replay_buffer(self.params.replay_size, self.params.n_warm_up_episodes,
                                                   self.params.last_few)

        if behavior is None:
            # TODO: Get state size from env
            # state_size = env.observation_space.shape[0]
            state_size = 10
            action_size = self.env.action_space.n
            behavior = self.initialize_behavior_function(state_size,
                                                         action_size,
                                                         self.params.learning_rate,
                                                         [self.params.return_scale, self.params.horizon_scale])

        for i in range(1, self.params.n_main_iter + 1):
            mean_loss = self.train_behavior(behavior, buffer, self.params.n_updates_per_iter, self.params.batch_size)

            print(f"Iter: {i}, Loss: {mean_loss:.4f}")

            # Sample exploratory commands and generate episodes
            self.generate_episodes(behavior,
                                   buffer,
                                   self.params.last_few)

            if i % self.params.evaluate_every == 0:
                command = self.sample_command(buffer, self.params.last_few)
                mean_return = self.evaluate_agent(behavior, command)

                learning_history.append({
                    'training_loss': mean_loss,
                    'desired_return': command[0],
                    'desired_horizon': command[1],
                    'actual_return': mean_return,
                })

                if self.params.stop_on_solved and mean_return >= self.params.target_return:
                    break

        return behavior, buffer, learning_history

    def train_behavior(self, behavior, buffer, n_updates, batch_size):
        all_loss = []
        for update in range(n_updates):
            episodes = buffer.random_batch(batch_size)

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
