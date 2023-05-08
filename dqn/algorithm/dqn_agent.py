import math
import os
import random
from itertools import count
from pathlib import Path

import torch
import torch.nn.functional as functional
import tqdm as tqdm
from torch import optim

from dqn.algorithm.environment import Environment
from dqn.algorithm.model.neural_network import NeuralNetwork
from dqn.algorithm.model.replay_memory import ReplayMemory, Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DqnAgent:
    def __init__(self, state_size, batch_size, gamma, replay_memory_size, target_update, n_step):
        """
        Deep Q-Network Agent using a replay buffer.
        @param state_size: size of the observation space.
        @param batch_size: number of observations batched into one back propagation.
        @param gamma: dqn discount factor.
        @param replay_memory_size: size of the replay buffer the Q-Network learns from.
        @param target_update: the number of episodes the target network is updated after.
        @param n_step: ToDo -> understand what this seems to be, its closely related to gamma
        """

        # Memory initialization
        self.replay_memory_size = replay_memory_size
        self.memory = ReplayMemory(replay_memory_size)

        # Configure internal neural networks
        self.policy_net: NeuralNetwork = NeuralNetwork(state_size).to(device)
        self.target_net: NeuralNetwork = NeuralNetwork(state_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.target_update = target_update

        # Mathematical values
        self.n_step = n_step
        self.batch_size = batch_size
        self.gamma = gamma
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 500

        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()

        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                self.policy_net.eval()
                action = self.policy_net(state).max(1)[1].view(1, 1)
                self.policy_net.train()
                return action
        else:
            return torch.tensor([[random.randrange(3)]], device=device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        # Using policy-net, we calculate the action-value of the previous actions we have taken before.
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * (self.gamma ** self.n_step)) + reward_batch

        # Compute Huber loss
        loss = functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self, environment: Environment, num_episodes=50):
        for i_episode in tqdm.tqdm(range(num_episodes)):
            # Initialize the environment and state
            environment.reset()
            state = torch.tensor([environment.get_current_state()], dtype=torch.float, device=device)
            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                done, reward, next_state = environment.step(action.item())

                reward = torch.tensor([reward], dtype=torch.float, device=device)

                if next_state is not None:
                    next_state = torch.tensor([next_state], dtype=torch.float, device=device)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                if not done:
                    state = torch.tensor([environment.get_current_state()], dtype=torch.float, device=device)

                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                if done:
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        print('Complete')

    def save_model(self, dir_path: Path):
        gamma_str = str(self.gamma).replace('.', '')
        param_str = f'BS{self.batch_size}_G{gamma_str}_RMS{self.replay_memory_size}_TU{self.target_update}_NS{self.n_step}'
        model_name = f'model_{param_str}.pkl'
        model = self.policy_net.state_dict()
        path = os.path.join(dir_path, model_name)
        torch.save(model, path)