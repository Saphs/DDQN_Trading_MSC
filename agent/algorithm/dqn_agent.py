import math
import os
import random
from itertools import count
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as functional
import tqdm as tqdm
from pandas import DataFrame
from torch import optim, Tensor
from torch._C._profiler import ProfilerActivity
from torch.profiler import profile

from agent.algorithm.config_parsing.agent_parameters import AgentParameters
from agent.algorithm.config_parsing.dqn_config import DqnConfig
from agent.algorithm.environment import Environment
from agent.algorithm.model.neural_network import NeuralNetwork
from agent.algorithm.model.replay_memory import ReplayMemory, Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DqnAgent:

    @classmethod
    def init(cls, state_size: int, config: DqnConfig, name: str = None):
        c: AgentParameters = config.agent
        return DqnAgent(
            state_size,
            batch_size=config.batch_size,
            gamma=c.gamma,
            replay_memory_size=c.replay_memory_size,
            target_update=c.target_net_update_interval,
            alpha=c.alpha,
            eps_start=c.epsilon_start,
            eps_end=c.epsilon_end,
            eps_decay=c.epsilon_decay,
            name=name
        )

    def __init__(self, state_size, batch_size, gamma, alpha, eps_start, eps_end, eps_decay, replay_memory_size, target_update, name: str = None):
        """
        Deep Q-Network Agent using a replay buffer.
        @param state_size: size of the observation space.
        @param batch_size: number of observations batched into one back propagation.
        @param gamma: agent discount factor.
        @param replay_memory_size: size of the replay buffer the Q-Network learns from.
        @param target_update: the number of episodes the target network is updated after.
        """

        if name is None:
            characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            self.name = ''.join(random.choice(characters) for i in range(8))
        else:
            self.name = name

        # Memory initialization
        self.replay_memory_size = replay_memory_size
        self.memory = ReplayMemory(replay_memory_size)

        # Configure internal neural networks
        self.policy_net: NeuralNetwork = NeuralNetwork(state_size).to(device)
        self.target_net: NeuralNetwork = NeuralNetwork(state_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.alpha = alpha
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha)
        self.target_update = target_update

        # Mathematical values
        self.batch_size = batch_size
        self.gamma = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay

        self.steps_done = 0

    def select_action(self, state) -> Tensor:

        sample = random.random()

        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                #print(f"GOT INTO POLICY NET ! STATE_ {state}")
                action = self.policy_net(state).max(1)[1].unsqueeze(1)
                #print(f"GOT INTO POLICY NET ! ACTUIN_ {action}")
                return action
        else:
            return torch.randint(0, 3, size=(self.batch_size, 1), device=device, dtype=torch.int64)

    def train(self, environment: Environment, num_episodes) -> DataFrame:
        reward_df: DataFrame = pd.DataFrame(data={"episode": [], "avg_reward": [], "avg_loss": []})
        for i_episode in tqdm.tqdm(range(num_episodes), ncols=80):
            # Initialize the environment and state
            #environment.reset()
            reward_sum: Tensor = torch.tensor(0, dtype=torch.float, device=device)
            loss_sum = 0
            steps_done = 0
            for t, state_batch in enumerate(environment):
                action_batch = self.select_action(state_batch)
                done, reward_batch, next_state_batch = environment.step(action_batch)
                self.memory.push(state_batch, action_batch, next_state_batch, reward_batch)
                loss = self.optimize_model()

                # Move to the next state
                if not done:
                    #state_batch = environment.get_current_state()
                    reward_sum += torch.sum(reward_batch)
                    if loss is not None:
                        loss_sum += loss.item()
                else:
                    steps_done = t
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            avg_reward: float = reward_sum.item() / steps_done
            avg_loss: float = loss_sum / steps_done
            reward_df.loc[len(reward_df)] = [i_episode, avg_reward, avg_loss]
            #print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        return reward_df

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # next_states might be None for final states
        if batch.next_state[0] is None:
            return

        next_states = torch.cat(batch.next_state)
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        #print("-"*80)
        #print(next_states, state_batch, action_batch, reward_batch)
        #print("-" * 80)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        # Using policy-net, we calculate the action-value of the previous actions we have taken before.
        # aka.: Q(s, a; θ)
        state_q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        # aka.: Q(s', a'; θ_t)
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        #print(f"{next_q_values=}")
        #print("-" * 80)
        #print(f"{reward_batch=}")
        #print("-" * 80)
        # Compute the expected Q values, aka.: y_i
        target_values = reward_batch + self.gamma * next_q_values
        #print(f"{target_values=}")
        #print("-" * 80)

        # Compute Huber loss
        loss: Tensor = functional.smooth_l1_loss(state_q_values, target_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        #raise RuntimeError("Temprary stop")

        return loss

    def save_model(self, dir_path: Path):
        path = os.path.join(dir_path, f'model_{self.name}.pkl')
        torch.save(self.policy_net.state_dict(), path)
