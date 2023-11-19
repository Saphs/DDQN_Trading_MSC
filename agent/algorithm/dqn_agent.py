import hashlib
import logging
import math
import os
import random
from datetime import datetime
from itertools import count
from multiprocessing import Process
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
import tqdm as tqdm
from pandas import DataFrame
from torch import optim, Tensor, nn
from torch._C._profiler import ProfilerActivity
from torch.autograd.profiler import record_function
from torch.profiler import profile

from agent.algorithm.config_parsing.agent_parameters import AgentParameters
from agent.algorithm.config_parsing.dqn_config import DqnConfig
from agent.algorithm.environment import Environment
from agent.algorithm.model.neural_network import NeuralNetwork
from agent.algorithm.model.replay_memory import ReplayMemory, Transition
from agent.chart_builder import ChartBuilder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _are_equal(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

def _model_hash(model_dict):
    hash = hashlib.md5(f"{model_dict}".encode("UTF-8")).hexdigest()
    return hash

class DqnAgent:
    activation = {}

    @classmethod
    def init(cls, state_size: int, config: DqnConfig, name: str = None):
        c: AgentParameters = config.agent
        return DqnAgent(
            state_size,
            batch_size=config.batch_size,
            gamma=c.gamma,
            replay_memory_size=c.replay_memory_size,
            memory_sample_count=c.memory_sample_count,
            target_update=c.target_net_update_interval,
            alpha=c.alpha,
            eps_start=c.epsilon_start,
            eps_end=c.epsilon_end,
            eps_decay=c.epsilon_decay,
            name=name,
            check_points=config.check_points
        )

    def __init__(self, state_size, batch_size, gamma, alpha, eps_start, eps_end, eps_decay, replay_memory_size, memory_sample_count, target_update, name: str = None, check_points: int = 100):
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
        self.memory_sample_count = memory_sample_count
        self.memory = ReplayMemory(replay_memory_size)

        # Configure internal neural networks
        self.policy_net: NeuralNetwork = NeuralNetwork(state_size).to(device)
        logging.info(f"Network init hash: {_model_hash(self.policy_net.state_dict())}")
        #self.policy_net.load_state_dict(torch.load("C:\\Users\\tizia\\PycharmProjects\\DDQN_Trading_MSC\\dqn_legacy_code\\legacy_init.pkl"))
        self.target_net: NeuralNetwork = NeuralNetwork(state_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.alpha = alpha
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha)
        self.huber_loss = nn.SmoothL1Loss()
        self.target_update = target_update

        # Mathematical values
        self.batch_size = batch_size
        self.gamma = gamma
        self.EPS_START = eps_start
        self.EPS_END = eps_end
        self.EPS_DECAY = eps_decay
        self.eps_threshold = self.EPS_START

        # Internal state used for bookkeeping
        self.check_points = check_points
        self.target_path = "."
        self.steps_done = 0
        self.progress_df: DataFrame = pd.DataFrame(data={"episode": [], "steps": [], "avg_reward": [], "avg_loss": [], "avg_td_error": [], "capital": [], "sma_reward": [], "updated":[]})
        self.reward_sum: Tensor = torch.tensor(0, dtype=torch.float, device=device)
        self.loss_sum = 0
        self.td_error_sum: Tensor = torch.tensor(0, dtype=torch.float, device=device)
        self.last_tau_update = 0
        self.max_avg_reward = float('-inf')
        self.best_model: tuple[int, dict] = (0, self.policy_net.state_dict().copy())
        self.cb = ChartBuilder()


    def select_action(self, state) -> Tensor:
        self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

        if random.random() > self.eps_threshold:
            """
            This is mighty strange - this should be replaced by:
            return self.policy_net(state).max(1)[1].unsqueeze(1)
            """
            #self.policy_net.eval()
            action = self.policy_net(state).max(1)[1].unsqueeze(1)
            #self.policy_net.train()
            return action
        else:
            return torch.randint(0, 3, size=(self.batch_size, 1), device=device, dtype=torch.int64)

    def train(self, environment: Environment, num_episodes) -> DataFrame:
        self.cb.set_target_path(Path(self.target_path))
        updated = True
        p_bar = tqdm.tqdm(range(num_episodes), ncols=120, unit="ep")
        for i_episode in p_bar:
            #with record_function("Episode"):
            p_bar.set_postfix({
                "step": self.steps_done,
                "rwrd": self.max_avg_reward,
                "tau": self.last_tau_update,
                "eps": self.eps_threshold
            })
            #print(environment.current_state)
            for t, state_batch in enumerate(environment):
                #with record_function("Select_action"):
                action_batch = self.select_action(state_batch)
                #with record_function("Calc_Reward"):
                _, reward_batch, next_state_batch = environment.act(action_batch)

                #print(f"\n{state_batch=}\n{action_batch=}\n{reward_batch=}\n{next_state_batch=}")

                #with record_function("Optimize_model"):
                self.memory.push(state_batch, action_batch, next_state_batch, reward_batch)
                loss = self.optimize_model()

                #with record_function("Sum_analytics"):
                # Update the target network, copying all weights and biases in DQN
                if self.steps_done % self.target_update == 0:
                    self.last_tau_update = i_episode
                    self.target_net.load_state_dict(self.best_model[1].copy())
                    updated = True

                # Keep some analytical values
                self.reward_sum += torch.sum(reward_batch)
                if loss is not None:
                    self.loss_sum += loss.item()

            #with record_function("Gen_Plots"):
            # Keep some more analytical values
            self.protocol(i_episode, environment, updated)
            updated = False
            if i_episode % self.check_points == 0:
                self._plot()


        # Save resulting models
        self._save_last_model()
        self._save_best_model()

        return self.progress_df

    def _plot(self):
        self.cb.plot_loss(self.progress_df, self.target_update)
        self.cb.plot_capital(self.progress_df, self.target_update)
        self.cb.plot_rewards(self.progress_df, self.target_update)
        self.cb.plot_td_error(self.progress_df)
        self._save_checkpoint_model()

    def optimize_model(self):
        #print(f"{len(self.memory)=} | {self.batch_size=} => {(len(self.memory) < self.batch_size)=}")
        if len(self.memory) < self.memory_sample_count:
            return
        s, a, s_prime, r = self.memory.sample(self.memory_sample_count)

        # Calc Q_pol(s, a) values (The value of taking action a in state s)
        #print(f"{s_prime=}, {s=}, {a=}, {r=}")
        current_q = self.policy_net(s).gather(1, a)
        #raise RuntimeError("HARDSTOP")
        # Calc max_a( Q_tar(s', a) ) values (The max value achievable in state s' taking the 'best' action a)
        max_q_prime = self.target_net(s_prime).max(1)[0].unsqueeze(1)
        # Y_dqn = R + γ * max_a( Q_tar(s', a) )
        target_q = r + self.gamma * max_q_prime

        # Optimize the model
        loss: Tensor = functional.smooth_l1_loss(input=current_q, target=target_q)
        self.td_error_sum = self.td_error_sum + target_q.sum()
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss

    def protocol(self, i_ep: int, env: Environment, updated: bool):
        #print(self.reward_sum.item(), len(env))
        avg_reward: float = self.reward_sum.item() / len(env)
        self.reward_sum = torch.tensor(0, dtype=torch.float, device=device)
        avg_loss: float = self.loss_sum / len(env)
        self.loss_sum = 0
        avg_td_error: float = self.td_error_sum.item() / len(env)
        self.td_error_sum = torch.tensor(0, dtype=torch.float, device=device)

        #cap = env.dyn_context['current_capital'].item()
        cap = env.dyn_context['current_capital']
        self.progress_df.loc[len(self.progress_df)] = [i_ep, self.steps_done, avg_reward, avg_loss, avg_td_error,
                                                       cap, 0, str(updated)]
        if avg_reward > self.max_avg_reward:
            #print(f"New max reward found: {avg_reward}, capital: {env.dyn_context['current_capital']}")
            self.best_model = (i_ep, self.policy_net.state_dict().copy())
            self.max_avg_reward = avg_reward

    def set_target_path(self, dir_path: Path):
        self.target_path = dir_path
        os.makedirs(os.path.join(self.target_path, f'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.target_path, f'final'), exist_ok=True)

    def _save_checkpoint_model(self):
        suffix = str(int(datetime.now().utcnow().timestamp()))
        path = os.path.join(self.target_path, f'checkpoints/model_{suffix}.pkl')
        torch.save(self.policy_net.state_dict(), path)

    def _save_last_model(self):
        logging.info(f"Saving last model under: {self.target_path}")
        path = os.path.join(self.target_path, f'final/model_{self.name}.pkl')
        torch.save(self.policy_net.state_dict(), path)

    def _save_best_model(self):
        logging.info(f"Saving best model under: {self.target_path}")
        path = os.path.join(self.target_path, f'final/best_{self.best_model[0]}_{self.name}.pkl')
        torch.save(self.best_model[1], path)

