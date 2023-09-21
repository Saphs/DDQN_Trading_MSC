import random
import torch
import torch.nn.functional as F

from dqn_legacy_code.DataLoader.DataAutoPatternExtractionAgent import DataAutoPatternExtractionAgent
from dqn_legacy_code.VanillaInput.DeepQNetwork import NeuralNetwork
from dqn_legacy_code.ReplayMemory import ReplayMemory, Transition

from itertools import count
from tqdm import tqdm
import math
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self,
                 dataset_name,
                 state_mode=1,
                 transaction_cost=0.0,
                 BATCH_SIZE=30,
                 GAMMA=0.7,
                 ReplayMemorySize=50,
                 TARGET_UPDATE=5,
                 n_step=10):
        """
        This class is the base class for training across multiple models in the DeepRLAgent directory.
        @param dataset_name: for using in the name of the result file
        @param state_mode: for using in the name of the result file
        @param transaction_cost: for using in the name of the result file
        @param BATCH_SIZE: batch size for batch training
        @param GAMMA: in the algorithm
        @param ReplayMemorySize: size of the replay buffer
        @param TARGET_UPDATE: hard update policy network into target network every TARGET_UPDATE iterations
        @param n_step: for using in the name of the result file
        """
        self.DATASET_NAME = dataset_name
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.ReplayMemorySize = ReplayMemorySize
        self.transaction_cost = transaction_cost
        self.state_mode = state_mode

        self.policy_net: NeuralNetwork
        self.target_net: NeuralNetwork

        self.TARGET_UPDATE = TARGET_UPDATE
        self.n_step = n_step

        self.memory = ReplayMemory(ReplayMemorySize)

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
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
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
        #print(state_batch)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        #raise RuntimeError("HARD SSTOPOPT")

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * (self.GAMMA ** self.n_step)) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self, environment: DataAutoPatternExtractionAgent, num_episodes=50):
        with open("states.text", 'w') as f:
            f.write(str(environment.states))
        for i_episode in tqdm(range(num_episodes)):
            # Initialize the environment and state
            environment.reset()
            state = torch.tensor([environment.get_current_state()], dtype=torch.float, device=device)
            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                done, reward, next_state = environment.step(action.item())

                reward = torch.tensor([reward], dtype=torch.float, device=device)

                #print(f"\n{state=}\n{action=}\n{reward=}\n{next_state=}")

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
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
        print('Complete')

    def save_model(self, dir_path: str):
        gamma_str = str(self.GAMMA).replace('.', '')
        param_str = f'BS{self.BATCH_SIZE}_G{gamma_str}_RMS{self.ReplayMemorySize}_TU{self.TARGET_UPDATE}_NS{self.n_step}'
        model_name = f'model_{param_str}.pkl'
        model = self.policy_net.state_dict()
        path = os.path.join(dir_path, model_name)
        torch.save(model, path)
