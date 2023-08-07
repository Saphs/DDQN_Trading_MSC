import random
import torch
import torch.nn.functional as F

from Reimplementation.VanillaInput.DeepQNetwork import NeuralNetwork
from Reimplementation.ReplayMemory import ReplayMemory, Transition

# from DeepQNetwork import DQN
# from ReplayMemory import ReplayMemory, Transition


from itertools import count
from tqdm import tqdm
import math
import os

from pathlib import Path

from Reimplementation.PatternDetectionInCandleStick.Evaluation import Evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BaseTrain:
    def __init__(self,
                 data_train,
                 data_test,
                 dataset_name,
                 model_kind,
                 state_mode=1,
                 window_size=1,
                 transaction_cost=0.0,
                 BATCH_SIZE=30,
                 GAMMA=0.7,
                 ReplayMemorySize=50,
                 TARGET_UPDATE=5,
                 n_step=10):
        """
        This class is the base class for training across multiple models in the DeepRLAgent directory.
        @param data_train: of type DataAutoPatternExtractionAgent
        @param data_test: of type DataAutoPatternExtractionAgent
        @param dataset_name: for using in the name of the result file
        @param state_mode: for using in the name of the result file
        @param window_size: for using in the name of the result file
        @param transaction_cost: for using in the name of the result file
        @param BATCH_SIZE: batch size for batch training
        @param GAMMA: in the algorithm
        @param ReplayMemorySize: size of the replay buffer
        @param TARGET_UPDATE: hard update policy network into target network every TARGET_UPDATE iterations
        @param n_step: for using in the name of the result file
        """
        self.training_environment = data_train
        self.test_environment = data_test
        self.DATASET_NAME = dataset_name
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.ReplayMemorySize = ReplayMemorySize
        self.transaction_cost = transaction_cost
        self.model_kind = model_kind
        self.state_mode = state_mode
        self.window_size = window_size

        self.policy_net: NeuralNetwork
        self.target_net: NeuralNetwork

        self.TARGET_UPDATE = TARGET_UPDATE
        self.n_step = n_step

        self.memory = ReplayMemory(ReplayMemorySize)

        self.train_test_split = True if data_test is not None else False

        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 500

        self.steps_done = 0

        self.PATH = os.path.join(Path(os.path.abspath(os.path.dirname(__file__))).parent,
                                 f'Results/{self.DATASET_NAME}/'
                                 f'{self.model_kind}; '
                                 f'StateMode({self.state_mode}); '
                                 f'WindowSize({self.window_size}); '
                                 f'BATCH_SIZE{self.BATCH_SIZE}; '
                                 f'GAMMA{self.GAMMA}; '
                                 f'REPLAY_MEMORY_SIZE{self.ReplayMemorySize}; '
                                 f'TARGET_UPDATE{self.TARGET_UPDATE}; '
                                 f'N_STEP{self.n_step}')

        if not os.path.exists(self.PATH):
            os.makedirs(self.PATH)

        self.model_dir = os.path.join(self.PATH, f'model.pkl')

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
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

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
            param.grad.base_data.clamp_(-1, 1)
        self.optimizer.act()

    def train(self, num_episodes=50):
        print('Training', self.model_kind, '...')
        for i_episode in tqdm(range(num_episodes)):
            # Initialize the environment and state
            self.training_environment.reset()
            state = torch.tensor([self.training_environment.get_current_state()], dtype=torch.float, device=device)
            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                done, reward, next_state = self.training_environment.act(action.item())

                reward = torch.tensor([reward], dtype=torch.float, device=device)

                if next_state is not None:
                    next_state = torch.tensor([next_state], dtype=torch.float, device=device)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                if not done:
                    state = torch.tensor([self.training_environment.get_current_state()], dtype=torch.float, device=device)

                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                if done:
                    break
            # Update the target network, copying all weights and biases in DQN
            if i_episode % self.TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        self.save_model(self.policy_net.state_dict())

        print('Complete')

    def save_model(self, model):
        torch.save(model, self.model_dir)

    def evaluate(self, initial_investment=1000, test_type='test') -> Evaluation:
        """
        :@param file_name: name of the .pkl file to load the model
        :@param test_type: test results on train data or test data
        :@return returns an Evaluation object to have access to different evaluation metrics.
        """
        data = self.training_environment if test_type == 'train' else self.test_environment

        self.test_net.load_state_dict(torch.load(self.model_dir))
        self.test_net.to(device)

        action_list = []
        data.__iter__()

        for batch in data:
            try:
                action_batch = self.test_net(batch).max(1)[1]
                action_list += list(action_batch.cpu().numpy())
            except ValueError:
                action_list += [1]



        data.make_investment(action_list)
        ev = Evaluation(data.base_data, data.action_name, initial_investment, self.transaction_cost)

        path_0 = os.path.join(self.model_dir, '..\data_input.txt')
        with open(path_0, 'w') as f:
            print(f"Writing file to: {path_0}")
            f.write(str(data.data_preprocessed))
        with open(os.path.join(self.model_dir, '..\evaluation_action_list.txt'), 'w') as f:
            f.write(str(action_list))
        with open(os.path.join(self.model_dir, '..\evaluation_actions.txt'), 'w') as f:
            f.write(str(data.base_data))

        print(test_type)
        ev.build_and_print_metrics()
        return ev
