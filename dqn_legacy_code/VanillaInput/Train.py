import torch
import torch.optim as optim
from Reimplementation.BaseTrain import BaseTrain
from Reimplementation.VanillaInput.DeepQNetwork import NeuralNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Train(BaseTrain):
    def __init__(self,
                 data_train,
                 data_test,
                 dataset_name,
                 state_mode=1,
                 window_size=1,
                 transaction_cost=0.0,
                 BATCH_SIZE=30,
                 GAMMA=0.7,
                 ReplayMemorySize=50,
                 TARGET_UPDATE=5,
                 n_step=10):
        """
        This class is inherited from the BaseTrain class to initialize networks and other stuff that are specific to this
        model. For those parameters in the following explanation that I wrote: "for using in the name of the result file"
        the effect of those parameters has been applied in the Data class and are mentioned here only for begin used as
        part of the experiment's result filename.
        @param data_train: of type DataAutoPatternExtractionAgent
        @param data_test: of type DataAutoPatternExtractionAgent
        @param n_classes: this is the feature vector size of the encoder's output.
        @param BATCH_SIZE: batch size for batch training
        @param GAMMA: in the algorithm
        @param ReplayMemorySize: size of the replay buffer
        @param TARGET_UPDATE: hard update policy network into target network every TARGET_UPDATE iterations
        @param n_step: for using in the name of the result file
        """

        super(Train, self).__init__(
                                    data_train,
                                    data_test,
                                    dataset_name,
                                    'DeepRL',
                                    state_mode,
                                    window_size,
                                    transaction_cost,
                                    BATCH_SIZE,
                                    GAMMA,
                                    ReplayMemorySize,
                                    TARGET_UPDATE,
                                    n_step)

        self.policy_net: NeuralNetwork = NeuralNetwork(data_train.state_size, 3).to(device)
        self.target_net: NeuralNetwork = NeuralNetwork(data_train.state_size, 3).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters())

        self.test_net = NeuralNetwork(self.training_environment.state_size, 3)
        self.test_net.to(device)
