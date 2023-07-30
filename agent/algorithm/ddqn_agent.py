from torch import Tensor

from agent.algorithm.config_parsing.agent_parameters import AgentParameters
from agent.algorithm.config_parsing.dqn_config import DqnConfig
from agent.algorithm.dqn_agent import DqnAgent
import torch
import torch.nn.functional as functional
from agent.algorithm.model.replay_memory import Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDqnAgent(DqnAgent):

    @classmethod
    def init(cls, state_size: int, config: DqnConfig, name: str = None):
        c: AgentParameters = config.agent
        return DDqnAgent(
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

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Next steps s' might contain ending states
        if batch.next_state[0] is None:
            return



        s_prime = torch.cat(batch.next_state)
        s = torch.cat(batch.state)
        a = torch.cat(batch.action)
        r = torch.cat(batch.reward)

        with torch.no_grad():
            # aka.: Q(s', a'; θ_t)
            #print(f"{s_prime=}")
            argmax_a: Tensor = self.policy_net(s_prime).argmax(dim=1).unsqueeze(1)
            #print(f"{argmax_a=}")
            max_q_prime: Tensor = self.target_net(s_prime).gather(dim=0, index=argmax_a)
            #print(f"{max_q_prime=}")

            # Compute the expected Q values, aka.: y_i
            target_q = r + self.gamma * max_q_prime
            target_q = target_q.unsqueeze(1)

        current_q = self.policy_net(s).gather(1, a)

        # Compute Huber loss
        loss = functional.smooth_l1_loss(current_q, target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss
