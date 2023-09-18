from torch import Tensor, nn

from agent.algorithm.config_parsing.agent_parameters import AgentParameters
from agent.algorithm.config_parsing.dqn_config import DqnConfig
from agent.algorithm.dqn_agent import DqnAgent
import torch
import torch.nn as functional
from agent.algorithm.model.replay_memory import Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _are_equal(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True

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
            name=name,
            check_points=config.check_points
        )

    def optimize_model(self):
        batch = self.memory.sample()

        # Next steps s' might contain ending states
        if batch.next_state[0] is None:
            return

        s_prime = torch.cat(batch.next_state)
        s = torch.cat(batch.state)
        a = torch.cat(batch.action)
        r = torch.cat(batch.reward)

        # Calc Q_pol(s, a) values (The value of taking action a in state s)
        current_q = self.policy_net(s).gather(1, a)

        with torch.no_grad():
            # Calc argmax_a( Q_pol(s', a) ) values (The action giving the largest Q-value in state s')
            argmax_a: Tensor = self.policy_net(s_prime).argmax(dim=1).unsqueeze(1)
            # Calc Q_tar(s', argmax_a( Q_pol(s', a) )) values (The value of taking the action selected by Q_pol)
            max_q_prime: Tensor = self.target_net(s_prime).gather(1, argmax_a)
            target_q = r + self.gamma * max_q_prime


        # Optimize the model
        loss = self.huber_loss(input=current_q, target=target_q)
        self.td_error_sum = self.td_error_sum + (target_q - current_q).sum()
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        return loss
