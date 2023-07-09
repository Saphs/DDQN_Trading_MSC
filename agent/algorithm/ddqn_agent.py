from agent.algorithm.dqn_agent import DqnAgent
import torch
import torch.nn.functional as functional
from agent.algorithm.model.replay_memory import Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDqnAgent(DqnAgent):

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

        # Next steps s' might contain ending states
        s_prime = torch.cat([s for s in batch.next_state if s is not None])
        s = torch.cat(batch.state)
        a = torch.cat(batch.action)
        r = torch.cat(batch.reward)

        with torch.no_grad():
            # aka.: Q(s', a'; θ_t)
            max_q_prime = torch.zeros(self.batch_size, device=device)
            argmax_a = self.policy_net(s_prime).argmax(dim=1).unsqueeze(-1).detach()  # ToDo: view actual tensor
            print(f"\n{argmax_a=} {argmax_a.size()}\n")
            max_q_prime = self.target_net(s_prime).detach()
            max_q_prime = max_q_prime.gather(dim=0, index=argmax_a)
            print(f"{max_q_prime=}  {max_q_prime.size()}\n")

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