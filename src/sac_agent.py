"""
This module implements the Soft Actor-Critic (SAC) algorithm, a state-of-the-art
reinforcement learning algorithm known for its stability and sample efficiency.

SAC is an off-policy algorithm that optimizes a stochastic policy in an
actor-critic framework. It aims to maximize a trade-off between expected return
and entropy, encouraging exploration and robustness.

Key components:
- Actor Network: Learns a stochastic policy (Gaussian distribution) to select actions.
- Critic Network (Q-networks): Learns the action-value function (Q-value) to evaluate actions.
- Target Networks: Stabilize training by providing consistent Q-value targets.
- Entropy Regularization: Encourages exploration by adding an entropy term to the reward.
- Automatic Entropy Tuning: Dynamically adjusts the entropy regularization weight (alpha).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import copy

# Determine the device to run computations on (GPU if available, otherwise CPU).
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants for clamping the log standard deviation of the Gaussian policy.
# This helps in numerical stability during training.
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class Actor(nn.Module):
    """
    The Actor network for the SAC agent. It outputs the parameters (mean and log_std)
    of a Gaussian distribution, from which actions are sampled.

    The network architecture consists of fully connected layers with ReLU activations.
    """
    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden_dims: list[int]):
        """
        Initializes the Actor network.

        Args:
            state_dim (int): The dimensionality of the input state space.
            action_dim (int): The dimensionality of the output action space.
            max_action (float): The maximum absolute value for actions, used to scale
                                the tanh-squashed actions to the environment's action space.
            hidden_dims (list[int]): A list specifying the number of neurons in each
                                     hidden layer (e.g., [256, 256]).
        """
        super().__init__()
        # First linear layer mapping state input to the first hidden layer.
        self.l1 = nn.Linear(state_dim, hidden_dims[0])
        # Second linear layer mapping from the first to the second hidden layer.
        self.l2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        # Linear layer to output the mean of the Gaussian distribution for actions.
        self.mean_l = nn.Linear(hidden_dims[1], action_dim)
        # Linear layer to output the log standard deviation of the Gaussian distribution.
        self.log_std_l = nn.Linear(hidden_dims[1], action_dim)
        self.max_action = max_action # Store max_action for scaling

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the Actor network.

        Args:
            x (torch.Tensor): The input state tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the mean and
                                               log standard deviation of the action distribution.
        """
        # Apply ReLU activation after each hidden layer.
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        # Compute the mean and log_std from the final hidden layer output.
        mean = self.mean_l(x)
        log_std = self.log_std_l(x)
        # Clamp log_std to ensure numerical stability and prevent extremely small or large std values.
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Samples an action from the current policy distribution and computes its log probability.

        This method implements the reparameterization trick for sampling from a Gaussian
        distribution and applies a tanh squashing function to ensure actions are within
        the environment's bounds.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the sampled action
                                               and its log probability.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp() # Convert log_std to standard deviation
        
        # Create a normal distribution object.
        normal = Normal(mean, std)
        # Sample from the distribution using the reparameterization trick.
        x_t = normal.rsample() 
        # Apply tanh squashing to the sampled action to bound it between -1 and 1.
        y_t = torch.tanh(x_t)
        # Scale the action to the environment's specific action range.
        action = y_t * self.max_action

        # Compute the log probability of the sampled action.
        # This involves adjusting for the tanh squashing function.
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound: Adjust log_prob due to the tanh transformation.
        # The term `1e-6` is added for numerical stability to prevent log(0).
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        # Sum log probabilities across action dimensions to get a single log_prob per sample.
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    """
    The Distributional Critic network for the DSAC agent. It outputs a discrete
    distribution over Q-values instead of a single value.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int], n_atoms: int, v_min: float, v_max: float):
        """
        Initializes the Distributional Critic network.

        Args:
            state_dim (int): Dimensionality of the state space.
            action_dim (int): Dimensionality of the action space.
            hidden_dims (list[int]): Neurons in each hidden layer.
            n_atoms (int): Number of atoms for the discrete distribution.
            v_min (float): Minimum value of the Q-value support.
            v_max (float): Maximum value of the Q-value support.
        """
        super().__init__()
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        # Define the support of the distribution (the discrete Q-values).
        self.support = torch.linspace(v_min, v_max, n_atoms).to(device)

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.l2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.l3 = nn.Linear(hidden_dims[1], n_atoms)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.l5 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.l6 = nn.Linear(hidden_dims[1], n_atoms)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through both Q1 and Q2 networks.
        Returns the logits of the Q-value distributions.
        """
        sa = torch.cat([state, action], 1)
        
        x = F.relu(self.l1(sa))
        x = F.relu(self.l2(x))
        q1_logits = self.l3(x)

        x = F.relu(self.l4(sa))
        x = F.relu(self.l5(x))
        q2_logits = self.l6(x)
        
        return q1_logits, q2_logits

    def get_probs(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the probabilities of the Q-value distributions (after softmax).
        """
        q1_logits, q2_logits = self.forward(state, action)
        q1_probs = F.softmax(q1_logits, dim=-1)
        q2_probs = F.softmax(q2_logits, dim=-1)
        return q1_probs, q2_probs



class SAC:
    """
    Implementation of the Distributional Soft Actor-Critic (DSAC) algorithm.
    """
    def __init__(self, config: dict):
        """
        Initializes the DSAC agent.
        """
        self.gamma = config['rl_hyperparameters']['gamma']
        self.tau = config['rl_hyperparameters']['tau']
        self.alpha = config['rl_hyperparameters']['alpha']
        self.max_action = config['rl_hyperparameters']['max_action']
        self.auto_entropy_tuning = True

        state_dim = config['rl_hyperparameters']['state_dim']
        action_dim = config['rl_hyperparameters']['action_dim']
        actor_hidden_dims = config['rl_hyperparameters']['network_architecture']['actor']
        critic_hidden_dims = config['rl_hyperparameters']['network_architecture']['critic']
        
        # DSAC specific hyperparameters
        self.n_atoms = config['rl_hyperparameters']['num_atoms']
        self.v_min = config['rl_hyperparameters']['v_min']
        self.v_max = config['rl_hyperparameters']['v_max']

        self.actor = Actor(state_dim, action_dim, self.max_action, actor_hidden_dims).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['rl_hyperparameters']['learning_rate'])

        self.critic = Critic(state_dim, action_dim, critic_hidden_dims, self.n_atoms, self.v_min, self.v_max).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['rl_hyperparameters']['learning_rate'])

        if self.auto_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config['rl_hyperparameters']['learning_rate'])

        self.total_it = 0

    def select_action(self, state: np.ndarray) -> np.ndarray:
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size: int):
        self.total_it += 1

        state, next_state, action, reward, done = replay_buffer.sample(batch_size)
        
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)

        # --- Update Critic Networks ---
        with torch.no_grad():
            next_action, next_log_pi = self.actor.sample(next_state)
            
            # Get target Q-value distributions and select the one with the minimum expected value
            next_q1_probs, next_q2_probs = self.critic_target.get_probs(next_state, next_action)
            next_q1_expected = (next_q1_probs * self.critic.support).sum(dim=1, keepdim=True)
            next_q2_expected = (next_q2_probs * self.critic.support).sum(dim=1, keepdim=True)
            min_next_q_probs = torch.where(next_q1_expected < next_q2_expected, next_q1_probs, next_q2_probs)

            # The target distribution is based on the Bellman equation for distributional RL.
            # We project the discounted, entropy-regularized next-state distribution onto the current support.
            
            # The entropy term is subtracted from the support of the next state distribution
            entropy_term = self.alpha * next_log_pi
            target_support = self.critic.support.unsqueeze(0) - entropy_term.unsqueeze(2)
            
            # Apply Bellman update (reward + gamma * discounted_next_support)
            target_support = reward.unsqueeze(2) + (1 - done).unsqueeze(2) * self.gamma * target_support
            
            # Clamp the target support to be within [v_min, v_max]
            target_support = torch.clamp(target_support, self.v_min, self.v_max)

            # Distribute probabilities via linear interpolation (C51 projection)
            delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)
            b = (target_support - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            # Handle edge cases where l and u are the same
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.n_atoms - 1)) * (l == u)] += 1
            
            target_dist = torch.zeros_like(min_next_q_probs)
            offset = torch.linspace(0, (batch_size - 1) * self.n_atoms, batch_size).long().unsqueeze(1).to(device)
            
            # Project probabilities
            target_dist.view(-1).index_add_(0, (l + offset).view(-1), (min_next_q_probs * (u.float() - b)).view(-1))
            target_dist.view(-1).index_add_(0, (u + offset).view(-1), (min_next_q_probs * (b - l.float())).view(-1))

        # Get current Q-value distributions
        current_q1_logits, current_q2_logits = self.critic(state, action)
        
        # Critic loss is the cross-entropy between the current and target distributions
        critic_loss = - (target_dist * F.log_softmax(current_q1_logits, dim=1)).sum(dim=1).mean() - \
                      (target_dist * F.log_softmax(current_q2_logits, dim=1)).sum(dim=1).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Update Actor Network ---
        pi, log_pi = self.actor.sample(state)
        q1_probs, q2_probs = self.critic.get_probs(state, pi)
        
        q1_expected = (q1_probs * self.critic.support).sum(dim=1, keepdim=True)
        q2_expected = (q2_probs * self.critic.support).sum(dim=1, keepdim=True)
        min_q_expected = torch.min(q1_expected, q2_expected)

        actor_loss = ((self.alpha * log_pi) - min_q_expected).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Update Alpha ---
        if self.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

        # --- Soft Update Target Networks ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename: str):
        torch.save(self.actor.state_dict(), str(filename) + "_actor.pth")
        torch.save(self.critic.state_dict(), str(filename) + "_critic.pth")
        if self.auto_entropy_tuning:
            torch.save(self.log_alpha, str(filename) + "_log_alpha.pth")

    def load(self, filename: str):
        self.actor.load_state_dict(torch.load(str(filename) + "_actor.pth"))
        self.critic.load_state_dict(torch.load(str(filename) + "_critic.pth"))
        self.critic_target = copy.deepcopy(self.critic)
        if self.auto_entropy_tuning:
            self.log_alpha = torch.load(str(filename) + "_log_alpha.pth")
            self.alpha = self.log_alpha.exp() # Recalculate alpha from loaded log_alpha
