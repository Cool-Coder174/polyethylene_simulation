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
    The Critic network for the SAC agent. It consists of two Q-networks (Q1 and Q2)
    to mitigate overestimation bias in Q-value estimation.

    Each Q-network takes a state-action pair as input and outputs a single Q-value.
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list[int]):
        """
        Initializes the Critic network.

        Args:
            state_dim (int): The dimensionality of the input state space.
            action_dim (int): The dimensionality of the input action space.
            hidden_dims (list[int]): A list specifying the number of neurons in each
                                     hidden layer for both Q-networks.
        """
        super().__init__()
        # Q1 architecture: Takes concatenated state and action as input.
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.l2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.l3 = nn.Linear(hidden_dims[1], 1) # Output a single Q-value

        # Q2 architecture: Identical to Q1, but with separate weights.
        self.l4 = nn.Linear(state_dim + action_dim, hidden_dims[0])
        self.l5 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.l6 = nn.Linear(hidden_dims[1], 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through both Q1 and Q2 networks.

        Args:
            state (torch.Tensor): The input state tensor.
            action (torch.Tensor): The input action tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the Q-values from Q1 and Q2.
        """
        # Concatenate state and action tensors to form the input for the Q-networks.
        sa = torch.cat([state, action], 1)

        # Forward pass for Q1 network.
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        # Forward pass for Q2 network.
        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass only through the Q1 network.

        Args:
            state (torch.Tensor): The input state tensor.
            action (torch.Tensor): The input action tensor.

        Returns:
            torch.Tensor: The Q-value from the Q1 network.
        """
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class SAC:
    """
    Implementation of the Soft Actor-Critic (SAC) algorithm.

    This class manages the actor and critic networks, their optimizers,
    target networks, and the training process for the SAC agent.
    """
    def __init__(self, config: dict):
        """
        Initializes the SAC agent.

        Args:
            config (dict): A dictionary containing hyperparameters for the SAC algorithm,
                           including gamma, tau, learning rates, network architectures, etc.
        """
        # Extract hyperparameters from the configuration dictionary.
        self.gamma = config['rl_hyperparameters']['gamma'] # Discount factor for future rewards
        self.tau = config['rl_hyperparameters']['tau']     # Soft update coefficient for target networks
        self.alpha = config['rl_hyperparameters']['alpha'] # Entropy regularization coefficient
        self.max_action = config['rl_hyperparameters']['max_action'] # Max action value for scaling
        self.auto_entropy_tuning = True # Flag to enable/disable automatic entropy tuning

        # Get state and action dimensions from config.
        state_dim = config['rl_hyperparameters']['state_dim']
        action_dim = config['rl_hyperparameters']['action_dim']
        # Get network hidden layer dimensions.
        actor_hidden_dims = config['rl_hyperparameters']['network_architecture']['actor']
        critic_hidden_dims = config['rl_hyperparameters']['network_architecture']['critic']

        # Initialize Actor network and its optimizer.
        self.actor = Actor(state_dim, action_dim, self.max_action, actor_hidden_dims).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config['rl_hyperparameters']['learning_rate'])

        # Initialize Critic network and its target network.
        self.critic = Critic(state_dim, action_dim, critic_hidden_dims).to(device)
        self.critic_target = copy.deepcopy(self.critic) # Target critic is a copy of the main critic
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config['rl_hyperparameters']['learning_rate'])

        # Initialize components for automatic entropy tuning if enabled.
        if self.auto_entropy_tuning:
            # Target entropy is typically set to -action_dim.
            self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(device)).item()
            # log_alpha is optimized instead of alpha directly for stability.
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config['rl_hyperparameters']['learning_rate'])

        self.total_it = 0 # Counter for total training iterations

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Selects an action based on the current policy for a given state.

        This method is used during environment interaction (e.g., in `train.py`)
        to get an action from the actor network.

        Args:
            state (np.ndarray): The current state observation from the environment.

        Returns:
            np.ndarray: The selected action, scaled to the environment's action space.
        """
        # Convert the numpy state array to a PyTorch tensor and move to the correct device.
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # Sample an action from the actor's distribution (log_pi is not needed here).
        action, _ = self.actor.sample(state)
        # Convert the action tensor back to a numpy array and flatten it.
        return action.cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size: int):
        """
        Performs one training step for the SAC agent.

        This involves:
        1. Sampling a batch of experiences from the replay buffer.
        2. Updating the Critic networks (Q1 and Q2).
        3. Updating the Actor network.
        4. Updating the entropy regularization coefficient (alpha) if auto-tuning is enabled.
        5. Soft-updating the target Critic networks.

        Args:
            replay_buffer: An instance of the ReplayBuffer containing past experiences.
            batch_size (int): The number of experiences to sample for this training step.
        """
        self.total_it += 1 # Increment total training iterations counter

        # Sample a batch of transitions from the replay buffer.
        state, next_state, action, reward, done = replay_buffer.sample(batch_size)
        
        # Convert numpy arrays to PyTorch tensors and move them to the appropriate device.
        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        # Reshape reward and done tensors to have a dimension of 1 for broadcasting.
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(done).unsqueeze(1).to(device)

        # --- Update Critic Networks (Q-functions) ---
        with torch.no_grad(): # Operations within this block do not track gradients
            # Sample next actions and their log probabilities from the current actor policy.
            next_action, next_log_pi = self.actor.sample(next_state)
            # Get Q-values from the target critic networks for the next state-action pair.
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            # Take the minimum of the two target Q-values to mitigate overestimation bias.
            # Subtract alpha * next_log_pi for entropy regularization.
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_pi
            # Compute the target Q-value using the Bellman equation.
            # (1 - done) handles terminal states (done=True means no future reward).
            target_Q = reward + (1 - done) * self.gamma * target_Q

        # Get current Q-values from the main critic networks for the current state-action pair.
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute the critic loss using Mean Squared Error (MSE) between current and target Q-values.
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # Zero gradients, perform backpropagation, and update critic network weights.
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Update Actor Network (Policy) ---
        # Sample actions and their log probabilities from the current actor policy.
        pi, log_pi = self.actor.sample(state)
        # Get Q-values from the main critic networks for the sampled actions.
        qf1_pi, qf2_pi = self.critic(state, pi)
        # Take the minimum of the two Q-values.
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Compute the actor loss. This aims to maximize Q-values while also maximizing entropy.
        # The negative sign is because optimizers minimize loss, but we want to maximize reward/entropy.
        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        # Zero gradients, perform backpropagation, and update actor network weights.
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Update Entropy Regularization Coefficient (Alpha) ---
        if self.auto_entropy_tuning:
            # Compute the loss for alpha. This aims to drive the entropy towards a target entropy.
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            # Zero gradients, perform backpropagation, and update alpha.
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            # Update the actual alpha value from its log version.
            self.alpha = self.log_alpha.exp()

        # --- Soft Update Target Critic Networks ---
        # Update the target critic network weights using a soft update (polyak averaging).
        # This slowly moves the target network towards the main network, stabilizing training.
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename: str):
        """
        Saves the state dictionaries of the actor, critic, and log_alpha (if auto-tuning).

        Args:
            filename (str): The base filename to use for saving the models.
                            (e.g., "model" will save "model_actor.pth", "model_critic.pth").
        """
        torch.save(self.actor.state_dict(), str(filename) + "_actor.pth")
        torch.save(self.critic.state_dict(), str(filename) + "_critic.pth")
        if self.auto_entropy_tuning:
            torch.save(self.log_alpha, str(filename) + "_log_alpha.pth")

    def load(self, filename: str):
        """
        Loads the state dictionaries for the actor, critic, and log_alpha (if auto-tuning).

        Args:
            filename (str): The base filename from which to load the models.
        """
        self.actor.load_state_dict(torch.load(str(filename) + "_actor.pth"))
        self.critic.load_state_dict(torch.load(str(filename) + "_critic.pth"))
        # After loading the critic, ensure the target critic is an exact copy.
        self.critic_target = copy.deepcopy(self.critic)
        if self.auto_entropy_tuning:
            self.log_alpha = torch.load(str(filename) + "_log_alpha.pth")
            self.alpha = self.log_alpha.exp() # Recalculate alpha from loaded log_alpha
