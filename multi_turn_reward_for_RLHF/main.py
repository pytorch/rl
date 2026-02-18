import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchrl.envs import EnvBase
from torchrl.envs.libs.gym import GymWrapper
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

# Define the Dialogue Environment in TorchRL Format
class DialogueEnvTorchRL(EnvBase):
    """TorchRL-compatible multi-turn dialogue environment that simulates conversations."""

    def __init__(self):
        super().__init__()
        self.turns = 5  # Each dialogue lasts 5 turns
        self.current_turn = 0
        self.conversation = []
        self.action_spec = torch.arange(10)  # 10 discrete actions (dialogue responses)
        self.observation_spec = torch.zeros(100)  # Fixed-size state representation

    def _reset(self):
        """Resets the environment for a new dialogue."""
        self.current_turn = 0
        self.conversation = []
        return {"observation": self._encode_state("Hi, how can I help you today?")}

    def _step(self, action):
        """Takes an action (a response) and advances the conversation."""
        action_text = f"Action {action.item()}"
        self.conversation.append(action_text)
        self.current_turn += 1

        if self.current_turn < self.turns:
            next_state = f"Response {self.current_turn}: How about this?"
            done = False
        else:
            next_state = "Conversation ended."
            done = True

        reward = self._human_feedback(action_text)
        return {"observation": self._encode_state(next_state), "reward": reward, "done": done}

    def _human_feedback(self, action):
        """Simulates human feedback by returning a random reward."""
        return np.random.choice([1, -1])  # 1 for positive feedback, -1 for negative

    def _encode_state(self, state, size=100):
        """Encodes state into a tensor format (pads or truncates)."""
        state_tensor = torch.tensor([ord(c) for c in state], dtype=torch.float32)
        if state_tensor.size(0) < size:
            padded_tensor = torch.cat([state_tensor, torch.zeros(size - state_tensor.size(0))])
        else:
            padded_tensor = state_tensor[:size]
        return padded_tensor.unsqueeze(0)  # Add batch dimension

# Define Policy Network
class PolicyNetwork(nn.Module):
    """Policy network that defines the agent's behavior."""

    def __init__(self, input_size=100, hidden_size=128, output_size=10):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass through the network."""
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Define Value Network for PPO
class ValueNetwork(nn.Module):
    """Value network for estimating the state value."""

    def __init__(self, input_size=100, hidden_size=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """Forward pass through the network."""
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# Training Setup using PPO in TorchRL
def train_rlhf_torchrl(num_episodes=1000, batch_size=32):
    """Trains the policy network using PPO with reinforcement learning with human feedback."""
    
    # Instantiate environment
    env = DialogueEnvTorchRL()

    # Create policy and value networks
    policy_model = PolicyNetwork(input_size=100, hidden_size=128, output_size=10)
    value_model = ValueNetwork(input_size=100, hidden_size=128)

    # Create policy distribution
    policy = ProbabilisticActor(
        module=policy_model,
        in_keys=["observation"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical
    )

    # Value function
    value_operator = ValueOperator(
        module=value_model,
        in_keys=["observation"]
    )

    # Optimizers
    policy_optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    value_optimizer = optim.Adam(value_operator.parameters(), lr=1e-3)

    # Setup collector
    collector = SyncDataCollector(
        env, policy, frames_per_batch=batch_size, total_frames=num_episodes * batch_size
    )

    # Replay buffer
    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=10000)
    )

    # Loss function (PPO)
    advantage_module = GAE(value_operator=value_operator, gamma=0.99, lmbda=0.95)
    loss_module = ClipPPOLoss(
        actor=policy,
        critic=value_operator,
        advantage_module=advantage_module,
        clip_epsilon=0.2
    )

    for episode in range(num_episodes):
        for batch in collector:
            buffer.extend(batch)

            # Sample from buffer
            sampled_batch = buffer.sample(batch_size)

            # Compute loss and update policy
            loss = loss_module(sampled_batch)
            policy_optimizer.zero_grad()
            loss["loss_objective"].backward()
            policy_optimizer.step()

            # Update value function
            value_optimizer.zero_grad()
            loss["loss_critic"].backward()
            value_optimizer.step()

        print(f"Episode {episode + 1}/{num_episodes}, Loss: {loss['loss_objective'].item()}")

if __name__ == "__main__":
    train_rlhf_torchrl(num_episodes=1000, batch_size=32)
