import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DialogueEnv:
    """Multi-turn dialogue environment that simulates conversations."""

    def __init__(self):
        self.turns = 5  # Each dialogue lasts 5 turns
        self.current_turn = 0
        self.conversation = []

    def reset(self):
        """Resets the environment for a new dialogue."""
        self.current_turn = 0
        self.conversation = []
        return "Hi, how can I help you today?"  # Starting dialogue

    def step(self, action):
        """Takes an action (a response) and advances the conversation."""
        self.conversation.append(action)
        self.current_turn += 1

        if self.current_turn < self.turns:
            # Generate the next response from the environment (placeholder)
            next_state = f"Response {self.current_turn}: How about this?"
            done = False
            reward = self._human_feedback(action)
        else:
            next_state = "Conversation ended."
            done = True
            reward = self._human_feedback(action)

        return next_state, reward, done

    def _human_feedback(self, action):
        """Simulates human feedback by returning a random reward."""
        return np.random.choice([1, -1])  # 1 for positive feedback, -1 for negative


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


def pad_or_truncate(state, size=100):
    """Pads or truncates the input state to match the required input size."""
    state_tensor = torch.tensor([ord(c) for c in state], dtype=torch.float32)
    if state_tensor.size(0) < size:
        padded_tensor = torch.cat([state_tensor, torch.zeros(size - state_tensor.size(0))])
    else:
        padded_tensor = state_tensor[:size]
    return padded_tensor.unsqueeze(0)  # Add batch dimension


def train_rlhf(env, model, optimizer, num_episodes=1000):
    """Trains the policy network using reinforcement learning with human feedback."""
    gamma = 0.99  # Discount factor for future rewards

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        log_probs = []
        rewards = []

        done = False
        while not done:
            # Pad or truncate the input state to the required size
            state_tensor = pad_or_truncate(state, size=100)
            logits = model(state_tensor)
            action_probs = torch.softmax(logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)

            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            log_probs.append(log_prob)

            # Take the action in the environment
            action_text = f"Action {action.item()}"
            next_state, reward, done = env.step(action_text)
            rewards.append(reward)
            total_reward += reward

            state = next_state

        # Calculate the discounted rewards
        discounted_rewards = []
        cumulative_reward = 0
        for r in reversed(rewards):
            cumulative_reward = r + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        # Normalize the rewards
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-6)

        # Policy Gradient: Update the policy
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)

        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")


if __name__ == "__main__":
    # Instantiate environment and model
    env = DialogueEnv()
    input_size = 100  # Placeholder for state size (e.g., fixed-length input of size 100)
    hidden_size = 128
    output_size = 10  # Placeholder for the number of possible actions (dialogue responses)

    model = PolicyNetwork(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the policy using RL with Human Feedback (simulated)
    train_rlhf(env, model, optimizer, num_episodes=1000)
