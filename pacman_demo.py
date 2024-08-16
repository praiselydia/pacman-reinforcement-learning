from PIL import Image
import IPython.display as display
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt
import time

# Define DQN model
class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1_input_dim = self.feature_size(input_shape)
        self.fc1 = nn.Linear(self.fc1_input_dim, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def feature_size(self, input_shape):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *input_shape)))).view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
#Define DQN Agent
class DQNAgent:
    def __init__(self, input_shape, num_actions):
        self.model = DQN(input_shape, num_actions)
        self.target_model = DQN(input_shape, num_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.update_target_frequency = 1000
        self.steps = 0
        self.num_actions = num_actions

    def act(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        current_q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.update_target_frequency == 0:
            self.target_model.load_state_dict(self.model.state_dict())

# Training function for DQN
def train_dqn(env_id, num_episodes=5):
    env = gym.make(env_id)
    input_shape = (3, 210, 160)
    num_actions = env.action_space.n

    dqn_agent = DQNAgent(input_shape, num_actions)

    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    epsilon = epsilon_start

    dqn_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        state = np.transpose(state, (2, 0, 1))  # Convert to channel-first format
        done = False
        total_reward = 0

        while not done:
            action = dqn_agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            next_state = np.transpose(next_state, (2, 0, 1))  # Convert to channel-first format
            dqn_agent.remember(state, action, reward, next_state, done)
            dqn_agent.train()
            state = next_state
            total_reward += reward

        epsilon = max(epsilon_end, epsilon_decay * epsilon)
        dqn_rewards.append(total_reward)
        print(f"Episode: {episode + 1}, Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    return dqn_agent, dqn_rewards

# Define Policy Network for PPO
class PolicyNetwork(nn.Module):
    def __init__(self, action_space):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1_input_dim = self.feature_size((3, 210, 160))  # Calculate input size to fc1
        self.fc1 = nn.Linear(self.fc1_input_dim, 512)
        self.mean = nn.Linear(512, action_space)
        self.std = nn.Linear(512, action_space)

    def feature_size(self, input_shape):
        x = torch.zeros(1, *input_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        mean = self.mean(x)
        std = torch.clamp(self.std(x), min=-20, max=2)
        return mean, std.exp()
    
# Preprocess state function for PPO
def preprocess_state(state):
    state = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1)  # Transpose to [channels, height, width]
    return state.unsqueeze(0)  # Add batch dimension

# Training function for PPO
def train_ppo(env_name, max_episodes, max_steps_per_episode, lr, gamma, clip_ratio, epochs):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    policy_net = PolicyNetwork(action_dim)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    ppo_rewards = []

    for episode in range(1, max_episodes + 1):
        state = preprocess_state(env.reset())
        episode_rewards = 0

        for step in range(max_steps_per_episode):
            with torch.no_grad():
                mean, std = policy_net(state)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()

            next_state, reward, done, _ = env.step(action.argmax().item())
            next_state = preprocess_state(next_state)
            episode_rewards += reward

            state = next_state

        ppo_rewards.append(episode_rewards)
        print(f"Episode {episode}:  Reward = {episode_rewards}")

    return policy_net, ppo_rewards

# Plotting function for rewards
def plot_rewards(dqn_rewards, ppo_rewards):
    # Moving average to smooth the reward curve
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    dqn_smoothed_rewards = moving_average(dqn_rewards, window_size=3)
    ppo_smoothed_rewards = moving_average(ppo_rewards, window_size=3)

    # Plot DQN and PPO rewards
    plt.figure(figsize=(12, 6))
    plt.plot(dqn_rewards, label='DQN (Original)', color='skyblue', alpha=0.5, linestyle='--')
    plt.plot(dqn_smoothed_rewards, label='DQN (Smoothed)', color='blue', linewidth=2)
    plt.plot(ppo_rewards, label='PPO (Original)', color='lightgreen', alpha=0.5, linestyle='--')
    plt.plot(ppo_smoothed_rewards, label='PPO (Smoothed)', color='green', linewidth=2)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Total Reward', fontsize=14)
    plt.title('DQN and PPO Training Rewards', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Define DQN model and other classes here...

def display_game_dqn(env_id, agent):
    env = gym.make(env_id, render_mode='rgb_array')
    state = env.reset()
    state = np.transpose(state, (2, 0, 1))  # Convert to channel-first format
    done = False
    frames = []

    while not done:
        action = agent.act(state, epsilon=0.0)  # Use the trained agent for actions
        next_state, reward, done, _ = env.step(action)
        next_state = np.transpose(next_state, (2, 0, 1))  # Convert to channel-first format
        state = next_state
        frames.append(env.render(mode='rgb_array'))
        time.sleep(0.05)

    env.close()

    # Display frames as an animation
    for frame in frames:
        img = Image.fromarray(frame)
        display.display(img)
        display.clear_output(wait=True)

def display_game_ppo(env_id, agent):
    env = gym.make(env_id, render_mode='rgb_array')
    state = preprocess_state(env.reset())
    done = False
    frames = []

    while not done:
        with torch.no_grad():
            mean, std = agent(state)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()

        next_state, reward, done, _ = env.step(action.argmax().item())
        next_state = preprocess_state(next_state)
        state = next_state
        frames.append(env.render(mode='rgb_array'))
        time.sleep(0.05)

    env.close()

    # Display frames as an animation
    for frame in frames:
        img = Image.fromarray(frame)
        display.display(img)
        display.clear_output(wait=True)

if __name__ == "__main__":
    env_id = 'MsPacman-v4'
    num_episodes = 500

    print("Training DQN Agent...")
    dqn_agent, dqn_rewards = train_dqn(env_id, num_episodes=num_episodes)
    print("DQN Training completed.")
    print("Training PPO Agent...")
    max_episodes = 5
    max_steps_per_episode = 200
    lr = 1e-3
    gamma = 0.99
    clip_ratio = 0.2
    epochs = 10
    ppo_agent, ppo_rewards = train_ppo(env_id, max_episodes, max_steps_per_episode, lr, gamma, clip_ratio, epochs)
    print("PPO Training completed.")
    print("Plotting rewards...")
    plot_rewards(dqn_rewards, ppo_rewards)
    print("Rendering game using trained DQN agent...")
    display_game_dqn(env_id, dqn_agent)
    print("Rendering game using trained PPO agent...")
    display_game_ppo(env_id, ppo_agent)