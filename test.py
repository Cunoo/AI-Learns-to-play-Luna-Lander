import time
import torch
import torch.nn as nn
import gymnasium as gym
import torch.nn.functional as F



class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

# Initialize the environment with render_mode
env = gym.make("LunarLander-v2", render_mode="human")

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN(n_observations, n_actions).to(device)
policy_net.load_state_dict(torch.load('policy_net.pth', map_location=device))
policy_net.eval()  # Set the model to evaluation mode

def select_action(state):
    with torch.no_grad():
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return policy_net(state).max(1).indices.item()

# Run the environment with the trained model
for i_episode in range(5):  # Play 5 episodes
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Render the environment with "human" mode
        env.render()

        # Add a short sleep to ensure the rendering updates properly
        time.sleep(0.01)

        action = select_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Update done flag based on termination conditions
        done = terminated or truncated

    print(f"Episode {i_episode + 1}: Total Reward: {total_reward}")

env.close()
