import gym
import torch
from torch import nn

from . import MuZeroAgent, MuZeroConfig

from utils.env import repeat_action

RepresentationNet = lambda: nn.Sequential()     # Identity mapping

class PredictionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(4, 32), nn.ReLU(inplace=True),
            nn.Linear(32, 16), nn.ReLU(inplace=True)
        )
        self.prior = nn.Sequential(
            nn.Linear(16, 2), nn.Softmax(dim=1)
        )
        self.value = nn.Sequential(
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.pre(x)
        return self.prior(x), self.value(x)

class DynamicsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(5, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 16), nn.ReLU(inplace=True)
        )
        self.state = nn.Linear(16, 4)
        self.reward = nn.Linear(16, 1)

    def forward(self, state, action):
        action = torch.as_tensor(action, dtype=torch.float).view(state.shape[0], -1)
        x = torch.cat((state, action), dim=1)
        x = self.pre(x)
        return self.state(x), self.reward(x)

if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    config = MuZeroConfig(
        representation_net_gen=RepresentationNet,
        prediction_net_gen=PredictionNet,
        dynamics_net_gen=DynamicsNet,
        c1=1.25,
        c2=19528,
        simulations=10,
        discount=0.99,
        policy_temperature=1.0,
        replay_capacity=1000,
        state_shape=(4, ),
        action_dim=2
    )

    agent = MuZeroAgent(config)

    for _ in range(100):
        
        states = [torch.as_tensor(env.reset(), dtype=torch.float)]
        actions = []
        rewards = []
        
        done = False
        tot_reward = 0
        while not done:

            action = int(agent.get_actions(states[-1].unsqueeze(0)))
            next_state, reward, done, _ = repeat_action(env, action, 2)
            next_state = torch.as_tensor(next_state, dtype=torch.float)
            
            states.append(next_state); actions.append(action); rewards.append(reward)
            
            tot_reward += reward
            env.render()
        
        agent.observe(torch.stack(states), torch.tensor(actions, dtype=torch.long), torch.tensor(rewards))
        if agent.replay.get_size() > 10:
            agent.train_step(5, 5)

        print(tot_reward)