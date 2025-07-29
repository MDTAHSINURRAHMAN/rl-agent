import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DuelingDQN(nn.Module):
    """Dueling DQN Architecture"""
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        
        # Feature layer with layer normalization
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # Value stream with layer normalization
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream with layer normalization
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
    def forward(self, state):
        features = self.feature_layer(state)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantage streams
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return qvals

class ReplayBuffer:
    """Experience Replay Buffer"""
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, *args):
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DDQNAgent:
    """Double DQN Agent with Dueling Architecture"""
    def __init__(self, state_size, action_size, 
                 batch_size=64, 
                 learning_rate=0.0005,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 target_update=10,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Networks
        self.policy_net = DuelingDQN(state_size, action_size).to(device)
        self.target_net = DuelingDQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.target_update = target_update
        self.batch_size = batch_size
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(100000)
        
        self.train_step = 0
    
    def state_dict_to_tensor(self, state_dict):
        """Convert state dictionary to tensor"""
        state_array = np.concatenate([
            state_dict['Age'],
            [float(state_dict['Previous_Complications'])],
            [float(state_dict['Preexisting_Diabetes'])],
            [float(state_dict['Visit'])],
            state_dict['Systolic_BP'],
            state_dict['Diastolic_BP'],
            state_dict['BS'],
            state_dict['Body_Temp'],
            state_dict['BMI'],
            state_dict['Heart_Rate'],
            [float(state_dict['Gestational_Diabetes'])],
            [float(state_dict['Mental_Health'])],
            [float(state_dict['Risk_Level'])]
        ])
        return torch.FloatTensor(state_array).to(self.device)
    
    def select_action(self, state_dict, legal_actions):
        """Select an action using epsilon-greedy policy"""
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = self.state_dict_to_tensor(state_dict).unsqueeze(0)
                q_values = self.policy_net(state)
                
                # Mask illegal actions with large negative values
                mask = torch.ones_like(q_values) * float('-inf')
                for action in legal_actions:
                    mask[0][action] = 0
                
                q_values = q_values + mask
                return q_values.max(1)[1].item()
        else:
            return random.choice(list(legal_actions))
    
    def train(self):
        """Train the agent on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.cat([self.state_dict_to_tensor(s).unsqueeze(0) for s in batch.state])
        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device=self.device)
        next_state_batch = torch.cat([self.state_dict_to_tensor(s).unsqueeze(0) for s in batch.next_state])
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float32)
        
        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute next Q values using target network (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Compute loss and optimize
        loss = nn.MSELoss()(current_q_values.squeeze(1), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.train_step += 1
        if self.train_step % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def save(self, path):
        """Save the model"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load(self, path):
        """Load the model"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
