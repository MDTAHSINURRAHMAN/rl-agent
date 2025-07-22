import os
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from environment.antenatal_env import AntenatalCareEnv
from agent.ddqn_agent import DDQNAgent

def plot_metrics(rewards, losses, save_dir):
    """Plot and save training metrics"""
    plt.figure(figsize=(12, 5))
    
    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Plot losses
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

def train(episodes=2000, save_dir='models'):
    """Train the DDQN agent"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize environment and agent
    env = AntenatalCareEnv()
    state_size = 13  # Number of features in state
    action_size = env.action_space.n
    agent = DDQNAgent(
        state_size, 
        action_size,
        batch_size=128,
        learning_rate=0.0001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.997,
        target_update=5
    )
    
    # Training metrics
    episode_rewards = []
    training_losses = []
    best_reward = float('-inf')
    
    # Training loop
    for episode in tqdm(range(episodes), desc='Training'):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        done = False
        
        while not done:
            # Get legal actions and select action
            flags = env.compute_risk_flags(state)
            legal_actions = env.get_legal_actions(flags, state['Visit'])
            action = agent.select_action(state, legal_actions)
            
            # Take action and observe next state
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            
            # Store experience and train
            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.train()
            
            if loss is not None:
                episode_loss.append(loss)
            
            state = next_state
            episode_reward += reward
        
        # Record metrics
        episode_rewards.append(episode_reward)
        if episode_loss:
            training_losses.extend(episode_loss)
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(save_dir, 'best_model.pth'))
        
        # Regular checkpoints
        if (episode + 1) % 100 == 0:
            agent.save(os.path.join(save_dir, f'checkpoint_{episode+1}.pth'))
            plot_metrics(episode_rewards, training_losses, save_dir)
    
    # Final save and plots
    agent.save(os.path.join(save_dir, 'final_model.pth'))
    plot_metrics(episode_rewards, training_losses, save_dir)
    
    return agent, episode_rewards, training_losses

if __name__ == '__main__':
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train the agent
    agent, rewards, losses = train()
    
    print(f'Training completed.\nFinal average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}')
