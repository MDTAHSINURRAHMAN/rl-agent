import os
import sys
import numpy as np
import torch
import gymnasium as gym
from environment.antenatal_env import AntenatalCareEnv
from agent.ddqn_agent import DDQNAgent

def train_agent_with_ctgan(
    data_path,
    output_dir='models',
    num_episodes=2000,
    max_steps=9,  # Max number of visits
    batch_size=32,
    learning_rate=0.001,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    target_update=10,
    save_interval=100,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train the DDQN agent using CTGAN-generated or real patient data.
    
    Args:
        data_path (str): Path to the CSV file containing patient data
        output_dir (str): Directory to save model checkpoints
        num_episodes (int): Number of training episodes
        max_steps (int): Maximum steps per episode
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for the optimizer
        gamma (float): Discount factor
        epsilon_start (float): Starting value of epsilon for epsilon-greedy action selection
        epsilon_end (float): Minimum value of epsilon
        epsilon_decay (float): Decay rate of epsilon
        target_update (int): How often to update target network
        save_interval (int): How often to save model checkpoints
        device (str): Device to use for training ('cuda' or 'cpu')
    """
    # Create environment with data
    env = AntenatalCareEnv(data_path)
    
    # Initialize agent
    # Calculate state dimension properly handling different space types
    state_dim = 0
    for space in env.observation_space.values():
        if isinstance(space, gym.spaces.Box):
            state_dim += space.shape[0]  # For continuous variables
        elif isinstance(space, gym.spaces.Discrete):
            state_dim += 1  # For discrete variables
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")
    
    action_dim = env.action_space.n
    
    agent = DDQNAgent(
        state_size=state_dim,
        action_size=action_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon_start,
        epsilon_min=epsilon_end,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        target_update=target_update,
        device=device
    )
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Training metrics
    total_rewards = []
    avg_rewards = []
    episode_lengths = []
    
    # Training loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        step = 0
        done = False
        
        while not done and step < max_steps:
            # Get legal actions
            legal_actions = env.get_legal_actions(env.compute_risk_flags(state), state['Visit'])
            
            # Get action
            action = agent.select_action(state, legal_actions)
            
            # Take action
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Store experience
            agent.memory.push(state, action, reward, next_state, done)
            
            # Learn from experience
            if len(agent.memory) > batch_size:
                loss = agent.train()
            
            state = next_state
            episode_reward += reward
            step += 1
        
        total_rewards.append(episode_reward)
        episode_lengths.append(step)
        
        # Calculate running average
        window_size = min(100, len(total_rewards))
        avg_reward = np.mean(total_rewards[-window_size:])
        avg_rewards.append(avg_reward)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Average Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_{episode+1}.pth')
            agent.save(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.pth')
    agent.save(final_model_path)
    print(f"Saved final model to {final_model_path}")
    
    # Save training metrics
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(avg_rewards, label='Average Reward')
    plt.plot(total_rewards, alpha=0.3, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_metrics.png'))
    plt.close()

if __name__ == "__main__":
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Construct path to data file
    data_path = os.path.join(project_root, "notebooks", "CTGAN_Synthetic_ANC_Visits.csv")
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)
        
    train_agent_with_ctgan(data_path)
