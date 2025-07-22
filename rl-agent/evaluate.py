import os
import numpy as np
import torch
from environment.antenatal_env import AntenatalCareEnv
from agent.ddqn_agent import DDQNAgent

def evaluate_episode(env, agent, render=False):
    """Evaluate a single episode"""
    state, _ = env.reset()
    total_reward = 0
    actions_taken = []
    flags_history = []
    done = False
    
    while not done:
        # Get current flags and legal actions
        flags = env.compute_risk_flags(state)
        legal_actions = env.get_legal_actions(flags, state['Visit'])
        
        # Select action (no exploration during evaluation)
        agent.epsilon = 0
        action = agent.select_action(state, legal_actions)
        
        # Take action
        next_state, reward, done, truncated, _ = env.step(action)
        done = done or truncated
        
        # Record step information
        total_reward += reward
        actions_taken.append(env.actions[action])
        flags_history.append(flags)
        
        if render:
            print(f"\nVisit {state['Visit']}")
            print("Risk Flags:", {k: v for k, v in flags.items() if v})
            print(f"Action taken: {env.actions[action]}")
            print(f"Reward: {reward}")
        
        state = next_state
    
    return total_reward, actions_taken, flags_history

def evaluate(model_path, n_episodes=100, render_episodes=5):
    """Evaluate the trained agent"""
    # Initialize environment and agent
    env = AntenatalCareEnv()
    state_size = 13
    action_size = env.action_space.n
    agent = DDQNAgent(state_size, action_size)
    
    # Load trained model
    agent.load(model_path)
    
    # Evaluation metrics
    rewards = []
    all_actions = []
    all_flags = []
    
    # Evaluate episodes
    for episode in range(n_episodes):
        render = episode < render_episodes
        reward, actions, flags = evaluate_episode(env, agent, render)
        rewards.append(reward)
        all_actions.append(actions)
        all_flags.append(flags)
        
        if render:
            print(f"\nEpisode {episode + 1}")
            print(f"Total Reward: {reward}")
            print("=" * 50)
    
    # Print summary statistics
    print("\nEvaluation Summary:")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    
    # Analyze action distribution
    all_actions_flat = [action for episode_actions in all_actions for action in episode_actions]
    unique_actions, action_counts = np.unique(all_actions_flat, return_counts=True)
    
    print("\nAction Distribution:")
    for action, count in zip(unique_actions, action_counts):
        percentage = (count / len(all_actions_flat)) * 100
        print(f"{action}: {percentage:.1f}%")
    
    # Analyze risk flag patterns
    risk_flags = {}
    for episode_flags in all_flags:
        for step_flags in episode_flags:
            for flag, value in step_flags.items():
                if value:
                    risk_flags[flag] = risk_flags.get(flag, 0) + 1
    
    print("\nRisk Flag Frequencies:")
    total_steps = sum(len(episode_flags) for episode_flags in all_flags)
    for flag, count in sorted(risk_flags.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_steps) * 100
        print(f"{flag}: {percentage:.1f}%")

if __name__ == '__main__':
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Evaluate the best model
    model_path = os.path.join('models', 'best_model.pth')
    evaluate(model_path)
