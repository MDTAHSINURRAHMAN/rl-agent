import sys
import os
import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from agent.ddqn_agent import DDQNAgent
from environment.antenatal_env import AntenatalCareEnv

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_model.pth")
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "notebooks", "CTGAN_Synthetic_ANC_Visits.csv"
)
RESULT_PATH = os.path.join(os.path.dirname(__file__), "ctgan_eval_results.csv")

# Load data
csv = pd.read_csv(DATA_PATH)

# Map CSV columns to environment state keys
csv_cols = [
    "Age",
    "Previous_Complications",
    "Preexisting_Diabetes",
    "Visit",
    "Systolic_BP",
    "Diastolic",
    "BS",
    "Body_Temp",
    "BMI",
    "Heart_Rate",
    "Gestational_Diabetes",
    "Mental_Health",
]


# The environment expects 'Diastolic_BP', not 'Diastolic'
def row_to_state(row):
    state = {
        "Age": np.array([row["Age"]], dtype=np.float32),
        "Previous_Complications": int(round(row["Previous_Complications"])),
        "Preexisting_Diabetes": int(round(row["Preexisting_Diabetes"])),
        "Visit": int(round(row["Visit"])),
        "Systolic_BP": np.array([row["Systolic_BP"]], dtype=np.float32),
        "Diastolic_BP": np.array([row["Diastolic"]], dtype=np.float32),
        "BS": np.array([row["BS"]], dtype=np.float32),
        "Body_Temp": np.array([row["Body_Temp"]], dtype=np.float32),
        "BMI": np.array([row["BMI"]], dtype=np.float32),
        "Heart_Rate": np.array([row["Heart_Rate"]], dtype=np.float32),
        "Gestational_Diabetes": int(round(row["Gestational_Diabetes"])),
        "Mental_Health": int(round(row["Mental_Health"])),
        # Risk_Level will be recomputed by the environment if needed
        "Risk_Level": 0,
    }
    return state


# Initialize environment and agent
env = AntenatalCareEnv()
state_size = 13  # As used in DDQNAgent
action_size = env.action_space.n
agent = DDQNAgent(state_size, action_size, epsilon=0.0)  # Set epsilon=0 for greedy
agent.load(MODEL_PATH)

results = []

for idx, row in csv.iterrows():
    # Set environment state from data row
    env.reset()
    env.current_state = row_to_state(row)
    state = env.current_state.copy()
    flags = env.compute_risk_flags(state)
    legal_actions = env.get_legal_actions(flags, state["Visit"])
    action = agent.select_action(state, legal_actions)
    next_state, reward, done, _, _ = env.step(action)
    results.append(
        {
            "idx": idx,
            "action": action,
            "action_name": env.actions[action],
            "reward": reward,
            "done": done,
            "visit": state["Visit"],
            "Age": state["Age"][0],
            "Systolic_BP": state["Systolic_BP"][0],
            "Diastolic_BP": state["Diastolic_BP"][0],
            "BS": state["BS"][0],
            "BMI": state["BMI"][0],
            "Heart_Rate": state["Heart_Rate"][0],
            "Risk_Level": state["Risk_Level"],
        }
    )

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(RESULT_PATH, index=False)

# Print summary
print(f"Evaluated {len(results)} cases. Action distribution:")
print(results_df["action_name"].value_counts())
print("\nAverage reward:", results_df["reward"].mean())
