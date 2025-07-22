import numpy as np
import gymnasium as gym
from gymnasium import spaces

class AntenatalCareEnv(gym.Env):
    """
    Antenatal Care Environment for reinforcement learning.
    Based on WHO and ACOG clinical guidelines for maternal care.
    """
    
    def __init__(self):
        super().__init__()
        
        # Define the observation space (13 features)
        self.observation_space = spaces.Dict({
            'Age': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            'Previous_Complications': spaces.Discrete(2),
            'Preexisting_Diabetes': spaces.Discrete(2),
            'Visit': spaces.Discrete(10),  # Visits 1-9
            'Systolic_BP': spaces.Box(low=0, high=250, shape=(1,), dtype=np.float32),
            'Diastolic_BP': spaces.Box(low=0, high=150, shape=(1,), dtype=np.float32),
            'BS': spaces.Box(low=0, high=500, shape=(1,), dtype=np.float32),
            'Body_Temp': spaces.Box(low=35, high=42, shape=(1,), dtype=np.float32),
            'BMI': spaces.Box(low=10, high=50, shape=(1,), dtype=np.float32),
            'Heart_Rate': spaces.Box(low=40, high=200, shape=(1,), dtype=np.float32),
            'Gestational_Diabetes': spaces.Discrete(2),
            'Mental_Health': spaces.Discrete(2),
            'Risk_Level': spaces.Discrete(2)
        })
        
        # Define action space (15 possible actions)
        self.action_space = spaces.Discrete(15)
        
        # Action mapping
        self.actions = {
            0: 'Routine_Bundle',
            1: 'Order_Repeat_BP',
            2: 'Refer_ED',
            3: 'Order_OGTT',
            4: 'Nutrition_Counsel',
            5: 'Glucose_Education',
            6: 'Insulin_Escalation',
            7: 'MH_Referral',
            8: 'Danger_Sign_Education',
            9: 'Fetal_Movement_Edu',
            10: 'Order_Growth_US',
            11: 'Give_Tetanus_Toxoid',
            12: 'Schedule_Extra_Visit',
            13: 'Schedule_Specialist_HTN',
            14: 'Plan_Induction'
        }
        
        self.reset()

    def compute_risk_flags(self, state):
        """Compute clinical risk flags based on state variables."""
        flags = {
            'age_high_risk': state['Age'][0] < 18 or state['Age'][0] >= 35,
            'htn_mild': (140 <= state['Systolic_BP'][0] < 160) or (90 <= state['Diastolic_BP'][0] < 110),
            'htn_severe': state['Systolic_BP'][0] >= 160 or state['Diastolic_BP'][0] >= 110,
            'gdm_screen_fail': state['BS'][0] >= 140,
            'gdm_severe': state['BS'][0] >= 200,
            'underweight': state['BMI'][0] < 18.5,
            'obese': state['BMI'][0] >= 30,
            'fever': state['Body_Temp'][0] >= 38,
            'maternal_tachy': state['Heart_Rate'][0] > 110,
            'diabetes_any': bool(state['Preexisting_Diabetes'] or state['Gestational_Diabetes']),
            'mh_positive': state['Mental_Health'] == 1,
            'model_high_risk': state['Risk_Level'] == 1,
            'previous_complications': state['Previous_Complications'] == 1,
            'preexisting_diabetes': state['Preexisting_Diabetes'] == 1
        }
        return flags

    def get_legal_actions(self, flags, visit):
        """Determine legal actions based on risk flags and visit number."""
        legal_actions = {0}  # Routine_Bundle is always legal
        
        # Track visit actions
        if not hasattr(self, 'visit_actions'):
            self.visit_actions = {}
        if visit not in self.visit_actions:
            self.visit_actions[visit] = set()
            
        # First action must handle critical conditions
        if not self.visit_actions[visit]:
            # Emergency conditions
            if flags['htn_severe'] or flags['fever'] or flags['maternal_tachy']:
                return {2}  # Only ED referral allowed
            
            # Critical conditions
            critical_actions = set()
            if flags['htn_mild']:
                critical_actions.add(1)  # BP check
            if flags['mh_positive'] and not hasattr(self, 'mh_referred'):
                critical_actions.add(7)  # Mental health referral
            if flags['model_high_risk'] and (
                not hasattr(self, 'last_danger_signs_visit') or 
                visit - getattr(self, 'last_danger_signs_visit', 0) >= 3
            ):
                critical_actions.add(8)  # Danger sign education
            
            if critical_actions:
                return critical_actions
            
            # Allow routine bundle if no critical conditions
            return {0}
            
        # After routine bundle, prioritize based on conditions
        # Emergency conditions
        if flags['htn_severe'] or flags['fever'] or flags['maternal_tachy']:
            if 2 not in self.visit_actions[visit]:  # ED referral not done yet
                legal_actions = {2}  # Only ED referral allowed
                return legal_actions
                
        # High priority conditions
        if flags['htn_mild']:
            if 1 not in self.visit_actions[visit]:  # BP check not done yet
                legal_actions.add(1)
            if not hasattr(self, 'htn_specialist_referred'):
                legal_actions.add(13)  # Specialist referral
                
        if flags['gdm_screen_fail'] and 4 <= visit <= 6:
            if not hasattr(self, 'ogtt_ordered'):
                legal_actions.add(3)  # OGTT ordering
                
        if flags['gdm_severe'] and 6 not in self.visit_actions[visit]:
            legal_actions.add(6)  # Insulin Escalation
            
        if flags['mh_positive'] and not hasattr(self, 'mh_referred'):
            legal_actions.add(7)  # Mental Health Referral
            
        # Emergency care takes precedence
        if flags['htn_severe'] or flags['fever'] or flags['maternal_tachy']:
            legal_actions.add(2)  # Refer_ED
            return legal_actions  # Emergency referral is the only action allowed
            
        # Urgent care for hypertension
        if flags['htn_mild'] or flags['htn_severe']:
            legal_actions.add(1)  # Order_Repeat_BP
            
        if (flags['htn_mild'] or flags['preexisting_diabetes']) and \
           not hasattr(self, 'htn_specialist_referred'):
            legal_actions.add(13)  # HTN/Diabetes Specialist Referral
            
        # GDM screening and management    
        if 4 <= visit <= 6 and flags['gdm_screen_fail'] and not hasattr(self, 'ogtt_ordered'):
            legal_actions.add(3)  # Order OGTT
            
        if flags['gdm_severe'] and not hasattr(self, 'insulin_started'):
            legal_actions.add(6)  # Insulin Escalation
            
        # Mental health care
        if flags['mh_positive'] and not hasattr(self, 'mh_referred'):
            legal_actions.add(7)  # Mental Health Referral
            
        # Regular care after handling urgent conditions
        if (flags['underweight'] or flags['obese']) and not hasattr(self, 'nutrition_counseled'):
            legal_actions.add(4)  # Nutrition Counseling
            
        if flags['diabetes_any'] and not hasattr(self, 'glucose_education_given'):
            legal_actions.add(5)  # Glucose Education
            
        # Risk-based education and monitoring
        if flags['model_high_risk']:
            if not hasattr(self, 'last_danger_signs_visit') or \
               (visit - getattr(self, 'last_danger_signs_visit', 0) >= 3):
                legal_actions.add(8)  # Danger Sign Education
                
        # Late pregnancy care
        if 7 <= visit <= 9:
            if not hasattr(self, 'fetal_movement_taught'):
                legal_actions.add(9)  # Fetal Movement Education
            if flags['obese'] and not hasattr(self, 'growth_us_ordered'):
                legal_actions.add(10)  # Growth US for obese patients
                
        # Standard interventions
        if visit in {2, 4} and not hasattr(self, f'tetanus_given_visit_{visit}'):
            legal_actions.add(11)  # Tetanus Toxoid
            
        # Additional monitoring for high-risk patients
        if (flags['age_high_risk'] or flags['previous_complications']):
            if not hasattr(self, 'last_extra_visit_scheduled') or \
               (visit - getattr(self, 'last_extra_visit_scheduled', 0) >= 3):
                legal_actions.add(12)  # Extra Visit Scheduling
                
        # Delivery planning
        if visit == 9 and flags['model_high_risk'] and not hasattr(self, 'induction_planned'):
            legal_actions.add(14)  # Plan Induction
            
        return legal_actions

    def calculate_reward(self, action, flags):
        """Calculate reward based on action and current risk flags."""
        reward = 0
        visit = self.current_state['Visit']
        
        # Initialize visit tracking if needed
        if not hasattr(self, 'visit_actions'):
            self.visit_actions = {}
        if visit not in self.visit_actions:
            self.visit_actions[visit] = set()
        self.visit_actions[visit].add(action)
        
        # Base reward for routine bundle (essential care)
        if action == 0:  # Routine_Bundle
            if not self.visit_actions[visit] - {0}:  # If this is the first action in the visit
                reward += 8  # Higher reward for doing routine bundle first
            else:
                reward += 3  # Lower reward for doing it later
        elif 0 not in self.visit_actions[visit]:
            reward -= 5  # Increased penalty for missing routine bundle
            
        # Priority-based rewards
        has_emergency = flags['htn_severe'] or flags['fever'] or flags['maternal_tachy']
        has_urgent = flags['htn_mild'] or flags['gdm_severe']
        
        # Emergency care reward boost - Highest priority
        if flags['htn_severe'] or flags['fever'] or flags['maternal_tachy']:
            if action == 2:  # Correct ED referral
                reward += 30  # Increased reward for life-saving action
            elif action != 2 and 2 not in self.visit_actions[visit]:  # Any other action when ED needed
                reward -= 25  # Stronger penalty for missing critical care
        
        # Urgent care reward boost - Second priority
        if flags['htn_mild'] or flags['gdm_severe']:
            if action == 1 and flags['htn_mild']:  # BP check for mild HTN
                reward += 15  # Increased reward for proper monitoring
            elif action == 6 and flags['gdm_severe']:  # Insulin for severe GDM
                reward += 15  # Matching reward for proper GDM management
            
        # High-risk specialist referrals - Third priority
        if (flags['htn_mild'] or flags['preexisting_diabetes']):
            if action == 13 and not hasattr(self, 'htn_specialist_referred'):
                reward += 20  # Increased reward for proper specialist care
                self.htn_specialist_referred = True
            elif action == 0 and not hasattr(self, 'htn_specialist_referred'):  # Routine care without referral
                reward -= 5  # Penalty for missing specialist care
        
        # Regular care actions with strong risk-based adjustments
        # Base rewards for actions
        base_rewards = {
            0: 10,  # Routine Bundle
            1: 15,  # BP Check
            2: 30,  # ED Referral
            7: 25,  # Mental Health Referral
            8: 20,  # Danger Sign Education
        }
        reward += base_rewards.get(action, 10)
        
        # Penalties for inappropriate actions
        if action == 0:  # Routine Bundle
            # Check for unhandled critical conditions
            if flags['htn_severe'] or flags['fever'] or flags['maternal_tachy']:
                reward = -30  # Severe penalty for missing emergency
                return reward
                
            if len(self.visit_actions[visit]) == 0:  # First action
                if flags['htn_mild'] or flags['mh_positive'] or flags['model_high_risk']:
                    reward = -20  # Penalty for wrong prioritization
                    return reward
            
            # Adjust reward based on risk status
            if any(flags.values()):
                reward = max(0, reward - 5)  # Reduced reward with risks
                
        # Bonus for proper risk management
        elif action in [1, 7, 8]:  # Risk management actions
            if len(self.visit_actions[visit]) == 0:  # First action
                reward += 10  # Bonus for proper prioritization
        
        # Risk-based education with strong incentives
        if flags['model_high_risk']:
            if action == 8:  # Danger sign education
                base_reward = 25  # High base reward
                if len(self.visit_actions[visit]) == 0:
                    base_reward += 15  # Major bonus for prioritizing education
                if not hasattr(self, 'last_danger_signs_visit'):
                    base_reward += 10  # Extra for first education session
                reward += base_reward
            elif action == 0 and not hasattr(self, 'last_danger_signs_visit'):
                reward -= 20  # Severe penalty for missing education
            
        # Diabetes screening and management
        if flags['gdm_screen_fail'] and 4 <= visit <= 6:
            if action == 3:  # OGTT ordering
                reward += 15  # Higher reward for proper screening
                self.ogtt_ordered = True
            elif visit == 6 and not hasattr(self, 'ogtt_ordered'):
                reward -= 10  # Increased penalty for missing screening window
        
        if flags['diabetes_any']:
            if action == 5 and not hasattr(self, 'glucose_education_given'):
                reward += 12  # Increased reward for education
                self.glucose_education_given = True
        
        # Mental health care with immediate action requirement
        if flags['mh_positive']:
            if action == 7 and not hasattr(self, 'mh_referred'):
                base_reward = 30  # Very high base reward
                if len(self.visit_actions[visit]) == 0:
                    base_reward += 20  # Major bonus for immediate attention
                reward += base_reward
                self.mh_referred = True
            elif action == 0:  # Penalties for routine care
                if not hasattr(self, 'mh_referred'):
                    reward -= 25  # Severe penalty for missing mental health care
                if len(self.visit_actions[visit]) == 0:
                    reward -= 15  # Additional penalty for wrong prioritization
        
        # Nutritional counseling
        if flags['underweight'] or flags['obese']:
            if action == 4 and not hasattr(self, 'nutrition_counseled'):
                reward += 15  # Higher reward for nutrition care
                self.nutrition_counseled = True
                if len(self.visit_actions[visit]) == 0:
                    reward += 5  # Small bonus for early intervention
            if flags['obese'] and 7 <= visit <= 9 and action == 10:
                reward += 6
        
        # High risk patient management
        if flags['model_high_risk']:
            if action == 8:  # Danger sign education
                if not hasattr(self, 'last_danger_signs_visit'):  # First time
                    reward += 8
                    self.last_danger_signs_visit = visit
                elif visit - self.last_danger_signs_visit >= 3:  # Repeat after 3 visits
                    reward += 5
                    self.last_danger_signs_visit = visit
                else:  # Too frequent repetition
                    reward -= 2
            if visit == 9 and action == 14:  # Induction planning
                reward += 10
        
        # Schedule management
        if (flags['age_high_risk'] or flags['previous_complications']):
            if action == 12:  # Extra visit scheduling
                if not hasattr(self, 'last_extra_visit_scheduled'):
                    reward += 8
                    self.last_extra_visit_scheduled = visit
                elif visit - self.last_extra_visit_scheduled >= 3:  # Allow scheduling every 3 visits
                    reward += 5
                    self.last_extra_visit_scheduled = visit
                else:
                    reward -= 2
        
        # Tetanus prophylaxis
        if visit in {2, 4}:
            if action == 11:  # Correct timing for tetanus
                reward += 8
                setattr(self, f'tetanus_given_visit_{visit}', True)
        elif action == 11:  # Wrong timing for tetanus
                reward -= 5
        
        # Fetal movement education (late pregnancy)
        if 7 <= visit <= 9:
            if action == 9 and not hasattr(self, 'fetal_movement_taught'):
                reward += 8
                self.fetal_movement_taught = True
            
        return reward

    def step(self, action):
        """Execute one step in the environment."""
        if action not in self.get_legal_actions(self.compute_risk_flags(self.current_state), 
                                              self.current_state['Visit']):
            return self.current_state, -10, True, False, {}  # Illegal action penalty
        
        # Calculate reward based on action and current state
        flags = self.compute_risk_flags(self.current_state)
        reward = self.calculate_reward(action, flags)
        
        # Store current visit number for reference
        current_visit = self.current_state['Visit']
        
        # Update visit counter
        self.current_state['Visit'] += 1
        
        # Simulate realistic state transitions based on pregnancy progression and interventions
        # Blood pressure tends to rise as pregnancy progresses
        if action != 13:  # If not referred to specialist
            bp_change = np.random.normal(2.0, 1.0) * (self.current_state['Visit'] / 3)
        else:  # Better BP control with specialist care
            bp_change = np.random.normal(1.0, 0.5) * (self.current_state['Visit'] / 3)
        self.current_state['Systolic_BP'][0] += bp_change
        self.current_state['Diastolic_BP'][0] += bp_change * 0.5
        
        # Blood sugar changes
        if flags['diabetes_any']:
            bs_change = np.random.normal(5.0, 2.0)
        else:
            bs_change = np.random.normal(2.0, 1.0)
        if action in [5, 6]:  # Glucose education or insulin
            bs_change *= 0.5  # Better control with intervention
        self.current_state['BS'][0] += bs_change
        
        # Update GDM status based on blood sugar
        if self.current_state['BS'][0] >= 140 and 4 <= self.current_state['Visit'] <= 6:
            self.current_state['Gestational_Diabetes'] = 1
        
        # Temperature fluctuations
        self.current_state['Body_Temp'][0] += np.random.normal(0, 0.2)
        
        # Heart rate changes
        self.current_state['Heart_Rate'][0] += np.random.normal(0, 5.0)
        
        # BMI changes through pregnancy
        bmi_change = np.random.normal(0.3, 0.1)
        self.current_state['BMI'][0] += bmi_change
        
        # Mental health can deteriorate without intervention
        if flags['mh_positive'] and action != 7:  # No mental health referral
            if np.random.random() < 0.7:  # 70% chance to stay positive
                self.current_state['Mental_Health'] = 1
        
        # Update risk level based on current state
        high_risk_conditions = sum([
            flags['htn_severe'],
            flags['gdm_severe'],
            flags['maternal_tachy'],
            flags['age_high_risk'],
            flags['previous_complications']
        ])
        self.current_state['Risk_Level'] = int(high_risk_conditions >= 1)
        
        # Clamp values to reasonable ranges
        self.current_state['Systolic_BP'][0] = np.clip(self.current_state['Systolic_BP'][0], 90, 200)
        self.current_state['Diastolic_BP'][0] = np.clip(self.current_state['Diastolic_BP'][0], 60, 120)
        self.current_state['BS'][0] = np.clip(self.current_state['BS'][0], 70, 250)
        self.current_state['Body_Temp'][0] = np.clip(self.current_state['Body_Temp'][0], 36, 40)
        self.current_state['Heart_Rate'][0] = np.clip(self.current_state['Heart_Rate'][0], 60, 140)
        self.current_state['BMI'][0] = np.clip(self.current_state['BMI'][0], 16, 45)
        
        done = self.current_state['Visit'] >= 9
        return self.current_state, reward, done, False, {}

    def reset(self, seed=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Clear all intervention tracking flags
        tracking_attrs = [
            'htn_specialist_referred',
            'extra_visit_scheduled',
            'ogtt_ordered',
            'glucose_education_given',
            'mh_referred',
            'nutrition_counseled',
            'danger_signs_taught',
            'fetal_movement_taught',
            'tetanus_given_visit_2',
            'tetanus_given_visit_4',
            'actions_this_visit',
            'next_visit'
        ]
        for attr in tracking_attrs:
            if hasattr(self, attr):
                delattr(self, attr)
        
        # Initialize age with realistic distribution
        age = np.random.normal(28, 6)  # Mean 28, SD 6 years
        age = np.clip(age, 16, 45)
        
        # Initialize risk factors
        prev_complications = np.random.choice([0, 1], p=[0.8, 0.2])
        preexisting_diabetes = np.random.choice([0, 1], p=[0.95, 0.05])
        
        # Initialize vitals with some randomness
        systolic_bp = np.random.normal(120, 10)
        diastolic_bp = np.random.normal(80, 8)
        blood_sugar = np.random.normal(100, 15)
        body_temp = np.random.normal(37, 0.3)
        bmi = np.random.normal(24, 4)
        heart_rate = np.random.normal(80, 8)
        
        # Initialize mental health and GDM
        mental_health = np.random.choice([0, 1], p=[0.9, 0.1])
        gestational_diabetes = 0  # Always starts at 0, can develop later
        
        # Compute initial risk level
        high_risk_conditions = sum([
            age < 18 or age >= 35,
            prev_complications == 1,
            preexisting_diabetes == 1,
            systolic_bp >= 140 or diastolic_bp >= 90,
            bmi < 18.5 or bmi >= 30
        ])
        risk_level = int(high_risk_conditions >= 1)
        
        # Initialize state with computed values
        self.current_state = {
            'Age': np.array([age], dtype=np.float32),
            'Previous_Complications': prev_complications,
            'Preexisting_Diabetes': preexisting_diabetes,
            'Visit': 1,
            'Systolic_BP': np.array([systolic_bp], dtype=np.float32),
            'Diastolic_BP': np.array([diastolic_bp], dtype=np.float32),
            'BS': np.array([blood_sugar], dtype=np.float32),
            'Body_Temp': np.array([body_temp], dtype=np.float32),
            'BMI': np.array([bmi], dtype=np.float32),
            'Heart_Rate': np.array([heart_rate], dtype=np.float32),
            'Gestational_Diabetes': gestational_diabetes,
            'Mental_Health': mental_health,
            'Risk_Level': risk_level
        }
        
        return self.current_state, {}
