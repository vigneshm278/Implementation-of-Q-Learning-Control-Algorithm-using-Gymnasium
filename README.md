# Implementation-of-Q-Learning-Control-Algorithm-using-Gymnasium

## Aim

To implement the **Q-Learning control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an optimal action-value function that enables the agent to select suitable actions for reaching the goal state while avoiding holes.

---

## Problem Statement

The objective is to implement a model-free Q-Learning control algorithm for the FrozenLake-v1 environment provided by Gymnasium.

The agent starts from a starting state and must learn how to reach the goal while avoiding hole states. Since the agent does not initially know which actions are good or bad, it learns through repeated interaction with the environment.

The Q-Learning algorithm maintains a Q-table containing the estimated value of taking each possible action in each state. Through repeated exploration and Q-value updates, the agent learns an optimal policy.

The main objectives are:

Create the FrozenLake environment.
Initialize the Q-table.
Implement epsilon-greedy action selection.
Train the agent using the Q-Learning update rule.
Estimate the state-value function.
Extract the learned optimal policy.
Analyze the learning performance using the average reward.



## Software Requirements
Hardware Requirements
Computer/Laptop
Minimum 4 GB RAM
Processor: Intel i3 or equivalent and above
Software Requirements
Python 3.x
Jupyter Notebook / Google Colab
Gymnasium
NumPy
Matplotlib



## Environment Description

The experiment uses the FrozenLake-v1 environment from Gymnasium.

FrozenLake is a grid-world environment in which an agent must move from a starting position to a goal while avoiding holes.

A typical 4 × 4 environment is:

S F F F
F H F H
F F F H
H F F G

Where:

Symbol	Meaning
S	Starting state
F	Frozen/Safe state
H	Hole
G	Goal


## Theory

Q-Learning estimates the optimal action-value function directly.

The action-value function $Q(s,a)$ represents the expected return obtained when the agent takes action $a$ in state $s$, and then follows the best possible policy afterward.

The Q-Learning update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma \max_{a} Q(S_{t+1},a) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |
| $max_{a} Q(S_{t+1},a)$ | Maximum action value in the next state |

---

## Epsilon-Greedy Action Selection

During training, the agent uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_{a} Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---

## Algorithm
Q-Learning Algorithm

Step 1: Create the FrozenLake environment.

Step 2: Determine the number of states and actions.

Step 3: Initialize the Q-table with zeros.

Step 4: Set the hyperparameters:

Learning rate alpha
Discount factor gamma
Exploration rate epsilon
Minimum epsilon
Epsilon decay
Number of episodes

Step 5: For each episode, reset the environment.

Step 6: Select an action using the epsilon-greedy strategy.

Step 7: Execute the selected action.

Step 8: Observe:

Next state
Reward
Termination status

Step 9: Calculate the Q-Learning target:

$$ Target = R + \gamma \max_a Q(S',a) $$

Step 10: Update the Q-value:

$$ Q(S,A) \leftarrow Q(S,A) +\alpha[Target-Q(S,A)] $$

Step 11: Move to the next state.

Step 12: Continue until the episode terminates.

Step 13: Reduce epsilon after each episode.

Step 14: Repeat the process for all training episodes.

Step 15: Extract the state-value function and learned policy.

Step 16: Calculate the average reward of the last 1000 episodes.

Step 17: Plot the learning curve.

## Name: Vignesh M
## Reg No: 212223240176
## Python Program

```python

# -------------------------------------------------
# Import Libraries
# -------------------------------------------------

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------

env = gym.make(
    "FrozenLake-v1",
    is_slippery=False
)

num_states = env.observation_space.n
num_actions = env.action_space.n

print("Number of states:", num_states)
print("Number of actions:", num_actions)


# -------------------------------------------------
# Hyperparameters
# -------------------------------------------------

alpha = 0.1
gamma = 0.99

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9995

num_episodes = 20000


# -------------------------------------------------
# Initialize Q-table
# -------------------------------------------------

Q = np.zeros(
    (num_states, num_actions)
)


# -------------------------------------------------
# Epsilon-Greedy Action Selection
# -------------------------------------------------

def choose_action(state, epsilon):

    if np.random.random() < epsilon:

        # Exploration
        return env.action_space.sample()

    else:

        # Exploitation
        return np.argmax(Q[state])


# -------------------------------------------------
# Q-Learning Training
# -------------------------------------------------

episode_rewards = []

for episode in range(num_episodes):

    # Reset environment
    state, info = env.reset()

    terminated = False
    truncated = False

    total_reward = 0

    while not (terminated or truncated):

        # Choose action
        action = choose_action(
            state,
            epsilon
        )

        # Take action
        next_state, reward, terminated, truncated, info = env.step(
            action
        )

        # Calculate target
        if terminated:

            target = reward

        else:

            target = reward + gamma * np.max(
                Q[next_state]
            )

        # Q-Learning update
        Q[state, action] = Q[state, action] + alpha * (
            target - Q[state, action]
        )

        # Move to next state
        state = next_state

        # Store reward
        total_reward += reward

    # Store total reward of episode
    episode_rewards.append(
        total_reward
    )

    # Decay epsilon
    epsilon = max(
        epsilon_min,
        epsilon * epsilon_decay
    )


print("\nTraining completed!")
print("Final epsilon:", epsilon)


# -------------------------------------------------
# Extract State Values and Learned Policy
# -------------------------------------------------

state_values = np.max(
    Q,
    axis=1
)

learned_policy = np.argmax(
    Q,
    axis=1
)


# -------------------------------------------------
# Display Functions
# -------------------------------------------------

def print_value_function(values):

    print("\nEstimated State-Value Function:")

    print(
        np.round(
            values.reshape(4, 4),
            3
        )
    )


def print_policy(policy):

    action_symbols = {
        0: "L",
        1: "D",
        2: "R",
        3: "U"
    }

    policy_grid = np.array(
        [
            action_symbols[action]
            for action in policy
        ]
    ).reshape(4, 4)

    print("\nLearned Policy:")

    print(policy_grid)


# -------------------------------------------------
# Output
# -------------------------------------------------

print("\nFinal Q-table:")

print(
    np.round(
        Q,
        3
    )
)

print_value_function(
    state_values
)

print_policy(
    learned_policy
)

average_reward = np.mean(
    episode_rewards[-1000:]
)

print(
    "\nAverage reward over last 1000 episodes:",
    average_reward
)


# -------------------------------------------------
# Plot Learning Curve
# -------------------------------------------------

window = 500

moving_average = np.convolve(
    episode_rewards,
    np.ones(window) / window,
    mode="valid"
)

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    moving_average
)

plt.xlabel(
    "Episode"
)

plt.ylabel(
    "Average Reward"
)

plt.title(
    "Q-Learning Curve - FrozenLake"
)

plt.grid(
    True
)

plt.show()


# -------------------------------------------------
# Close Environment
# -------------------------------------------------

env.close()

```
---

## Output

Final Q-table:

<img width="1225" height="406" alt="image" src="https://github.com/user-attachments/assets/e865782d-26d8-4318-a7cf-3fc2f0152641" />




Estimated State-Value Function:

<img width="1321" height="137" alt="image" src="https://github.com/user-attachments/assets/3d5c9c9c-5236-4643-94f5-903d41ee4f36" />





Learned Policy:

<img width="1182" height="138" alt="image" src="https://github.com/user-attachments/assets/b3314007-bb3c-442e-882b-ad91cbf6101e" />



Average reward over last 1000 episodes: 
<img width="1332" height="50" alt="image" src="https://github.com/user-attachments/assets/a232e576-7241-449a-8f56-122c71867f09" />

Q-Learning Curve - FrozenLake
<img width="1508" height="602" alt="image" src="https://github.com/user-attachments/assets/7473f3b4-55bd-437e-b00b-7cc9f0d23a6b" />



---

## Result
The Q-Learning control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment.

---

## Inference
Q-Learning learns without requiring a predefined model of the environment.The Q-table stores the estimated value of every state-action pair.Initially, all Q-values are zero because the agent has no prior knowledge.The agent uses epsilon-greedy exploration to discover useful actions.---

