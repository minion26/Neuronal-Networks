# Deep Q-Learning

## Task Description
The task is to implement a Deep Q-Learning algorithm to solve the CartPole-v1 environment from OpenAI Gym. 
The algorithm should be able to learn how to balance a pole on a cart by moving the cart left or right. 
The agent receives a reward of +1 for each time step the pole is balanced.


## Explanation of the Solved Task

### 1. Introduction
Deep Q-Learning is a powerful reinforcement learning technique widely used to train agents in 
environments with discrete action spaces. It is often tested on **CartPole** due to its simplicity and 
suitability for understanding the core concepts of reinforcement learning. In CartPole, the agent has two actions: 
moving the cart left or right, and the goal is to balance the pole as long as possible. 
Rewards are sparse, making it an ideal starting point for applying reinforcement learning 
methods before moving on to more complex environments.


### 2. Project Structure
- **Actions:**
  - `0`: Move cart left.
  - `1`: Move cart right.

- **Exploration vs. Exploitation:**
  - **Exploration:** The agent starts by exploring randomly (epsilon = 1) and gradually shifts to exploitation (epsilon → 0.005).
  - **Exploitation:** After the exploration phase, the agent chooses actions based on the highest Q-value (maximum reward) from its learned policy.

1. **Experience Replay:**
   - The agent stores its experiences `(state, action, reward, next_state, and termination status)` in a memory buffer for later use.
   - The experiences are sampled in **mini-batches** to train the agent.

2. **Neural Networks:**
   - **Prediction Network:** This network predicts the Q-values for each state-action pair. It is updated frequently based on the experiences.
   - **Target Network:** This network provides stable target Q-values. It is updated less frequently than the prediction network to avoid instability during training.

3. **Optimization:**
   - The agent minimizes the **loss function**, which measures the difference between the predicted Q-values and the target Q-values. The target Q-value is calculated as the maximum reward the agent can achieve from a given state.


### 3. Training Process

| Episode | Best Reward |
|---------|-------------|
| 0       | 26.0        |
| 10      | 30.0        |
| 12      | 34.0        |
| 27      | 47.0        |
| 43      | 49.0        |
| 56      | 55.0        |
| 57      | 75.0        |
| 93      | 138.0       |
| 114     | 139.0       |
| 115     | 154.0       |
| 136     | 236.0       |
| 138     | 313.0       |
| 191     | 836.0       |
| 203     | 926.0       |
| 218     | 1770.0      |
| 221     | 2883.0      |
| 227     | 3928.0      |
| 237     | 5583.0      |
| 241     | 6093.0      |




    
### 4. Conclusion
The agent learns by exploring states and actions, balancing between exploration and exploitation. 
Experiences are stored and used for training via a replay mechanism. Two networks ensure stable and effective Q-learning: 
one for predictions and another for setting targets. The training loop minimizes the loss to improve Q-value approximations, 
enabling the agent to choose actions that maximize long-term rewards.


