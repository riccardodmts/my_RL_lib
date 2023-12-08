# Gymansium Acrobot-v1, results with PPO

PPO params used:

- $\gamma=0.99$.
- Number of actors: 32.
- Training steps: 60.
- Trajectory length: $T=500$.
- $\lambda=0.94$.
- PPO clipping param: $\epsilon=0.2$.
- Optimization:
  1. Number of epochs: 4.
  2. Mini-batch size: 64.
  3. Optimizer: Adam with $\beta_1=0.9, \beta_2=0.99$.
  4. Learning rate: 0.0003.
- $c1$ (coefficient for state-value function loss): 0.5.
- No entropy penalty ($c_2=0$).
- Actor net: FFNN with 2 hidden layers with 64 units.
- Critic net: FFNN with 2 hidden layers with 64 units.
- Clip gradient norm: 0.5

MyAcroBot class is just a env wrapper for the Acrobot env: the last two observation features are normalized (min-max normalization).

Policy evaluated by running 16 episodes (agent evaluated at each training steps). Best result, mean_cumulative_reward (episodic): $-75.8125\pm4.3387$
