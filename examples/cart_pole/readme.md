# Gymansium CartPole-v1, results with PPO

PPO params used:

- $\gamma=0.99$.
- Number of actors: 10.
- Training steps: 30.
- Trajectory length: $T=500$.
- $\lambda=0.95$.
- PPO clipping param: $\epsilon=0.2$.
- Optimization:
  1. Number of epochs: 5.
  2. Mini-batch size: 256 (batch size = 5000).
  3. Optimizer: Adam with $\beta_1=0.9, \beta_2=0.99$.
  4. Learning rate: 0.0003.
- $c1$ (coefficient for state-value function loss): 1.0.
- No entropy penalty ($c_2=0$).
- Actor net: FFNN with 2 hidden layers with 64 units.
- Critic net: FFNN with 2 hidden layers with 64 units.
- Clip gardient norm: 0.5

Policy evaluated by running 16 episodes (agent evaluated at each training steps). Best result, mean_cumulative_reward (episodic): $500\pm0$
