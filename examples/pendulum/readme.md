# Gymansium Pendulum-v1, results with PPO

PPO params used:

- $\gamma=0.9$.
- Number of actors: 32.
- Training steps: 100.
- Trajectory length: $T=256$.
- $\lambda=0.95$.
- PPO clipping param: $\epsilon=0.2$.
- Optimization:
  1. Number of epochs: 10.
  2. Mini-batch size: 64.
  3. Optimizer: Adam with $\beta_1=0.9, \beta_2=0.99$.
  4. Learning rate: 0.0003.
- $c1$ (coefficient for state-value function loss): 0.5.
- No entropy penalty ($c_2=0$).
- Actor net: FFNN with 2 hidden layers with 64 units.
- Critic net: FFNN with 2 hidden layers with 64 units.
- Clip gradient norm: 0.5
- $\sigma^2$ for sampling action (gaussian variance): fixed, $0.05$.

Policy evaluated by running 16 episodes (agent evaluated at each training steps). Best result, mean_cumulative_reward (episodic): $-153.334\pm32.838$
