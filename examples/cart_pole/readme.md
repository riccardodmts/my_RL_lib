# Gymansium Cart-Polev1 results with PPO

PPO params used:

- $\gamma=0.99$.
- Number of actors: 10.
- Training steps: 50.
- Trajectory length: $T=500$.
- $\lambda=0.995$.
- PPO clipping param: $\epsilon=0.2$.
- Optimization:
  1. Number of epochs: 5.
  2. Mini-batch size: 500 (batch size = 5000).
  3. Optimizer: Adam with $\beta_1=0.9,\;\beta_2=0.99$.
  4. Learning rate: 0.003.
- $c1$ (coefficient for state-value function loss): 1.0.
- No entropy penalty ($c_2=0$).
- Actor net: FFNN with 2 hidden layers with 32 units.
- Critic net: FFNN with one hidden layer with 32 units.

Policy evaluated by running 10 episodes (done it during training, every 5 training steps):
