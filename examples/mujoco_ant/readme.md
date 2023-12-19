# MuJoCo Ant-v4, results with PPO

PPO params used:

- $\gamma=0.99$.
- Number of actors: 4.
- Training steps: 100.
- Trajectory length: $T=1024$.
- $\lambda=0.95$.
- PPO clipping param: $\epsilon=0.2$.
- Optimization:
  1. Number of epochs: 10.
  2. Mini-batch size: 64.
  3. Optimizer: Adam with $\beta_1=0.9, \beta_2=0.99$.
  4. Learning rate: 0.0003.
- $c1$ (coefficient for state-value function loss): 1.0.
- No entropy penalty ($c_2=0$).
- Actor net: FFNN with 2 hidden layers with 128 units.
- Critic net: FFNN with 2 hidden layers with 128 units.
- Clip gradient norm: 0.5
- log_std learnable, but not state dependent. Intial value: log_std=-1

Policy evaluated by running 21 episodes (agent evaluated at each training steps):
![Ant-v4](https://github.com/riccardodmts/my_RL_lib/assets/83876494/74de809d-7e0a-460e-aac7-3f6dde0bb7e0)
