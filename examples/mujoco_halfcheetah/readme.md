# MuJoCo HalfCheetah-v4, results with PPO

PPO params used:

- $\gamma=0.99$.
- Number of actors: 4.
- Training steps: 250.
- Trajectory length: $T=1024$.
- $\lambda=0.95$.
- PPO clipping param: $\epsilon=0.15$.
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
- log_std learnable, but not state dependent. Initial value: log_std=-2

Policy evaluated by running 21 episodes (agent evaluated at each training steps).
![HalfCheetah-v4](https://github.com/riccardodmts/my_RL_lib/assets/83876494/1b13f3b1-7f6c-4b48-bc5b-d89a31aaedd0)
