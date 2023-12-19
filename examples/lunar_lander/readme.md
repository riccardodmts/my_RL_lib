# Gymansium LunarLander-v2, results with PPO

PPO params used:

- $\gamma=0.99$.
- Number of actors: 16.
- Training steps: 40.
- Trajectory length: $T=2048$.
- $\lambda=0.98$.
- PPO clipping param: $\epsilon=0.2$.
- Optimization:
  1. Number of epochs: 4.
  2. Mini-batch size: 64.
  3. Optimizer: Adam with $\beta_1=0.9, \beta_2=0.99$.
  4. Learning rate: 0.0003.
- $c1$ (coefficient for state-value function loss): 0.5.
- Entropy penalty, $c_2=0.01$.
- Actor net: FFNN with 2 hidden layers with 64 units.
- Critic net: FFNN with 2 hidden layers with 64 units.
- Clip gardient norm: 0.5

Policy evaluated by running 16 episodes (agent evaluated at each training steps). Best result, mean_cumulative_reward (episodic): $274.58\pm10.04$

![lunar_lander_eval](https://github.com/riccardodmts/my_RL_lib/assets/83876494/376dab50-7b61-440f-9595-98fccd5a0e5a)
