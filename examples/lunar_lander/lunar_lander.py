import gymnasium as gym

from sampler.sampler import SequentialSampler
from PPO.ppo_learner import PPOLearner
from PPO.ppo_policy import PPOPolicy
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

def plot_eval_results(scores_list):

    means = scores_list[0]
    confs = scores_list[1]

    fig, ax = plt.subplots()
    y = np.array(means)
    ax.plot(y)
    ax.fill_between(range(len(y)), y-np.array(confs), y+np.array(confs), alpha=.1)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Score")
    plt.savefig("lunar_lander_eval.jpg")

if __name__ == "__main__":
    model_dict = {

            "input_dim": 8,
            "num_actions": 4,
            "actor": {
                "first_hidden": 64,
                "second_hidden": 64
            },

            "critic": {

                "first_hidden": 64

            }
        }
    policy_dict = {
        "env_info": {
            "action_space": gym.spaces.Discrete(4)
        },
        "sampling": {
            "info_to_sample": ["vf", "logp"]
        }
    }
    policy_d = {"model_info": "example", "model_config": model_dict, "policy_config": policy_dict}
    env_str = "LunarLander-v2"
    env_info = {}

    sampler = SequentialSampler(16, env_str, env_info, PPOPolicy, policy_d, 2048)
    learner = PPOLearner(0.99, policy_d)
    learner.set_training_params(lamb=0.98, minibatch_size=64, epochs=4, lr=0.0003, vf_loss_coeff=0.5, entropy_coeff=0.01)
    initial_weights = learner.get_model_dict()
    sampler.synch_weights(initial_weights)

    # store evaluation scores (mean, conf)
    eval_scores = ([], [])
    # save best model
    best_weights = None
    best_mean = None
    steps = 40

    for i in range(steps):
        batch, _ = sampler.sample_batch(stack_trajectories=True)

        weights, metrics = learner.train(batch)
        print(metrics)
        sampler.synch_weights(weights)
        scores = []
        for j in range(16):
            batch = sampler.evaluate_policy(1, weights, 1000)
            done_at = np.where(np.squeeze(batch["dones"], axis=2)>=1.0)[1][0]
            #print(done_at)
            rewards = np.squeeze(batch["rewards"], axis=2)
            rewards[:,done_at+1:] = 0.0
            print(f"Last reward: {rewards[:, done_at]}")
            score = np.sum(rewards)
            scores.append(score)
        mean = np.mean(scores)
        conf = np.std(scores)*1.96/np.sqrt(16)
        print(f"{mean}, {conf}")

        eval_scores[0].append(mean)
        eval_scores[1].append(conf)

    plot_eval_results(eval_scores)
    print("finished")

    # best: -75.8125, +-4.3387
    # clip grad 0.5
