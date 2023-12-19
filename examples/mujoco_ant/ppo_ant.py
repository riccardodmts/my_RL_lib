from sampler.sampler import SequentialSampler
from PPO.ppo_learner import PPOLearner
from PPO.ppo_policy import PPOPolicy
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from policy.torch_dist import TorchNormalV2


def plot_eval_results(scores_list):

    means = scores_list[0]
    confs = scores_list[1]

    fig, ax = plt.subplots()
    y = np.array(means)
    ax.plot(y)
    ax.fill_between(range(len(y)), y-np.array(confs), y+np.array(confs), alpha=.1)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Score")
    plt.savefig("Ant-v4.jpg")

if __name__ == "__main__":
    model_dict = {

            "input_dim": 27,
            "num_actions": 8,
            "actor": {
                "first_hidden": 128,
                "second_hidden": 128
            },

            "critic": {

                "first_hidden": 128

            },
            "initial_std_const": -1.0
        }
    policy_dict = {
        "env_info": {
            "action_space": gym.spaces.Discrete(4)
        },
        "sampling": {
            "info_to_sample": ["vf", "logp"]
        }
    }
    policy_d = {"model_info": "mujoco_model", "model_config": model_dict, "policy_config": policy_dict, "dist_info": TorchNormalV2}
    env_str = "Ant-v4"
    env_info = {}

    sampler = SequentialSampler(4, env_str, env_info, PPOPolicy, policy_d, 1024)
    learner = PPOLearner(0.99, policy_d)
    learner.set_training_params(lamb=0.95, minibatch_size=64, epochs=10, lr=0.0003, vf_loss_coeff=1.0, max_grad_norm=0.5,
                                clip_param=0.2)
    initial_weights = learner.get_model_dict()
    sampler.synch_weights(initial_weights)

    # store evaluation scores (mean, conf)
    eval_scores = ([], [])
    # save best model
    best_weights = None
    best_mean = None
    steps = 100

    for i in range(steps):
        batch, _ = sampler.sample_batch(stack_trajectories=True)

        weights, metrics = learner.train(batch)
        print(metrics)
        sampler.synch_weights(weights)
        scores = []
        for j in range(21):
            batch = sampler.evaluate_policy(1, weights, 1000)
            done_at = np.where(np.squeeze(batch["dones"], axis=2)>=1.0)[1][0]
            rewards = np.squeeze(batch["rewards"], axis=2)
            rewards[:,done_at+1:] = 0.0
            score = np.sum(rewards)
            scores.append(score)
        mean = np.mean(scores)
        conf = np.std(scores)*1.96/np.sqrt(21)
        print(f"Iter: {i}, {mean}, {conf}")

        eval_scores[0].append(mean)
        eval_scores[1].append(conf)

    plot_eval_results(eval_scores)
    print("finished")
