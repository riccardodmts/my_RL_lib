"""
Gymnasium Cart-Pole env solved by use of PPO
"""

from sampler.sampler import SequentialSampler
from PPO.ppo_learner import PPOLearner
from PPO.ppo_policy import PPOPolicy
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt


def compute_score(data):

    dones = data["dones"]  # B x T x 1
    dones = np.squeeze(dones, axis=2)
    first_done = np.argmax(dones == 1, axis=1) + 1
    mean = np.mean(first_done)
    max_ = np.max(first_done)
    min_ = np.min(first_done)
    std = np.std(first_done)
    conf = 1.96 / np.sqrt(len(first_done))*std

    return f"mean score: {mean}, max score: {max_}, min score: {min_}, conf: {conf}", (mean, max_, min_, conf)

def plot_eval_results(scores_list):

    means = []
    confs = []

    for item in scores_list:
        means.append(item[0])
        confs.append(item[3])

    fig, ax = plt.subplots()
    y = np.array(means)
    ax.plot(y)
    ax.fill_between(range(len(y)), y-np.array(confs), y+np.array(confs), alpha=.1)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Score")
    plt.savefig("ppo_cart_pole_eval.jpg")


if __name__ == "__main__":

    # MODEL DICT: see LinearPPOModel for the FF architecture
    model_dict = {

            "input_dim": 4,
            "num_actions": 2,

            "actor": {
                "first_hidden": 64,
                "second_hidden": 64
            },

            "critic": {

                "first_hidden": 64

            }
        }

    # info for the sampling
    policy_dict = {
        "env_info": {
            "action_space": gym.spaces.Discrete(4)
        },
        "sampling": {
            "info_to_sample": ["vf", "logp"]
        }
    }

    # stack model info and sampling info
    policy_d = {"model_info": "example", "model_config": model_dict, "policy_config": policy_dict}

    # env to test
    env_str = "CartPole-v1"
    env_info = {}

    # create a sampler: pass env and sampling info: env, ..., Policy class used
    # for sampling, model and sampling info, T (trajectory length)
    sampler = SequentialSampler(10, env_str, env_info, PPOPolicy, policy_d, 500)

    # create a learner with gamma and model info
    learner = PPOLearner(0.99, policy_d)
    # set params for training
    learner.set_training_params(lamb=0.95, minibatch_size=256, lr=0.0003, epochs=5)

    # synch weights among all the sampling actors
    initial_weights = learner.get_model_dict()
    sampler.synch_weights(initial_weights)

    # store evaluation scores (mean, max and min)
    eval_scores = []
    # save best model
    best_weights = None
    best_mean = None

    # number of training steps
    steps = 30
    for i in range(steps):
        # sample N trajectories
        batch, _ = sampler.sample_batch(stack_trajectories=True)

        # learn on batch
        weights, metrics = learner.train(batch)
        # update weights for sampling actors
        sampler.synch_weights(weights)

        # every 10 training steps evaluate policy
        batch = sampler.evaluate_policy(16, weights)
        string_to_print, score = compute_score(batch)
        print(f"Training step: {i}, "+string_to_print)
        if best_mean is None:
            best_mean = score[0]
            best_weights = weights
        elif score[0] > best_mean:
            best_mean = score[0]
            best_weights = weights
        eval_scores.append(score)

    plot_eval_results(eval_scores)

    print("finished")