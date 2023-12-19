from sampler.sampler import SequentialSampler
from PPO.ppo_learner import PPOLearner
from PPO.ppo_policy import PPOPolicy
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

def plot_eval_results(scores_list):

    fig, ax = plt.subplots()
    y = np.array(scores_list[0])
    ax.plot(y)
    ax.fill_between(range(len(y)), y - np.array(scores_list[1]), y + np.array(scores_list[1]), alpha=.1)
    ax.set_xlabel("Training steps ( * 5)")
    ax.set_ylabel("Score")
    plt.savefig("ppo_pendulum_eval.jpg")
    #plt.show()

if __name__ == "__main__":

    # MODEL DICT: see LinearPPOModelContinuous for the FF architecture
    model_dict = {

            "input_dim": 3,
            "num_actions": 1,
            "actor": {
                "first_hidden": 64,
                "second_hidden": 64
            },

            "critic": {

                "first_hidden": 64

            },

            "output_mul": [2.0]
        }

    # info for the sampling
    policy_dict = {
        "env_info": {
            "action_space": gym.spaces.Box(-1, 1, (1,))
        },
        "sampling": {
            "info_to_sample": ["vf", "logp"]
        }
    }

    # stack model info and sampling info
    policy_d = {"model_info": "cont", "model_config": model_dict, "policy_config": policy_dict}
    # env to test
    env_str = "Pendulum-v1"
    env_info = {}

    # create a sampler: pass env and sampling info: env, ..., Policy class used
    # for sampling, model and sampling info, T (trajectory length)
    sampler = SequentialSampler(32, env_str, env_info, PPOPolicy, policy_d, 256)

    # create a learner with gamma and model info
    learner = PPOLearner(0.9, policy_d)
    # set params for training
    learner.set_training_params(lamb=0.95, vf_loss_coeff=0.5, minibatch_size=64, epochs=10, lr=0.0003)
    # 8 , 2048, 32, relu around 300 lamb)0.95, gamma 0.99, sigma2=0.1, lr 0.0003, epochs 10, minibatch 32
    # synch weights among all the sampling actors
    initial_weights = learner.get_model_dict()
    sampler.synch_weights(initial_weights)

    # store evaluation scores (mean, max and min)
    eval_scores = ([], [])
    # save best model
    best_weights = None
    best_mean = None
    steps = 100

    for i in range(steps):
        # sample N trajectories
        batch, _ = sampler.sample_batch(stack_trajectories=True)

        # learn on batch
        weights, metrics = learner.train(batch)
        print(metrics)
        # synch new weights
        sampler.synch_weights(weights)

        if (i % 1 == 0) or i == 0:

            scores = []
            for j in range(16):
                batch = sampler.evaluate_policy(1, weights, 200)
                done_at = np.where(np.squeeze(batch["dones"], axis=2) >= 1.0)[1][0]
                rewards = np.squeeze(batch["rewards"], axis=2)
                rewards[0, done_at+1:] = 0.0
                scores.append(np.sum(rewards))

            mean = np.mean(scores)
            conf = np.std(scores) * 1.96 / np.sqrt(16)
            print(f"Training step: {i}, {mean}, {conf}")
            eval_scores[0].append(mean)
            eval_scores[1].append(conf)

    plot_eval_results(eval_scores)
    print("finished")