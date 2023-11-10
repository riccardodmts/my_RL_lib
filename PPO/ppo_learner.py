from learner.learner import SynchLearner
from PPO.ppo_dataset import PPODataset
from policy.policy import Policy
from PPO.ppo_policy import PPOPolicy
import gymnasium as gym
import numpy as np
import torch


class PPOLearner(SynchLearner):

    def __init__(self, gamma, policy_dict, policy_cls=PPOPolicy, device="cpu"):

        super().__init__(gamma, policy_cls, policy_dict, device)

        # default PPO specific training params
        self.gamma = gamma
        self.device = device
        self.lamb = 0.995
        self.clip_param = 0.2
        self.vf_loss_coeff = 1.0
        self.entropy_coeff = 0.0

        # verify if entropy needed
        if policy_dict.get("training", False):
            if policy_dict["training"].get("entropy_coeff", False):
                self.entropy_coeff = policy_dict["training"]["entropy_coeff"]
                if self.entropy_coeff > 0:
                    self.policy.collect_entropy()

        # last losses (most recent loss computation)
        self.pi_loss = 0
        self.vf_loss = 0
        self.entropy = 0
        self.total_loss = 0

        # store metrics along one single epoch (the current one)
        self.training_metrics_per_epoch = {
            "pi_loss": [],
            "vf_loss": [],
            "entropy": [],
            "total_loss": []
        }

        # store metrics per each epoch
        self.training_metrics = {
            "pi_loss": [],
            "vf_loss": [],
            "entropy": [],
            "total_loss": []
        }

    def set_training_params(self,
                            lamb=None,
                            clip_param=None,
                            lr=None,
                            minibatch_size=None,
                            epochs=None,
                            vf_loss_coeff=None,
                            entropy_coeff=None
                            ):
        """
        Set PPO parameters for training
        :param lamb: lambda use for GAE
        :param clip_param: clipping param for clipped loss
        :param lr: learning rate
        :param minibatch_size: mini-batches size used during SGD-based training
        :param epochs: # SGD iterations
        :param vf_loss_coeff: coefficient for the state-value function loss
        :param entropy_coeff: coefficient for the entropy loss. if None or <= 0, not used
        :return:
        """

        if lamb is not None:
            self.lamb = lamb
        if clip_param is not None:
            self.clip_param = clip_param
        if lr is not None:
            self.lr = lr
        if minibatch_size is not None:
            self.minibatch_size = minibatch_size
        if epochs is not None:
            self.epochs = epochs
        if vf_loss_coeff is not None:
            self.vf_loss_coeff = vf_loss_coeff
        if entropy_coeff is not None:
            self.entropy_coeff = entropy_coeff
            if self.entropy_coeff > 0.0:
                self.policy.collect_entropy()

    def _create_torch_dataset(self, batch):
        """
        Given a batch of trajectories, this function returns d PPODataset
        :param batch: dict with tensors [B x T ...] per obs, action, ...
        :return: PPODataset
        """
        return PPODataset(batch, self.gamma, self.lamb, "cpu")

    def loss(self, batch):
        """
        Compute PPO loss: for actor - clipped surrogate objective, for critic squared loss. If needed
                          also the entropy term is added.
        :param batch: mini-batch (or batch) of data
        :return: loss (torch scalar)
        """
        if isinstance(batch["obs"], dict):
            self._detach_obs_dict(batch["obs"])
        else:
            batch["obs"] = batch["obs"].detach()
        logp_theta, vf, entropy = self.policy.training_forward(batch["obs"], batch["action"].detach())
        logp_theta_old = torch.squeeze(batch["logp"].detach(), dim=1)
        advantage = torch.squeeze(batch["advantage"].detach(), dim=1)
        value_target = torch.squeeze(batch["value_target"].detach(), dim=1)

        """Surrogate objective for actor"""
        # log p ratio
        likelihood = torch.exp(logp_theta - logp_theta_old)

        # clipped surrogate objective
        surrogate_objective = torch.mean(
            torch.min(
                advantage * likelihood,
                advantage * torch.clamp(likelihood, min=1-self.clip_param, max=1+self.clip_param)
            )
        )
        self.pi_loss = -surrogate_objective.item()

        # vf squared loss
        vf_loss = torch.mean(torch.pow(value_target - vf, 2))
        self.vf_loss = vf_loss.item()

        # overall loss
        loss = self.vf_loss_coeff * vf_loss - surrogate_objective

        # add entropy penalty term if needed
        if self.entropy_coeff > 0.0:
            entropy_loss = torch.mean(entropy)
            loss += self.entropy_coeff * entropy_loss
            self.entropy = entropy_loss.item()

        self.total_loss = loss.item()
        return loss

    def _save_training_stats(self):
        """
        Save training losses for the last batch/mini-batch
        :return: None
        """

        self.training_metrics["pi_loss"].append(self.pi_loss)
        self.training_metrics["vf_loss"].append(self.vf_loss)
        self.training_metrics["entropy"].append(self.entropy)
        self.training_metrics["total_loss"].append(self.total_loss)

    def _compute_training_stats(self):
        """
        Compute average losses for the last epoch just performed
        :return: None
        """

        for key in self.training_metrics_per_epoch:
            # compute average loss along
            self.training_metrics_per_epoch[key].append(np.mean(self.training_metrics[key]))

        # reset loss buffer (ready to store losses for next epoch)
        for key in self.training_metrics:
            self.training_metrics[key] = []

    def _get_recent_training_stats(self):
        """
        Get losses for last training epoch for the current batch
        :return: dict with losses (vf loss, policy loss, entropy and total loss)
        """
        results = {key: self.training_metrics_per_epoch[key][-1] for key in self.training_metrics_per_epoch}
        for key in self.training_metrics_per_epoch:
            self.training_metrics_per_epoch[key] = []

        return results


if __name__ == "__main__":
    # Test PPODataset and PPOLearner
    """
    model_dict = {

        "input_dim": 1,
        "num_actions": 2,

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
            "action_space": gym.spaces.Discrete(2)
        },
        "sampling": {
            "info_to_sample": ["vf", "logp"]
        }
    }

    policy_d = {"model_info": "example", "model_config": model_dict, "policy_config": policy_dict}

    actions = np.zeros((1, 1, 1), dtype=np.float32)
    logp = -0.1*np.ones((1, 1, 1), dtype=np.float32)
    vf = np.zeros((1, 2, 1), dtype=np.float32)
    dones = np.zeros((1, 1, 1), dtype=np.float32)
    rewards = np.zeros((1, 1, 1), dtype=np.float32)
    obs = 1*np.ones((1, 2, 1), dtype=np.float32)
    rewards[0,0,0] = 1

    batch2 = {"actions": actions, "logp": logp, "vf": vf, "dones": dones, "rewards": rewards, "obs_prima": obs, "obs_seconda": np.ones((1, 2, 1), dtype=np.float32)}

    ppo_learner = PPOLearner(0.99, PPOPolicy, policy_d)

    print(ppo_learner.train(batch2)[1])
    """