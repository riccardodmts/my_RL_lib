from learner.learner import SynchLearner
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np


"""
PPODataset is a class used for preparing data to be used for training both actor and critic net.
Given several trajectories, this class post-processes them by computing both the advantage and the 
value function target. In particular given B trajectories, for each sample in one of the trajectories,
the tuple (o_t, a_t, log p(a_t|o_t), A_t, V_t), where t denotes the time instant, A_t the advantage at 
instant t, V_t the target for the critic, is obtained. 
This class is a custom torch Dataset class. This means that a Dataloader can be used for creating
mini-batches randomly sampled.
Since this class supports the dict observation space, o_t is a dict in that case. By using a Dataloader,
if o_t has to be a dict, for each key a tensor [B x ...] is returned by the dataloader.
"""
class PPODataset(Dataset):

    def __init__(self, batch, gamma, lamb, device="cpu"):
        """
        :param batch:   dictionary with tensors for obs, action, ... of type [B x T x ...] where
                        B = # trajectories
        :param gamma:   discout factor
        :param lamb:    lambda param used for computing the GAE
        :param device:
        """

        self.lamb = lamb
        self.gamma = gamma
        self.device = device

        # convert data in tensors
        vf = torch.tensor(batch.pop("vf"), device=device).float()
        rewards = torch.tensor(batch.pop("rewards"), device=device).float()
        dones = torch.tensor(batch.pop("dones"), device=device).float()
        self.actions = torch.tensor(batch.pop("actions"), device=device).float()
        self.logp = torch.tensor(batch.pop("logp"), device=device).float()
        self.obs, self.obs_is_dict = self._retrieve_observations(batch)

        # compute advantage for each sample
        self.advantages_un = self._compute_advantage(rewards, vf, dones)
        self.value_target_un = self.advantages_un + vf[:, :-1, :]
        # normalize advantage and vf target (over entire batch)
        self.advantages = (self.advantages_un - torch.mean(self.advantages_un))/(torch.std(self.advantages_un))
        self.value_target = (self.value_target_un - torch.mean(self.value_target_un))/(torch.std(self.value_target_un))

    def _compute_advantage(self, rewards, vf, dones):
        """
        Compute the advantage using the formula in https://arxiv.org/abs/1506.02438
        :param rewards: tensor [B x T x 1] with the rewards collected
        :param vf: tensor [B x T+1 x 1] with the value function estimations
        :param dones: tensor [B x T x 1] with either zeros and ones (one = end episode)
        :return: tensor [B x T x 1] with the GAE for each sample
        """

        # delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
        vf_next = vf[:, 1:, :]  # V(s_{t+1}): [B x T x 1]
        discounts = self.gamma * (1 - dones)  # [B x T x 1]
        deltas = rewards + discounts * vf_next - vf[:, :-1, :]  # [B x T x 1]

        # GAE computation
        advantages = [torch.zeros_like(vf_next[:, 0, :])]  # list with tensors [B x 1]

        for i in reversed(range(vf_next.shape[1])):
            discount_t, delta_t = discounts[:, i, :], deltas[:, i, :]  # [B x 1]
            advantages.append(delta_t + self.lamb * discount_t * advantages[-1])

        # advantages is a list with T+1 (first one "fake") tensors [B x 1] in a reverse order (t=T, t=T-1, ..., t=1)
        advantages = torch.stack(advantages[1:])  # [T x B x 1] in reverse order
        advantages = torch.flip(advantages, dims=[0])  # [T x B x 1] correct order
        advantages = torch.transpose(advantages, 0, 1)  # [B x T x 1]

        return advantages

    def _retrieve_observations(self, batch):
        """
        Given a batch it returns just the observations. In case of a dict space it returns a dict
        with each single observation space with the key used by the environment.
        Gym space = {"first_space" : ..., "second_space" : ....} => obs = {"first_space" :
        tensor, "second_space" : tensor} where the tensors have shape = [B x T x ...].
        Otherwise obs is a tensor [B x T x ...]
        :param batch: dict with all the tensors collected during sampling (obs, actions, rewards, ...)
        :return: either dict with tensors or a tensor (see above)
        """

        is_dict = False
        for key in batch:
            if key == "obs":
                obs = torch.tensor(batch[key][:, :-1], device=self.device).float()
                break
            elif "obs_" in key:
                if not is_dict:
                    obs = {}
                is_dict = True
                new_key = key.split("_")[1]
                obs[new_key] = torch.tensor(batch[key][:, :-1], device=self.device).float()

        return obs, is_dict

    def __len__(self):
        return self.advantages.shape[0]*self.advantages.shape[1]

    def __getitem__(self, idx):
        """
        Return (o, a, log pi(a|o), A, V) for idx ranging from 0 to B * T - 1, where B=# of trajectories
        :param idx: integer index in the interval [0, B * T - 1]
        :return: tuple: o, a, log pi(a|o), A, V. o can be a dict
        """

        if idx >= self.__len__():
            raise IndexError(f"Out of range: index given is {idx}, but len of dataset is {self.__len__()}")
        if idx < 0:
            raise IndexError("Out of range: negative index!")

        trajectory = idx//(self.advantages.shape[1])
        time_instant = idx % (self.advantages.shape[1])

        if self.obs_is_dict:
            obs = {}
            for key in self.obs:
                obs[key] = self.obs[key][trajectory, time_instant]
        else:
            obs = self.obs[trajectory, time_instant]

        action = self.actions[trajectory, time_instant]
        logp = self.logp[trajectory, time_instant]
        advantage = self.advantages[trajectory, time_instant]
        value_target = self.value_target[trajectory, time_instant]

        return {"obs": obs, "action": action, "logp": logp, "advantage": advantage, "value_target": value_target}


if __name__ == "__main__":
    # Test PPODataset
    actions = np.zeros((5, 10, 1), dtype=np.float32)
    logp = np.zeros((5, 10, 1), dtype=np.float32)
    vf = np.zeros((5, 11, 1), dtype=np.float32)
    dones = np.zeros((5, 10, 1), dtype=np.float32)
    dones[0, 5, 0] = 1
    dones[1, 5, 0] = 1
    vf[0, :, 0] = 0.5
    dones[0, 5, 0] = 1
    vf[0, 6, 0] = 0.7
    rewards = np.zeros((5, 10, 1), dtype=np.float32)
    rewards[:2,:,:] = 1
    obs = np.zeros((5, 11, 5), dtype=np.float32)
    obs[0, :, 0] = np.array([1,2,3,4,5,6,7,8,9,10,11], dtype=np.float32)
    obs[1, :, :] = 1

    batch2 = {"actions": actions, "logp": logp, "vf": vf, "dones": dones, "rewards": rewards, "obs_prima": obs, "obs_seconda": np.ones((5, 11, 1), dtype=np.float32)}

    dataset = PPODataset(batch2, 0.99, 0.95)
    print(dataset.advantages)
    print(len(dataset))
    print(dataset[4])

    train_dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    mini_batch = next(iter(train_dataloader))
    print(mini_batch["obs"])
    print(mini_batch["advantage"])




