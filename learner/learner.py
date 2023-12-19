import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch


class Learner:

    def __init__(self, gamma, policy_cls, policy_dict, device="cpu"):

        self.device = device
        self.policy = None
        self.gamma = gamma

        self._init_policy(policy_cls, policy_dict)

    def _init_policy(self, policy_cls, policy_dict):
        """
        Initialize policy object (used by the learner for the loss computation).
        :param policy_cls:
        :param policy_dict:
        :return:
        """

        model_info = policy_dict["model_info"]
        model_config = policy_dict["model_config"]
        policy_config = policy_dict["policy_config"]
        dist_info = policy_dict.get("dist_info", None)

        self.policy = policy_cls(model_info, model_config, policy_config, dist_info, self.device)

    def get_model_dict(self):

        return self.policy.get_model_dict()

    def set_model_dict(self, model_dict):

        self.policy.set_model_dict(model_dict)


class SynchLearner(Learner):

    def __init__(self, gamma, policy_cls, policy_dict, device="cpu"):

        super().__init__(gamma, policy_cls, policy_dict, device)

        # default params
        self.lr = 0.001
        self.adam_betas = (0.9, 0.99)
        self.epochs = 1
        self.minibatch_size = -1  # -1 = do not split batch in mini-batches
        self.optimizers = []
        self.set_to_train = False
        self.max_grad_norm = 0.5

    def loss(self, batch):

        pass

    def set_for_training(self):
        """
        Prepare the learner for the training, namely set the optimizers
        :return:
        """
        # set model to train mode
        self.policy.model.train()

        if len(self.optimizers) == 0:

            net_params = self.policy.get_model_parameters()
            self.optimizers.append(optim.Adam(net_params, self.lr, self.adam_betas, eps=1e-7))

        self.set_to_train = True

    def set_training_params(self,
                            max_grad_norm = None
                            ):
        if max_grad_norm is not None:
            self.max_grad_norm = max_grad_norm

    def _create_torch_dataset(self, batch):

        pass

    def _to_device(self, batch):
        """
        Move tensors to the device used for training
        :param batch: batch of data to move to device
        :return:
        """
        for key, item in batch.items():
            if isinstance(item, dict):
                for k, el in item.items():
                    el.to(self.device)
            else:
                item.to(self.device)

    def _compute_training_stats(self):
        """
        To be defined by subclass
        :return:
        """
        pass

    def _save_training_stats(self):
        """
        :return:
        """
        pass

    def _get_recent_training_stats(self):
        """
        :return:
        """
        pass

    def train(self, batch):
        """
        Perform a policy update following the specific features of any algorithm (multiple SGD iters
        , use of mini-batches, ...)
        :param batch: batch with samples
        :return: updated weights of the models (policy, value function,...)
        """

        if not self.set_to_train:
            self.set_for_training()

        # Prepare dataset
        dataset = self._create_torch_dataset(batch)
        batch_size = len(dataset)
        minibatch_size = batch_size if self.minibatch_size == -1 else self.minibatch_size
        dataloader = DataLoader(dataset, batch_size=minibatch_size, shuffle=True)

        for epoch in range(self.epochs):

            for i, minibatch in enumerate(dataloader):
                # move to device (either cuda or cpu)
                self._to_device(minibatch)
                # zero grad all optimizers
                for opt in self.optimizers:
                    opt.zero_grad()
                # compute loss and gradient
                loss = self.loss(minibatch)
                self._save_training_stats()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.get_model_parameters(), self.max_grad_norm)

                # update weights
                for opt in self.optimizers:
                    opt.step()

            # compute average losses for this epoch
            self._compute_training_stats()

        # get average losses for the last epoch
        training_metrics = self._get_recent_training_stats()
        return self.policy.get_model_dict(), training_metrics

    def _detach_obs_dict(self, obs_dict):
        """
        Detach each tensor in obs dict
        :param obs_dict: observation dictionary
        :return:
        """
        for key in obs_dict:
            obs_dict[key] = obs_dict[key].detach()
