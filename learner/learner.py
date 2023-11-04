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

    def loss(self, batch):

        pass

    def set_for_training(self):
        """
        Prepare the learner for the training, namely set the optimizers
        :return:
        """

        if len(self.optimizers) == 0:

            net_params = self.policy.get_weights()
            self.optimizers.append(optim.Adam(net_params, self.lr, self.adam_betas))

        self.set_to_train = True

    def _create_torch_dataset(self, batch):

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
        dataloader = DataLoader(dataset, minibatch_size, shuffle=True)

        for epoch in range(self.epochs):

            for i, minibatch in enumerate(dataloader):
                # zero grad all optimizers
                for opt in self.optimizers:
                    opt.zero_grad()
                # compute loss and gradient
                loss = self.loss(minibatch)
                loss.backward()

                # update weights
                for opt in self.optimizers:
                    opt.step()

        return self.policy.get_weights()
