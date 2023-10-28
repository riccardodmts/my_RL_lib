from gymnasium import spaces
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
import torch


def get_dist_from_action_space(action_space: spaces.Space):
    """
    This function returns the right torch distribution class based on the actions space
    :param action_space: action_space obj
    :return:
    """

    if isinstance(action_space, spaces.Discrete):
        return TorchCategorical

    elif isinstance(action_space, spaces.MultiDiscrete):

        raise Exception("Not implemented!!!")

    elif isinstance(action_space, spaces.Box):

        return TorchNormal


class TorchCategorical:
    """
    Torch distribution wrapper for categorical distribution
    """

    def __init__(self, logits):

        self.inner_dist = Categorical(logits=logits)

    def sample(self):

        return self.inner_dist.sample()

    def log_prob(self, value):

        return self.inner_dist.log_prob(value)

    def entropy(self):

        return self.inner_dist.entropy()


class TorchNormal:
    """
    Torch distribution wrapper for multimodal gaussian with fixed diagonal covariance matrix
    """

    def __init__(self, mean, sigma=0.01):

        dim = mean.shape[0]
        covariance_matrix = sigma * torch.eye(dim)

        self.inner_dist = MultivariateNormal(mean, covariance_matrix)

    def sample(self):

        return self.inner_dist.sample()

    def log_prob(self, value):

        return self.inner_dist.log_prob(value)

    def entropy(self):

        return self.inner_dist.entropy()

