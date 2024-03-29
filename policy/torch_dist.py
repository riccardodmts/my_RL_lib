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

    def __init__(self, mean, sigma_2=0.05):

        B, dim = mean.shape[0], mean.shape[1]
        covariance_matrix = sigma_2 * torch.eye(dim).reshape((1, dim, dim)).repeat((B, 1, 1))

        self.inner_dist = MultivariateNormal(mean, covariance_matrix)

    def sample(self):

        return self.inner_dist.sample()

    def log_prob(self, value):
        if len(value.shape) == 1:
            value = torch.unsqueeze(value, dim=1)
        return self.inner_dist.log_prob(value)

    def entropy(self):

        return self.inner_dist.entropy()


class TorchNormalV2:

    def __init__(self, model_output):

        mean = model_output[0]
        log_std = model_output[1]

        B, dim = mean.shape[0], mean.shape[1]

        if isinstance(log_std, torch.Tensor):
            if not log_std.shape[0] == B:
                # in case log_std has dim [1 x action space]
                log_std = log_std.expand_as(mean)
        else:
            # in case std is a scalar
            log_std = torch.ones((B, dim)).float() * log_std

        std = torch.exp(log_std)  # [B x action space] std for actions
        self.inner_dist = Normal(mean, std)

    def sample(self):

        return self.inner_dist.sample()

    def log_prob(self, value):

        if len(value.shape) == 1:
            value = torch.unsqueeze(value, dim=1)

        return self.inner_dist.log_prob(value).sum(1)

    def entropy(self):

        return self.inner_dist.entropy().sum(1)
