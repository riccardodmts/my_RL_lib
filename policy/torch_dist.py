from gymnasium import spaces
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal


def get_dist_from_action_space(action_space: spaces.Space):
    """
    This function returns the right torch distribution class based on the actions space
    :param action_space: action_space obj
    :return:
    """

    if isinstance(action_space, spaces.Discrete):
        return Categorical

    elif isinstance(action_space, spaces.MultiDiscrete):

        raise Exception("Not implemented!!!")

    elif isinstance(action_space, spaces.Box):

        return Normal

# TODO: Add wrappers for torch distributions
