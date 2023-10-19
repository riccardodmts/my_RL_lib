"""
Base class for any Policy: a Policy is used for sampling. It is an interface to the model
and the distribution from which an action is sampled
"""
import torch
import inspect

from utils import model_catalog
from .torch_model import TorchModel
from . import torch_dist



class Policy:

    def __init__(self, model_info, model_config, policy_config, dist_info=None,  device="cpu"):

        self.conf = policy_config
        self.device = torch.device(device)
        self.model = None
        self.dist_cls = None

        # create model instance and get distribution class
        self._set_model_and_dist(model_info, model_config, dist_info)

        # set requirements during sampling (e.g. for PPO: action, vf, logp(action))
        self._info_to_gather_during_sampling = ["action"]
        self._update_sampling_requirements()
        

    def _set_model_and_dist(self, model_info, model_config, dist_info=None):

        """
        This function initialize the model and save the dist class
        :param model_info: either a model class or the key to a model in the catalog
        :param model_config: model config dict
        :param dist_info: Optional. If None, the action space is used to retrieve the right class.
        :return:
        """
        if inspect.isclass(model_info):
            if issubclass(model_info, TorchModel):
                self.model = model_info(model_config).to(self.device)
            else:
                raise Exception("This is not a subclass of TorchModel")
        elif model_info in model_catalog.model_catalog:
            self.model = model_catalog.import_model(model_info, model_config).to(self.device)
        else:
            raise Exception("The model info are neither a torch model class nor a key with a model in the catalog")

        if dist_info is None:

            if self.conf["env_info"].get("action_space", None) is None:
                raise Exception("Neither action space nor distribution class are provided")
            else:
                self.dist_cls = torch_dist.get_dist_from_action_space(self.conf["env_info"]["action_space"])

        else :

            self.dist_cls = dist_info

    def _update_sampling_requirements(self):

        list_info_needed_during_sampling = self.conf["sampling"].get("info_to_sample", None)

        if list_info_needed_during_sampling is not None:
            for item in list_info_needed_during_sampling:
                self._info_to_gather_during_sampling.append(item)

    def sample(self, observation, state=None, hidden=None):

        """
        This function sample an action based on the observation/state provided
        :param observation:
        :param state:
        :param hidden:
        :return: dict with what is specified in self._info_to_gather_during_sampling
        e.g. for PPO: {"action" : ..., "vf" : ..., "logp" : ...}
        NOTE: each item in the dict has dim [B=1 x ....]
        """

        if hidden is None:
            with torch.no_grad():
                observation = torch.unsqueeze(observation, dim=0)  # B=1 x ...
                output_model = self.model(observation, state)
                # TODO: implement dist wrapper for managing different scenarios
                dist = self.dist_cls(logits=output_model)

                action = dist.sample()

        else:
            raise Exception("Not implemented!")

        sample_dict = {"action": torch.squeeze(action, dim=0)}

        if "logp" in self._info_to_gather_during_sampling:
            sample_dict["logp"] = torch.squeeze(dist.log_prob(action), dim=0)
        if "vf" in self._info_to_gather_during_sampling:
            sample_dict["vf"] = torch.squeeze(self.model.get_vf(), dim=0)

        return sample_dict

    def training_forward(self,  observation, state=None, hidden=None):

        """
        To be defined by a subclass
        :param observation:
        :param state:
        :param hidden:
        :return:
        """
        pass
