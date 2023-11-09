import torch

from policy.policy import Policy


class PPOPolicy(Policy):

    def __init__(self,  model_info, model_config, policy_config, dist_info=None,  device="cpu"):

        super().__init__(model_info, model_config, policy_config, dist_info, device)
        self.entropy_for_training = True

    def collect_entropy(self):
        self.entropy_for_training = True

    def training_forward(self, observation, action, state=None, hidden=None):

        model_output = self.model(observation, state)
        dist = self.dist_cls(model_output)

        logp_theta = dist.log_prob(torch.squeeze(action, dim=1))  # B
        vf = torch.squeeze(self.model.get_vf(), dim=1)  # B
        entropy = None

        if self.entropy_for_training:
            entropy = dist.entropy()

        return logp_theta, vf, entropy





