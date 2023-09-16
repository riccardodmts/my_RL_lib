"""
Base class for any pytorch based model
"""

from torch import nn as nn
import torch


class TorchModel(nn.Module):

    def __init__(self, model_config):

        super(TorchModel, self).__init__()
        # store state value function if critic is used
        self._value = None

    def forward(self, obs, state=None, hidden=None):
        """
        :param obs: observation of the agent. tensor [B x ...] or [B x T x ...]
        :param state: state in case critic is used. Otherwise, not used. same as obs, tensor
        :param hidden: eventually for Recurrent RL algorithm.
        :return: tensor [B x action_space] or [B x T x action_space]
        """

        pass

    def get_vf(self):

        """
        This function should return V(s_t) in case a critic is used
        :return: tensor [B x 1] or [B x T x 1]
        """

        return self._value


""" Example model for PPO: actor + critic nets """


class LinearPPOModel(TorchModel):

    def __init__(self, model_config):

        super(LinearPPOModel, self).__init__(model_config)

        actor_config = model_config["actor"]
        critic_config = model_config["critic"]

        # we assume 2 hidden layers for the actor and discrete action space
        self.actor_hidden_1 = nn.Linear(model_config["input_dim"], actor_config["first_hidden"])
        self.actor_hidden_2 = nn.Linear(actor_config["first_hidden"], actor_config["second_hidden"])
        self.actor_output = nn.Linear(actor_config["second_hidden"], model_config["num_actions"])

        self.actor_act_1 = nn.ReLU()
        self.actor_act_2 = nn.ReLU()

        # critic one hidden. no partial observability -> state and obs are the same
        self.critic_hidden = nn.Linear(model_config["input_dim"], critic_config["first_hidden"])
        self.critic_output = nn.Linear(critic_config["first_hidden"], 1)

        self.critic_act = nn.ReLU()

    def forward(self, obs, state=None, hidden=None):

        x = self.actor_hidden_1(obs)
        x = self.actor_act_1(x)
        x = self.actor_hidden_2(x)
        x = self.actor_act_2(x)

        actions_logits = self.actor_output(x)

        v = self.critic_hidden(obs)
        v = self.critic_act(v)
        self._value = self.critic_output(v)

        return actions_logits


if __name__ == "__main__":

    # example model with LinearPPOModel

    model_dict = {

        "input_dim": 10,
        "num_actions": 4,

        "actor": {
            "first_hidden": 32,
            "second_hidden": 32
        },

        "critic": {

            "first_hidden": 32

        }
    }

    model = LinearPPOModel(model_dict)

    input = torch.ones((5, 10))  # batch size B = 5
    logits = model(input)

    print(logits)  # [B x 4]
    print(model.get_vf()) #  [B x 1]

