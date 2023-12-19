"""
Base class for any pytorch based model
"""
import numpy as np
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
        :return: tensor [B x ...] or [B x T ...]
        """

        pass

    def get_vf(self):

        """
        This function should return V(s_t) in case a critic is used
        :return: tensor [B x 1] or [B x T x 1]
        """

        return self._value


""" Example model for PPO: actor + critic nets """


class MLPModel(TorchModel):

    def __init__(self, model_config):

        super(MLPModel, self).__init__(model_config)

        actor_config = model_config["actor"]
        critic_config = model_config["critic"]
        self.num_actions = model_config["num_actions"]

        # we assume 2 hidden layers for the actor and discrete action space
        self.actor_hidden_1 = nn.Linear(model_config["input_dim"], actor_config["first_hidden"])
        self.actor_hidden_2 = nn.Linear(actor_config["first_hidden"], actor_config["second_hidden"])
        self.actor_output = nn.Linear(actor_config["second_hidden"], model_config["num_actions"])

        self.actor_act_1 = nn.ReLU()
        self.actor_act_2 = nn.ReLU()

        # critic one hidden. no partial observability -> state and obs are the same
        self.critic_hidden = nn.Linear(model_config["input_dim"], critic_config["first_hidden"])
        self.critic_hidden2 = nn.Linear(critic_config["first_hidden"], critic_config["first_hidden"])
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
        v = self.critic_hidden2(v)
        v = self.critic_act(v)
        self._value = self.critic_output(v)

        return actions_logits


class MLPModelContinuous(TorchModel):

    def __init__(self, model_config):

        super(MLPModelContinuous, self).__init__(model_config)

        actor_config = model_config["actor"]
        critic_config = model_config["critic"]
        self.num_actions = model_config["num_actions"]
        self.output_mul = model_config.get("output_mul", None)
        if self.output_mul is not None:
            self.output_mul = torch.tensor(self.output_mul).float()

        # we assume 2 hidden layers for the actor and discrete action space
        self.actor_hidden_1 = nn.Linear(model_config["input_dim"], actor_config["first_hidden"])
        self.actor_hidden_2 = nn.Linear(actor_config["first_hidden"], actor_config["second_hidden"])
        self.actor_output = nn.Linear(actor_config["second_hidden"], model_config["num_actions"])

        self.actor_act_1 = nn.ReLU()
        self.actor_act_2 = nn.ReLU()
        self.actor_act_output = nn.Tanh()

        # critic one hidden. no partial observability -> state and obs are the same
        self.critic_hidden = nn.Linear(model_config["input_dim"], critic_config["first_hidden"])
        self.critic_hidden2 = nn.Linear(critic_config["first_hidden"], critic_config["first_hidden"])
        self.critic_output = nn.Linear(critic_config["first_hidden"], 1)

        self.critic_act = nn.ReLU()

    def forward(self, obs, state=None, hidden=None):

        x = self.actor_hidden_1(obs)
        x = self.actor_act_1(x)
        x = self.actor_hidden_2(x)
        x = self.actor_act_2(x)

        y = self.actor_output(x)
        mean = self.actor_act_output(y)
        if self.output_mul is not None:
            mean = self.output_mul * mean

        v = self.critic_hidden(obs)
        v = self.critic_act(v)
        v = self.critic_hidden2(v)
        v = self.critic_act(v)
        self._value = self.critic_output(v)

        return mean


class MLPModelContinuousV2(TorchModel):

    def __init__(self, model_config):

        super(MLPModelContinuousV2, self).__init__(model_config)

        actor_config = model_config["actor"]
        critic_config = model_config["critic"]
        self.num_actions = model_config["num_actions"]
        self.output_mul = model_config.get("output_mul", None)
        if self.output_mul is not None:
            self.output_mul = torch.tensor(self.output_mul).float()
        self.initial_std_const = model_config.get("initial_std_const", 0.0)

        # we assume 2 hidden layers for the actor and discrete action space
        self.actor_hidden_1 = nn.Linear(model_config["input_dim"], actor_config["first_hidden"])
        self.actor_hidden_2 = nn.Linear(actor_config["first_hidden"], actor_config["second_hidden"])
        self.actor_output = nn.Linear(actor_config["second_hidden"], model_config["num_actions"])

        self.actor_act_1 = nn.ReLU()
        self.actor_act_2 = nn.ReLU()
        self.actor_act_output = nn.Tanh()

        # critic one hidden. no partial observability -> state and obs are the same
        self.critic_hidden = nn.Linear(model_config["input_dim"], critic_config["first_hidden"])
        self.critic_hidden2 = nn.Linear(critic_config["first_hidden"], critic_config["first_hidden"])
        self.critic_output = nn.Linear(critic_config["first_hidden"], 1)

        self.critic_act = nn.ReLU()
        self.log_std = nn.Parameter(self.initial_std_const*torch.ones((1, self.num_actions)).float())

    def forward(self, obs, state=None, hidden=None):

        x = self.actor_hidden_1(obs)
        x = self.actor_act_1(x)
        x = self.actor_hidden_2(x)
        x = self.actor_act_2(x)

        y = self.actor_output(x)
        mean = self.actor_act_output(y)
        if self.output_mul is not None:
            mean = self.output_mul * mean

        v = self.critic_hidden(obs)
        v = self.critic_act(v)
        v = self.critic_hidden2(v)
        v = self.critic_act(v)
        self._value = self.critic_output(v)

        return mean, self.log_std

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

    model = MLPModel(model_dict)

    input = torch.ones((5, 10))  # batch size B = 5
    logits = model(input)

    print(logits)  # [B x 4]
    print(model.get_vf()) #  [B x 1]

