"""
Actor class: an actor has to instantiate the environment and collect samples from it by using a policy
object for choosing actions
"""

import gymnasium as gym
from collections import OrderedDict
import numpy as np
import torch
import copy


class Actor:

    def __init__(self, env_info, env_config, policy_cls, policy_dict, T=None, device="cpu"):
        """
        :param env_info: env info: either env class or env string to gym registered env
        :param env_config: config dict for the env
        :param policy_cls: class for the policy
        :param policy_dict: dict for the policy with:
            -model_info
            -model_config
            -policy_config
            -dist_info
        :param T: trajectory length. If None, no truncation.
        :param device: cpu or cuda
        """
        self.device = device
        self.env = None
        # initialize env
        self._init_env(env_info, env_config)
        self.last_obs, _ = self.env.reset()
        self.last_obs = np.array(self.last_obs, dtype=np.float32)

        self.T = T

        # init policy obj
        self.policy = None
        self._init_policy(policy_cls, policy_dict)

        # to store all the observations
        self.observations = None
        self._init_observations()

        # to store reward and actions
        self.actions = None
        self.rewards = None
        self.dones = None
        # initialized during first sampling

        # to store other info (base on the first return of the policy, the keys are added)
        # e.g. for PPO value_function, logp
        self.sampling_info = {}


    def _init_env(self, env_info, env_config):
        """
        Init the env
        :param env_info:
        :param env_config:
        :return:
        """
        if isinstance(env_info, str):
            if env_info in gym.envs.registry.keys():
                self.env = gym.make(env_info, **env_config)
            else:
                raise Exception("Env not registered in gym!")

        else:
            # gym class
            self.env = env_info(env_config)

    def _init_observations(self):
        """
        Initialize self.observations based on the kind of observation space (Dict or tensor. No other).
        In case of a Dict space, for each key a tensor is needed.
        :return:
        """

        self._is_obs_dict = isinstance(self.env.observation_space, gym.spaces.Dict)
        self.observations = OrderedDict()
        if self._is_obs_dict:
            for item in self.last_obs:

                if self.T:
                    # if we know in advance the length of the trajectory, create a tensor of zeros
                    # for the current key create a suitable tensor with dim [T x ...]
                    self.observations[item[0]] = np.zeros((self.T+1,)+item[1].shape, dtype=np.float32)
                    self.observations[item[0]][0] = item[1]
                else:
                    # if not we append to a list, we concatenate at the end
                    self.observations[item[0]] = [item[1]]
        else:

            if self.T:
                self.observations = np.zeros((self.T+1,)+self.last_obs.shape, dtype=np.float32)
                self.observations[0] = self.last_obs
            else:
                self.observations = [self.last_obs]

    def _init_policy(self, policy_cls, policy_dict):
        """
        Initialize policy object (used by the actor for sampling).
        :param policy_cls:
        :param policy_dict:
        :return:
        """

        model_info = policy_dict["model_info"]
        model_config = policy_dict["model_config"]
        policy_config = policy_dict["policy_config"]
        dist_info = policy_dict.get("dist_info", None)

        self.policy = policy_cls(model_info, model_config, policy_config, dist_info, self.device)

    def _init_buffer_trajectory(self, action, reward, done, other_info=None):
        """
        Init buffer for actions, rewards, dones and other_info
        :param action:
        :param reward:
        :param done:
        :param other_info:
        :return:
        """

        if self.T:

            self.rewards = np.zeros((self.T, 1), dtype=np.float32)
            self.rewards[0, 0] = reward
            self.dones = np.zeros((self.T, 1), dtype=np.int8)
            self.dones[0, 0] = 1 if done else 0
            self.actions = np.zeros((self.T,) + action.shape, dtype=action.dtype)
            self.actions[0] = action
            if other_info is not None:
                for key in other_info.keys():
                    item = other_info[key]
                    self.sampling_info[key] = np.zeros((self.T,)+item.shape, dtype=item.dtype)
                    self.sampling_info[key][0] = item
        else:
            self.rewards = [reward]
            self.dones = [1 if done else 0]
            self.actions = [action]
            if other_info is not None:
                for key in other_info.keys():
                    self.sampling_info[key] = [other_info[key]]

    def _add_sample_to_buffer(self, t, obs, action, reward, done, other_info=None):
        """
        Add the new sample to the buffer with all the samples collected up to now in one trajectory
        :param t: time instant
        :param obs: new observation (o_{t+1})
        :param action: action a_t sampled given o_t
        :param reward: reward r_t obtained by applying a_t
        :param done: done or not (end of episode reached or not)
        :param other_info: other info (e.g. PPO log pi(a_t|o_t))
        :return:
        """

        # obs

        if self._is_obs_dict:
            for key in self.observations:
                if self.T:
                    self.observations[key][t+1] = obs[key]
                else:
                    self.observations[key].append(obs[key])
        else:
            if self.T:
                self.observations[t+1] = obs
            else:
                self.observations.append(obs)

        # action, reward, done and other info

        if self.actions is None:
            # initialize buffer for actions, ...
            self._init_buffer_trajectory(action, reward, done, other_info)

        else:

            if self.T:
                self.rewards[t, 0] = reward
                self.dones[t, 0] = 1 if done else 0
                self.actions[t] = action
                if other_info is not None:
                    for key in other_info.keys():
                        self.sampling_info[key][t] = other_info[key]
            else:
                self.rewards.append(reward)
                self.dones.append(1 if done else 0)
                self.actions.append(action)
                if other_info is not None:
                    for key in other_info.keys():
                        self.sampling_info[key].append(other_info[key])

    def sample(self):
        """
        Sample from the environment a tuple of kind (a_t, o_{t+1}, r_t, done, ...)
        :return:
            action: a_t
            next_obs: o_{t+1}
            reward: r_t
            done: boolean
            other_info: if no info are needed, None
        """

        current_obs = torch.tensor(self.last_obs)
        sample_dict = self.policy.sample(current_obs)

        action = sample_dict.pop("action").numpy()
        if len(action.shape) == 0:
            action_to_env = action.item()
        else:
            action_to_env = action

        if len(list(sample_dict.keys())) > 0:
            other_info = {key: sample_dict[key].numpy() for key in sample_dict.keys()}
        else:
            other_info = None

        next_obs, reward, terminated, truncated, info = self.env.step(action_to_env)

        done = terminated or truncated

        return action, np.array(next_obs, dtype=np.float32), reward, done, other_info

    def _manage_end_episode(self, t):
        """
        Handle end of an episode in both cases self.T is or not None
        :param t: time instant in between a trajectory
        :return:
        """

        # reset env and get new initial state
        new_obs, _ = self.env.reset()
        new_obs = np.array(new_obs, dtype=np.float32)
        # save new state in buffer (the final state of the previous episode is not saved)
        # if self.T None, the observation buffer will be initialized in any case
        if self.T is not None:
            if self._is_obs_dict:

                for key in self.observations:
                    self.observations[key][t+1] = new_obs[key]
            else:

                self.observations[t+1] = new_obs

        self.last_obs = new_obs

    def sample_trajectory(self):

        t = 0
        done = False
        """
        --- INIT SAMPLING --- (new trajectory)
        """
        if self.T is None:
            # if complete episode, initialize the buffers. Otherwise, we just overwrite the old one.
            # by setting self.actions to None, _add_sample_to_buffer will call _init_buffer_trajectory
            self.actions = None
            self._init_observations()

        """
        --- SAMPLING ---
        """
        while True:
            # exit from loop if:
            #   - if T not None and T samples have been collected
            #   - if end episode (T is None)
            if self.T:
                if t == self.T:
                    break
            else:
                if done:
                    break

            action, next_obs, reward, done, other_info = self.sample()
            self._add_sample_to_buffer(t, next_obs, action, reward, done, other_info)

            if not done:
                self.last_obs = next_obs
            else:
                self._manage_end_episode(t)

            t += 1

        """
        --- STACK IN A TENSOR ---
        """
        # create tensor (numpy array) of type [T x ...] for each quantity and create a dict with them
        # if T is None, the length varies based on the number of action in the episode collected
        # if T set, the tensor is ready, just the dict has to be created
        if self.T is None:

            actions_trajectory = np.stack(self.actions, axis=0)
            rewards_trajectory = np.stack(self.rewards, axis=0)
            dones_trajectory = np.stack(self.dones, axis=0)

            other_info_trajectory = {}

            if len(list(self.sampling_info.keys())) > 0:
                for key in self.sampling_info:
                    other_info_trajectory[key] = np.stack(self.sampling_info[key], axis=0)
            else:
                other_info_trajectory = None

            obs_trajectory = {}
            if self._is_obs_dict:
                for key in self.observations:
                    obs_trajectory[key] = np.stack(self.observations[key], axis=0)

            return {"obs": obs_trajectory, "actions": actions_trajectory, "rewards": rewards_trajectory,
                    "dones": dones_trajectory, "other_info": other_info_trajectory}

        return {"obs": copy.deepcopy(self.observations), "actions": copy.deepcopy(self.actions),
                "dones": copy.deepcopy(self.dones), "rewards": copy.deepcopy(self.rewards),
                "other_info": copy.deepcopy(self.sampling_info)}



















