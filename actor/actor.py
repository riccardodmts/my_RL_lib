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
        # if continuous action space, store possible min/max values (used for clipping sampled actions)
        self.bounds = None
        self._check_bounds()

        self.last_obs, _ = self.env.reset()
        self.last_obs = self._to_float32(self.last_obs)

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

    def _to_float32(self, obs):
        """
        Convert numpy arrays to float32 arrays. If dict, each array is converted.
        :param obs: observation to convert
        :return: obs converted to np.array with dtype=np.float32
        """

        is_dict = isinstance(obs, dict)

        if is_dict:
            for key in obs:
                obs[key] = np.array(obs[key], dtype=np.float32)
        else:
            obs = np.array(obs, dtype=np.float32)

        return obs

    def _to_tensor(self, obs):
        """
        Convert obs to torch tensor. If dict, each array is converted.
        :param obs: observation to convert
        :return: obs converted to tensor
        """

        is_dict = isinstance(obs, dict)
        if is_dict:
            obs_new = {}
        else:
            obs_new = None

        if is_dict:
            for key in obs:
                obs_new[key] = torch.tensor(obs[key])
        else:
            obs_new = torch.tensor(obs)

        return obs_new

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
            for item in self.last_obs.items():

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

    def _check_bounds(self):
        """
        Check if in case of continuous action space, any bound is set. In that case update
        self.bounds
        :return: boolean (if any bound exists)
        """

        action_space = self.env.action_space
        if isinstance(action_space, gym.spaces.Box):
            self.bounds = []
            bounds_above = action_space.high
            bounds_below = action_space.low
            has_any_bound = 2 - np.isinf(bounds_above) - np.isinf(bounds_below)
            has_any_bound = has_any_bound > 0
            for i in range(len(bounds_above)):
                self.bounds.append((has_any_bound[i], bounds_below[i], bounds_above[i]))

    def _add_obs_to_buffer(self, obs, t):
        """
        Add observation to buffer
        :param obs: observation to add
        :param t: time instant in a trajectory
        :return:
        """

        if self._is_obs_dict:
            for key in self.observations:
                if self.T:
                    self.observations[key][t] = obs[key]
                else:
                    self.observations[key].append(obs[key])
        else:
            if self.T:
                self.observations[t] = obs
            else:
                self.observations.append(obs)

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
            # if action space is a scalar, add extra dimension e.g tensor [T x 1] instead of [T,]
            action_dim = (1, ) + action.shape if not len(action.shape) else action.shape

            self.actions = np.zeros((self.T,) + action_dim, dtype=action.dtype)
            self.actions[0] = action
            if other_info is not None:
                for key in other_info.keys():
                    item = other_info[key]
                    # if the extra info is a scalar add extra dimension
                    other_item_dim = (1, ) + item.shape if not len(item.shape) else item.shape
                    # if vf, add 1 extra sample
                    extra_sample = 1 if key == "vf" else 0
                    self.sampling_info[key] = np.zeros((self.T+extra_sample,)+other_item_dim, dtype=item.dtype)
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
        self._add_obs_to_buffer(obs, t+1)

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

    def _clip_action(self, action):
        """
        Given a sampled action, clip it based on self.bounds
        :param action: action sampled
        :return: action clipped
        """
        # clip each single component (if necessary)
        action_clipped = copy.deepcopy(action)
        for i in range(len(action)):
            if self.bounds[i][0]:
                min_ = None if np.isinf(self.bounds[i][1]) else self.bounds[i][1]
                max_ = None if np.isinf(self.bounds[i][2]) else self.bounds[i][2]
                action_clipped[i] = np.clip(action[i], a_min=min_, a_max=max_)

        return action_clipped

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

        current_obs = self._to_tensor(self.last_obs)
        sample_dict = self.policy.sample(current_obs)

        action = sample_dict.pop("action").numpy()
        action_clipped = self._clip_action(action) if self.bounds is not None else action

        if len(action_clipped.shape) == 0:
            action_to_env = action_clipped.item()
        else:
            action_to_env = action_clipped

        if len(list(sample_dict.keys())) > 0:
            other_info = {key: sample_dict[key].numpy() for key in sample_dict.keys()}
        else:
            other_info = None

        next_obs, reward, terminated, truncated, info = self.env.step(action_to_env)

        done = terminated or truncated

        return action, self._to_float32(next_obs), reward, done, other_info

    def _manage_end_episode(self, t):
        """
        Handle end of an episode in both cases: self.T is or not None
        :param t: time instant in between a trajectory
        :return:
        """

        # reset env and get new initial state
        new_obs, _ = self.env.reset()
        new_obs = self._to_float32(new_obs)
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
        else:
            # TODO: refactor this (_init_observation)
            # set last state as initial state (if this the n trajectory to collect, the last observation
            # from trajectory n-1)
            self._add_obs_to_buffer(self.last_obs, 0)

        """
        --- SAMPLING ---
        """
        while True:
            # exit from loop if:
            #   - if T not None and T samples have been collected
            #   - if end episode (T is None)
            if self.T:
                if t == self.T:
                    current_obs = self._to_tensor(self.last_obs)
                    sample_dict = self.policy.sample(current_obs)
                    if "vf" in sample_dict:
                        self.sampling_info["vf"][self.T] = sample_dict["vf"]

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
        --- STACK IN ONE TENSOR ---
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
