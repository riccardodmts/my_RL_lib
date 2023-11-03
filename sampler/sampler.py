from actor.actor import Actor
import numpy as np
from collections import OrderedDict


class Sampler:
    """
    A Sampler manages the sampling from several actors. This is the base class, each kind of sampling should
    be implemented by a child class. e.g. for a synchronous sampling with a multi-threading approach, a
    specific class should be developed.
    """

    def __init__(self, nr_actors):

        self.nr_actors = nr_actors

        # set of workers to be initialized (based on the kind of actor(thread, ray actor or normal actor))
        self.actors_set = {}

    def get_actor(self, actor_id):
        """
        Given an actor id, return the Actor object
        :param actor_id: integer = actor id
        :return: Actor requested
        """

        if actor_id >= self.nr_actors:
            raise Exception("Actor ID not valid!")

        return self.actors_set[actor_id]

    def get_actors(self):
        """
        Get all the actors used by these sampler
        :return: dict with all the actors
        """

        return self.actors_set

    def set_policies_weights(self, actor_ids, model_state_dict):
        """
        Set the weights of NNs for some of the actors (the ones specified in the actor_ids list)
        with the one in model_state_dict
        :param actor_ids: list
        :param model_state_dict:
        :return:
        """

        for actor_id in actor_ids:
            self.actors_set[actor_id].policy.set_weights(model_state_dict)

    def get_policies_weights(self, actor_ids):
        """
        Get the weights of the NNs for some of the actors (the ones specified in the actor_ids list)
        :param actor_ids:  list of actor ids
        :return:
        """

        for actor_id in actor_ids:
            if actor_id not in self.actors_set.keys():
                raise Exception("Actor IDs not valid!")

        return {actor_id: self.actors_set[actor_id].policy.get_weights() for actor_id in actor_ids}

    def sample_batch(self):
        """
        This method has to be implemented by a child class
        :return:
        """
        pass


class SequentialSampler(Sampler):

    def __init__(self, nr_actors, env_info, env_config, policy_cls, policy_dict, T=None, device="cpu"):

        super().__init__(nr_actors)

        self.actors_set = {actor_id: Actor(env_info, env_config, policy_cls, policy_dict, T, device)
                           for actor_id in range(nr_actors)
                           }
        self.sample_truncated_trajectories = T is not None

    def sample_batch(self, stack_trajectories=False):
        """
        Sample a batch of data
        :param stack_trajectories: stack arrays to obtain a batch [nr_actors x T x ...]
        :return: (batch, boolean)
        """

        stack_trajectories = stack_trajectories and self.sample_truncated_trajectories

        trajectories = {}
        for actor_id, actor in self.actors_set.items():
            trajectories[actor_id] = actor.sample_trajectory()

        batch = self._stack_trajectories(trajectories) if stack_trajectories else trajectories

        return batch, stack_trajectories

    def _stack_trajectories(self, trajectories):
        """
        Create numpy arrays of dimension [B x T x ...] for each quantity (obs, action, reward, ...)
        where B = number of actors
        :param trajectories:
        :return: Dict with numpy arrays
        """
        batch = {}
        B = 0
        for actor_id, trajectory in trajectories.items():

            for key, item in trajectory.items():
                if B == 0:
                    if key == "obs":
                        if isinstance(item, OrderedDict):
                            for k, el in item.items():
                                batch["obs_"+k] = np.zeros((self.nr_actors,)+el.shape, dtype=np.float32)
                                batch["obs_"+k][0] = el
                        else:
                            batch["obs"] = np.zeros((self.nr_actors,)+item.shape, dtype=np.float32)
                            batch["obs"][0] = item

                    elif isinstance(item, dict):
                        for k, el in item.items():
                            batch[k] = np.zeros((self.nr_actors,)+el.shape, dtype=np.float32)
                            batch[k][0] = el
                    else:
                        batch[key] = np.zeros((self.nr_actors,)+item.shape, dtype=np.float32)
                        batch[key][0] = item
                else:
                    if key == "obs":
                        if isinstance(item, OrderedDict):
                            for k, el in item.items():
                                batch["obs_"+k][B] = el
                        else:
                            batch["obs"][B] = item

                    elif isinstance(item, dict):
                        for k, el in item.items():
                            batch[k][B] = el
                    else:
                        batch[key][B] = item
            B += 1
        return batch





