# Copyright 2019 The RecSim Authors. Licensed under the Apache License, Version 2.0.
"""Python 3 agent lifecycle and backend-independent checkpoint contracts."""

import abc
from copy import deepcopy
import logging
import numpy as np


class AbstractRecommenderAgent(abc.ABC):
    """An agent selects candidate slot indices and owns its policy/RNG state."""

    _multi_user = False

    def __init__(self, action_space):
        self._slate_size = action_space.nvec.shape[0]
        self._rng = np.random.default_rng(0)

    @property
    def multi_user(self):
        return self._multi_user

    @property
    def eval_mode(self):
        return getattr(self, "_eval_mode", False)

    @eval_mode.setter
    def eval_mode(self, value):
        self._eval_mode = bool(value)
        for agent in getattr(self, "_base_agents", None) or []:
            agent.eval_mode = value

    @abc.abstractmethod
    def step(self, reward, observation):
        """Consume a nonterminal transition and return the next slate."""

    def state_dict(self):
        """Copy policy state, retaining constructor-owned callables in the instance.

        All counters, arrays, generators and hierarchy children are included.
        Specialized backends override this to encode network and optimizer state.
        """
        excluded = {
            "_summary_writer",
            "_base_agents",
            "_base_agent_ctors",
            "_exploration_functions",
            "_observation_featurizer",
            "_doc_equality_walker",
            "_doc_comparator",
            "_slate_comparator",
            "_kwargs",
        }
        state = {
            k: deepcopy(v)
            for k, v in self.__dict__.items()
            if k not in excluded and not callable(v)
        }
        state["_children"] = [a.state_dict() for a in getattr(self, "_base_agents", None) or []]
        return state

    def load_state_dict(self, state):
        state = deepcopy(state)
        children = state.pop("_children", [])
        agents = getattr(self, "_base_agents", None) or []
        if len(children) != len(agents):
            raise ValueError("Checkpoint agent hierarchy does not match")
        self.__dict__.update(state)
        for agent, child in zip(agents, children):
            agent.load_state_dict(child)

    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        """Compatibility adapter. New runners use state_dict instead."""
        return self.state_dict()

    def unbundle(self, checkpoint_dir, iteration_number, bundle_dict):
        """Compatibility adapter for a state dictionary created by this agent."""
        self.load_state_dict(bundle_dict)
        return True


class AbstractEpisodicRecommenderAgent(AbstractRecommenderAgent):
    def __init__(self, action_space, summary_writer=None):
        super().__init__(action_space)
        self._episode_num = 0
        self._summary_writer = summary_writer

    def begin_episode(self, observation=None):
        self._episode_num += 1
        return self.step(0, observation)

    def end_episode(self, reward, observation=None, *, terminated=True, truncated=False):
        """Consume the final transition; bootstrap only when not terminated."""

    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        """Preserve the old minimal bundle shape; use state_dict for full state."""
        return {"episode_num": self._episode_num}

    def unbundle(self, checkpoint_dir, iteration_number, bundle_dict):
        if "episode_num" not in bundle_dict:
            logging.warning("Missing episode_num in legacy bundle")
            return False
        self._episode_num = bundle_dict["episode_num"]
        return True


class AbstractMultiUserEpisodicRecommenderAgent(AbstractEpisodicRecommenderAgent):
    """Receives per-user reward tuples from the runner's multi-user adapter."""

    _multi_user = True

    def __init__(self, action_space):
        self._num_users = len(action_space)
        if self._num_users == 0:
            raise ValueError("Multi-user agent must have at least one user")
        super().__init__(action_space[0])


class AbstractHierarchicalAgentLayer(AbstractRecommenderAgent):
    """Composable observation/reward transforms and child policies."""

    def __init__(self, action_space, *base_agent_ctors):
        super().__init__(action_space)
        self._base_agent_ctors = base_agent_ctors
        self._base_agents = None

    def _preprocess_reward_observation(self, reward, observation):
        return reward, observation

    @abc.abstractmethod
    def _postprocess_actions(self, action_list):
        """Combine child actions into a slate."""

    def begin_episode(self, observation=None):
        if observation is not None:
            _, observation = self._preprocess_reward_observation(0, observation)
        return self._postprocess_actions([a.begin_episode(observation) for a in self._base_agents])

    def end_episode(self, reward, observation, *, terminated=True, truncated=False):
        reward, observation = self._preprocess_reward_observation(reward, observation)
        for agent in self._base_agents:
            agent.end_episode(reward, observation, terminated=terminated, truncated=truncated)

    def bundle_and_checkpoint(self, checkpoint_dir, iteration_number):
        return {
            f"base_agent_bundle_{i}": a.bundle_and_checkpoint(checkpoint_dir, iteration_number)
            for i, a in enumerate(self._base_agents)
        }

    def unbundle(self, checkpoint_dir, iteration_number, bundle_dict):
        if any(f"base_agent_bundle_{i}" not in bundle_dict for i in range(len(self._base_agents))):
            return False
        return all(
            a.unbundle(checkpoint_dir, iteration_number, bundle_dict[f"base_agent_bundle_{i}"])
            for i, a in enumerate(self._base_agents)
        )
