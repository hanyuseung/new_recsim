# Copyright 2019 The RecSim Authors. Licensed under the Apache License, Version 2.0.
"""Gymnasium adapter with stable candidate slots and explicit episode boundaries."""
from collections import OrderedDict, defaultdict
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from recsim.spaces import coerce, zeros
from recsim.simulator import environment


def _dummy_metrics_aggregator(responses, metrics, info):
    return metrics


def _dummy_metrics_writer(metrics, add_summary_fn):
    return None


class RecSimGymEnv(gym.Env):
    metadata = {'render_modes': []}
    render_mode = None

    def __init__(self, raw_environment, reward_aggregator,
                 metrics_aggregator=_dummy_metrics_aggregator,
                 metrics_writer=_dummy_metrics_writer, *, seed=None, max_episode_steps=None):
        self._environment = raw_environment
        self._reward_aggregator = reward_aggregator
        self._metrics_aggregator = metrics_aggregator
        self._metrics_writer = metrics_writer
        self._metrics = defaultdict(float)
        self._multi = isinstance(raw_environment, environment.MultiUserEnvironment)
        self._seed = seed
        self._needs_seed = seed is not None
        if max_episode_steps is not None and (not isinstance(max_episode_steps, int) or max_episode_steps <= 0):
            raise ValueError('max_episode_steps must be a positive integer')
        self.max_episode_steps = max_episode_steps
        self._steps = 0
        self._ended = True
        raw = self.environment
        def action():
            return spaces.MultiDiscrete(np.full(raw.slate_size, raw.num_candidates))
        models = raw.user_models
        self.action_space = spaces.Tuple(tuple(action() for _ in models)) if self._multi else action()
        user_spaces = [m.observation_space() for m in models]
        response_spaces = [m.response_space() for m in models]
        docs = raw.candidate_set.get_all_documents()
        self.observation_space = spaces.Dict({
            'user': spaces.Tuple(tuple(user_spaces)) if self._multi else user_spaces[0],
            'doc': spaces.Dict(OrderedDict((str(i), d.observation_space()) for i, d in enumerate(docs))),
            'response': spaces.Tuple(tuple(response_spaces)) if self._multi else response_spaces[0],
        })

    @property
    def environment(self):
        return self._environment

    @property
    def num_candidates(self):
        return self.environment.num_candidates

    @property
    def slate_size(self):
        return self.environment.slate_size

    def _observation(self, user, docs, responses=None):
        response_space = self.observation_space['response']
        if responses is None:
            response = tuple(tuple(m.get_response_model_ctor()().create_observation() for _ in range(self.slate_size)) for m in self.environment.user_models)
            if not self._multi:
                response = response[0]
        elif self._multi:
            response = tuple(tuple(r.create_observation() for r in rs) if rs else zeros(s)
                             for rs, s in zip(responses, response_space.spaces))
        else:
            response = tuple(r.create_observation() for r in responses) if responses else zeros(response_space)
        obs = {'user': user, 'doc': OrderedDict((str(i), v) for i, v in enumerate(docs.values())),
               'response': response}
        return coerce(self.observation_space, obs)

    def extract_env_info(self):
        return {'document_ids': np.array([int(k) for k in self.environment._current_documents], dtype=np.int64),
                'user_terminated': tuple(bool(m.is_terminal()) for m in self.environment.user_models)}

    def reset(self, *, seed=None, options=None):
        if seed is None and self._needs_seed:
            seed = self._seed
        super().reset(seed=seed)
        if seed is not None:
            self._seed = seed
            self.environment.reset_sampler(seed)
            self.action_space.seed(seed)
        self._needs_seed = False
        user, docs = self.environment.reset()
        self._steps, self._ended = 0, False
        info = self.extract_env_info()
        info['is_reset'] = True
        return self._observation(user, docs), info

    def step(self, action):
        if self._ended:
            raise RuntimeError('Call reset() before step() or after episode termination')
        previous_ids = self.extract_env_info()['document_ids']
        active = tuple(not m.is_terminal() for m in self.environment.user_models)
        user, docs, responses, terminated = self.environment.step(action)
        self._steps += 1
        truncated = self.max_episode_steps is not None and self._steps >= self.max_episode_steps
        self._ended = bool(terminated or truncated)
        info = self.extract_env_info()
        info['is_reset'] = False
        info['user_active'] = active
        info['action_document_ids'] = previous_ids[np.asarray(action, dtype=int)].copy()
        if self._multi:
            rewards = [float(self._reward_aggregator(rs)) if rs else 0.0 for rs in responses]
            info['user_rewards'] = tuple(rewards)
            reward = sum(rewards)
        else:
            reward = float(self._reward_aggregator(responses))
        return self._observation(user, docs, responses), reward, bool(terminated), bool(truncated), info

    def reset_sampler(self, seed=None):
        self._seed = self._seed if seed is None else seed
        self.environment.reset_sampler(self._seed)
        self._ended = True

    def reset_metrics(self):
        self._metrics = defaultdict(float)

    def update_metrics(self, responses, info=None):
        if self._multi:
            active = (info or {}).get('user_active', [True] * len(responses))
            for per_user, is_active in zip(responses, active):
                if not is_active:
                    continue
                self._metrics = self._metrics_aggregator(per_user, self._metrics, info)
        else:
            self._metrics = self._metrics_aggregator(responses, self._metrics, info)

    def write_metrics(self, add_summary_fn):
        self._metrics_writer(self._metrics, add_summary_fn)

    def render(self):
        return None

    def close(self):
        self._ended = True
