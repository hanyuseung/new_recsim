# Copyright 2019 The RecSim Authors. Licensed under the Apache License, Version 2.0.
"""Full-slate Q learning with a shared PyTorch user/slate network."""

from itertools import permutations
from math import perm
import gin
import numpy as np
from recsim.agents.torch.dqn_agent import DQNAgentRecSim, recsim_dqn_network, torch


@gin.configurable
class FullSlateQAgent(DQNAgentRecSim):
    def __init__(self, observation_space, action_space, *, max_slates=100000, **kwargs):
        n, k = int(action_space.nvec[0]), len(action_space.nvec)
        if perm(n, k) > max_slates:
            raise ValueError(
                f"FullSlateQ needs {perm(n, k)} slates; increase max_slates explicitly or use SlateDecompQ"
            )
        self._all_possible_slates = list(permutations(range(n), k))
        self._slate_indices = {s: i for i, s in enumerate(self._all_possible_slates)}
        super().__init__(observation_space, action_space, **kwargs)

    def _make_network(self, hidden_sizes):
        return recsim_dqn_network((1 + self._slate_size) * self._obs_adapter.width, hidden_sizes)

    def _q_values(self, network, states):
        states = states[..., 0]
        slates = self._tensor(self._all_possible_slates, torch.long)
        docs = states[:, 1:][:, slates].flatten(start_dim=2)
        users = states[:, :1].expand(-1, len(slates), -1)
        return network(torch.cat((users, docs), dim=-1)).squeeze(-1)

    def _greedy(self, state):
        q = self._q_values(self.online, self._tensor(state[None]))[0]
        return self._all_possible_slates[int(q.argmax())]

    def _loss(self, batch):
        states = self._tensor(np.stack([t["state"] for t in batch]))
        next_states = self._tensor(np.stack([t["next_state"] for t in batch]))
        indices = self._tensor([self._slate_indices[tuple(t["action"])] for t in batch], torch.long)
        predicted = self._q_values(self.online, states).gather(1, indices[:, None]).squeeze(1)
        with torch.no_grad():
            future = self._q_values(self.target, next_states).max(dim=1).values
            reward = self._tensor([t["reward"] for t in batch])
            terminal = self._tensor([t["terminated"] for t in batch])
            discount = self._tensor([t["discount"] for t in batch])
            target = reward + discount * (1 - terminal) * future
        return torch.nn.functional.mse_loss(predicted, target)
