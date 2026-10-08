# Copyright 2019 The RecSim Authors. Licensed under the Apache License, Version 2.0.
"""Slate-Q selection and all four bootstrap targets, implemented in PyTorch."""

from itertools import combinations
from math import comb
import warnings
import gin
import numpy as np
from recsim import choice_model
from recsim.agents.torch.dqn_agent import DQNAgentRecSim, torch


def score_documents(user_obs, doc_obs, no_click_mass=1.0, is_mnl=False, min_normalizer=-1.0):
    scores = np.append(np.asarray(doc_obs) @ np.asarray(user_obs), no_click_mass)
    scores = choice_model.softmax(scores) if is_mnl else scores - min_normalizer
    if not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError("Scores must be finite and nonnegative")
    return scores[:-1], scores[-1]


def score_documents_torch(user_obs, doc_obs, no_click_mass=1.0, is_mnl=False, min_normalizer=-1.0):
    scores = (doc_obs * user_obs).sum(dim=-1)
    scores = torch.cat((scores, scores.new_tensor([no_click_mass])))
    scores = torch.softmax(scores, dim=0) if is_mnl else scores - min_normalizer
    if not torch.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("Scores must be finite and nonnegative")
    return scores[:-1], scores[-1]


def compute_probs(slate, scores, score_no_click):
    selected = scores[torch.as_tensor(slate, dtype=torch.long, device=scores.device).reshape(-1)]
    denominator = selected.sum() + score_no_click
    if not torch.isfinite(denominator) or denominator <= 0:
        raise ValueError("Slate normalization must be positive and finite")
    return selected / denominator


def _validate_selection(slate_size, s, q):
    if s.ndim != 1 or q.shape != s.shape or not 1 <= slate_size <= len(s):
        raise ValueError("Invalid slate size or score shape")
    if not torch.isfinite(q).all() or not torch.isfinite(s).all() or (s < 0).any():
        raise ValueError("Nonfinite Q values or invalid scores")


def select_slate_topk(slate_size, s_no_click, s, q):
    _validate_selection(slate_size, s, q)
    return torch.argsort(s * q, descending=True, stable=True)[:slate_size]


def select_slate_greedy(slate_size, s_no_click, s, q):
    _validate_selection(slate_size, s, q)
    selected = []
    numerator, denominator = (
        s.new_tensor(0.0),
        torch.as_tensor(s_no_click, dtype=s.dtype, device=s.device),
    )
    for _ in range(slate_size):
        total = denominator + s
        values = torch.where(total > 0, (numerator + s * q) / total, -torch.inf)
        if selected:
            values[selected] = -torch.inf
        index = int(values.argmax())
        if not torch.isfinite(values[index]):
            raise ValueError("No slate with positive normalization")
        selected.append(index)
        numerator = numerator + s[index] * q[index]
        denominator = denominator + s[index]
    # Match the legacy greedy selector's candidate-order output.
    return torch.tensor(sorted(selected), dtype=torch.long, device=s.device)


def select_slate_optimal(slate_size, s_no_click, s, q, max_slates=100000):
    _validate_selection(slate_size, s, q)
    if comb(len(s), slate_size) > max_slates:
        raise ValueError("Optimal slate enumeration exceeds max_slates")
    # The normalizable choice objective is order-independent, so combinations suffice.
    slates = torch.tensor(list(combinations(range(len(s)), slate_size)), device=s.device)
    denominators = s[slates].sum(dim=1) + s_no_click
    values = torch.where(
        denominators > 0, (s[slates] * q[slates]).sum(dim=1) / denominators, -torch.inf
    )
    index = int(values.argmax())
    if not torch.isfinite(values[index]):
        raise ValueError("No slate with positive normalization")
    return slates[index]


def _target(
    reward,
    gamma,
    next_actions,
    next_q_values,
    next_states,
    terminals,
    selector=None,
    no_click_mass=1.0,
    is_mnl=False,
    min_normalizer=-1.0,
):
    future = []
    for actions, q, state, terminal in zip(next_actions, next_q_values, next_states, terminals):
        if bool(terminal):
            future.append(q.new_tensor(0.0))
            continue
        s, no_click = score_documents_torch(
            state[0, :, -1], state[1:, :, -1], no_click_mass, is_mnl, min_normalizer
        )
        slate = actions if selector is None else selector(len(actions), no_click, s, q)
        future.append((compute_probs(slate, s, no_click) * q[slate.long()]).sum())
    return reward + gamma * torch.stack(future) * (1 - terminals.to(reward.dtype))


def compute_target_sarsa(
    reward, gamma, next_actions, next_q_values, next_states, terminals, **kwargs
):
    return _target(reward, gamma, next_actions, next_q_values, next_states, terminals, **kwargs)


def compute_target_greedy_q(
    reward, gamma, next_actions, next_q_values, next_states, terminals, **kwargs
):
    return _target(
        reward,
        gamma,
        next_actions,
        next_q_values,
        next_states,
        terminals,
        select_slate_greedy,
        **kwargs,
    )


def compute_target_topk_q(
    reward, gamma, next_actions, next_q_values, next_states, terminals, **kwargs
):
    return _target(
        reward,
        gamma,
        next_actions,
        next_q_values,
        next_states,
        terminals,
        select_slate_topk,
        **kwargs,
    )


def compute_target_optimal_q(
    reward, gamma, next_actions, next_q_values, next_states, terminals, **kwargs
):
    return _target(
        reward,
        gamma,
        next_actions,
        next_q_values,
        next_states,
        terminals,
        select_slate_optimal,
        **kwargs,
    )


def score_documents_tf(*args, **kwargs):
    """Deprecated spelling; accepts and returns Torch tensors, never TF tensors."""
    warnings.warn("Use score_documents_torch", DeprecationWarning, stacklevel=2)
    return score_documents_torch(*args, **kwargs)


def compute_probs_tf(*args, **kwargs):
    """Deprecated spelling for compute_probs with Torch tensors."""
    warnings.warn("Use compute_probs", DeprecationWarning, stacklevel=2)
    return compute_probs(*args, **kwargs)


@gin.configurable
class SlateDecompQAgent(DQNAgentRecSim):
    def __init__(
        self,
        observation_space,
        action_space,
        *,
        select_slate_fn=select_slate_greedy,
        compute_target_fn=compute_target_greedy_q,
        no_click_mass=1.0,
        is_mnl=False,
        min_normalizer=-1.0,
        **kwargs,
    ):
        response = observation_space["response"][0]
        if not {"click", "watch_time"}.issubset(response.spaces):
            raise ValueError("SlateDecompQ requires click and watch_time responses")
        user_shape = observation_space["user"].shape
        if user_shape != next(iter(observation_space["doc"].spaces.values())).shape:
            raise ValueError("SlateDecompQ requires matching vector user/document features")
        self._select_slate_fn = select_slate_fn
        self._compute_target_fn = compute_target_fn
        self._score_config = dict(
            no_click_mass=no_click_mass, is_mnl=is_mnl, min_normalizer=min_normalizer
        )
        super().__init__(observation_space, action_space, **kwargs)
        self._config.update(
            score=self._score_config,
            selector=select_slate_fn.__name__,
            target=compute_target_fn.__name__,
        )

    def _q_values(self, network, states):
        states = states[..., 0]
        users = states[:, :1].expand(-1, self._num_candidates, -1)
        return network(torch.cat((users, states[:, 1:]), dim=-1)).squeeze(-1)

    def _greedy(self, state):
        state = self._tensor(state)
        q = self._q_values(self.online, state[None])[0]
        scores, no_click = score_documents_torch(
            state[0, :, 0], state[1:, :, 0], **self._score_config
        )
        return self._select_slate_fn(self._slate_size, no_click, scores, q).tolist()

    def _loss(self, batch):
        # Preserve Slate-Q's clicked-item regression and watch-time reward.
        batch = [t for t in batch if sum(r["click"] for r in t["response"]) == 1]
        if not batch:
            return None
        states = self._tensor(np.stack([t["state"] for t in batch]))
        next_states = self._tensor(np.stack([t["next_state"] for t in batch]))
        actions = self._tensor([t["action"] for t in batch], torch.long)
        clicks = self._tensor([[r["click"] for r in t["response"]] for t in batch])
        predicted = (self._q_values(self.online, states).gather(1, actions) * clicks).sum(dim=1)
        with torch.no_grad():
            rewards = self._tensor(
                [sum(r["click"] * float(r["watch_time"]) for r in t["response"]) for t in batch]
            )
            target = self._compute_target_fn(
                reward=rewards,
                gamma=self._tensor([t["discount"] for t in batch]),
                next_actions=self._tensor([t["next_action"] for t in batch], torch.long),
                next_q_values=self._q_values(self.target, next_states),
                next_states=next_states,
                terminals=self._tensor([t["terminated"] for t in batch]),
                **self._score_config,
            )
        return torch.nn.functional.mse_loss(predicted, target)


def create_agent(agent_name, **kwargs):
    if agent_name == "dp_random":
        return SlateDecompQAgent(
            select_slate_fn=select_slate_greedy,
            compute_target_fn=compute_target_sarsa,
            **{**kwargs, "epsilon_train": 1.0, "epsilon_eval": 1.0},
        )
    if agent_name.startswith("myopic_"):
        kwargs["gamma"] = 0.0
        agent_name = agent_name[len("myopic_") :]
    selectors = {
        "topk": select_slate_topk,
        "greedy": select_slate_greedy,
        "optimal": select_slate_optimal,
    }
    targets = {
        "sarsa": compute_target_sarsa,
        "greedy_q": compute_target_greedy_q,
        "topk_q": compute_target_topk_q,
        "optimal_q": compute_target_optimal_q,
    }
    parts = agent_name.split("_", 2)
    if (
        len(parts) != 3
        or parts[0] != "slate"
        or parts[1] not in selectors
        or parts[2] not in targets
    ):
        raise ValueError(f"Unknown Slate-Q agent: {agent_name}")
    return SlateDecompQAgent(
        select_slate_fn=selectors[parts[1]], compute_target_fn=targets[parts[2]], **kwargs
    )
