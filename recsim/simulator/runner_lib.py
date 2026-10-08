# Copyright 2019 The RecSim Authors. Licensed under the Apache License, Version 2.0.
"""Finite training/evaluation loops and versioned episode-boundary checkpoints."""

from collections import defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import re
import time
import gin
import numpy as np
from recsim.simulator.environment import MultiUserEnvironment


def load_gin_configs(gin_files, gin_bindings):
    gin.parse_config_files_and_bindings(gin_files, gin_bindings, finalize_config=False)


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


@gin.configurable
class Runner:
    def __init__(
        self,
        base_dir,
        create_agent_fn,
        env,
        episode_log_file="",
        checkpoint_file_prefix="ckpt",
        max_steps_per_episode=27000,
        tensorboard=False,
    ):
        if base_dir is None or max_steps_per_episode < 1:
            raise ValueError("base_dir and a positive episode limit are required")
        self._base_dir = Path(base_dir)
        self._create_agent_fn = create_agent_fn
        self._env = env
        self._episode_log_file = episode_log_file
        self._checkpoint_file_prefix = checkpoint_file_prefix
        self._max_steps_per_episode = max_steps_per_episode
        self._tensorboard = tensorboard
        self._episode_writer = None
        self._summary_writer = NullWriter()
        self._stats = defaultdict(list)
        self._closed = False

    def _set_up(self, eval_mode):
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self._tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self._summary_writer = SummaryWriter(str(self._output_dir))
        self._agent = self._create_agent_fn(
            self._env, eval_mode=eval_mode, summary_writer=self._summary_writer
        )
        self._agent.eval_mode = eval_mode
        if self._agent.multi_user != isinstance(self._env.environment, MultiUserEnvironment):
            self.close()
            raise ValueError("Agent and environment must agree on single/multi-user mode")
        if self._episode_log_file:
            path = self._output_dir / self._episode_log_file
            path.parent.mkdir(parents=True, exist_ok=True)
            self._episode_writer = path.open("a", encoding="utf-8")

    def _run_one_episode(self):
        observation, reset_info = self._env.reset()
        action = self._agent.begin_episode(observation)
        total_reward = 0.0
        for step in range(1, self._max_steps_per_episode + 1):
            previous = observation
            observation, reward, terminated, truncated, info = self._env.step(action)
            truncated = bool(truncated or step == self._max_steps_per_episode)
            total_reward += reward
            self._env.update_metrics(observation["response"], info)
            if self._episode_writer is not None:
                record = dict(
                    step=step - 1,
                    observation=previous,
                    action=action,
                    document_ids=info["action_document_ids"],
                    next_observation=observation,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                )
                self._episode_writer.write(json.dumps(json_value(record), allow_nan=False) + "\n")
            agent_reward = info["user_rewards"] if self._agent.multi_user else reward
            if terminated or truncated:
                self._agent.end_episode(
                    agent_reward, observation, terminated=terminated, truncated=truncated
                )
                break
            action = self._agent.step(agent_reward, observation)
        if self._episode_writer is not None:
            self._episode_writer.flush()
        self._stats["episode_length"].append(step)
        self._stats["episode_reward"].append(total_reward)
        return step, total_reward

    def _latest_checkpoint(self):
        pattern = re.compile(re.escape(self._checkpoint_file_prefix) + r"_(\d+)\.pkl$")
        candidates = [
            (int(m.group(1)), p)
            for p in self._checkpoint_dir.iterdir()
            if (m := pattern.fullmatch(p.name))
        ]
        return max(candidates, default=(None, None), key=lambda pair: pair[0])

    def _environment_signature(self):
        raw = self._env.environment
        return (
            type(raw).__name__,
            tuple(type(m).__name__ for m in raw.user_models),
            raw.num_candidates,
            raw.slate_size,
            raw._resample_documents,
            self._env.max_episode_steps,
            self._max_steps_per_episode,
        )

    def _checkpoint_experiment(self, iteration, total_steps):
        # Checkpoints contain Python/NumPy and possibly Torch objects. Load only trusted files.
        state = dict(
            schema_version=1,
            current_iteration=iteration,
            total_steps=total_steps,
            agent_type=f"{type(self._agent).__module__}.{type(self._agent).__qualname__}",
            agent_state_dict=self._agent.state_dict(),
            environment_signature=self._environment_signature(),
            environment_state=deepcopy(self._env.environment.__dict__),
            wrapper_seed=self._env._seed,
            needs_seed=self._env._needs_seed,
        )
        path = self._checkpoint_dir / f"{self._checkpoint_file_prefix}_{iteration}.pkl"
        temporary = path.with_suffix(".tmp")
        try:
            with temporary.open("wb") as stream:
                pickle.dump(state, stream, protocol=pickle.HIGHEST_PROTOCOL)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def _load_checkpoint(self, path, restore_environment=False):
        try:
            with Path(path).open("rb") as stream:
                state = pickle.load(stream)
            if not isinstance(state, dict) or state.get("schema_version") != 1:
                raise ValueError(
                    "Unsupported checkpoint schema (legacy TF checkpoints need conversion)"
                )
            if state["environment_signature"] != self._environment_signature():
                raise ValueError("Checkpoint environment configuration differs")
            if state["agent_type"] != (
                f"{type(self._agent).__module__}.{type(self._agent).__qualname__}"
            ):
                raise ValueError("Checkpoint agent type differs")
            self._agent.load_state_dict(state["agent_state_dict"])
            if restore_environment:
                self._env.environment.__dict__.update(deepcopy(state["environment_state"]))
                self._env._seed = state["wrapper_seed"]
                self._env._needs_seed = state["needs_seed"]
            return state
        except (OSError, pickle.UnpicklingError, EOFError, KeyError) as exc:
            raise ValueError(f"Cannot load checkpoint {path}: {exc}") from exc

    def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix=None):
        _, path = self._latest_checkpoint()
        if path is None:
            return 0, 0
        state = self._load_checkpoint(path, restore_environment=True)
        self._agent.eval_mode = False
        return state["current_iteration"] + 1, state["total_steps"]

    def _initialize_metrics(self):
        self._stats = defaultdict(list)
        self._env.reset_metrics()

    def _write_metrics(self, step, suffix):
        for name, values in self._stats.items():
            if values:
                self._summary_writer.add_scalar(f"{name}/{suffix}", float(np.mean(values)), step)
        self._env.write_metrics(
            lambda tag, value: self._summary_writer.add_scalar(f"{tag}/{suffix}", value, step)
        )
        self._summary_writer.flush()

    def close(self):
        if not self._closed:
            if self._episode_writer is not None:
                self._episode_writer.close()
            self._summary_writer.close()
            self._env.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


@gin.configurable
class TrainRunner(Runner):
    def __init__(
        self, max_training_steps=250000, num_iterations=100, checkpoint_frequency=1, **kwargs
    ):
        if min(max_training_steps, num_iterations, checkpoint_frequency) < 1:
            raise ValueError("Training limits must be positive")
        super().__init__(**kwargs)
        self._max_training_steps = max_training_steps
        self._num_iterations = num_iterations
        self._checkpoint_frequency = checkpoint_frequency
        self._output_dir = self._base_dir / "train"
        self._checkpoint_dir = self._output_dir / "checkpoints"
        self._set_up(False)

    def run_experiment(self):
        try:
            start, total_steps = self._initialize_checkpointer_and_maybe_resume()
            for iteration in range(start, self._num_iterations):
                self._initialize_metrics()
                steps = 0
                while steps < self._max_training_steps:
                    length, _ = self._run_one_episode()
                    steps += length
                    total_steps += length
                self._write_metrics(total_steps, "train")
                if (
                    iteration % self._checkpoint_frequency == 0
                    or iteration == self._num_iterations - 1
                ):
                    self._checkpoint_experiment(iteration, total_steps)
            return total_steps
        finally:
            self.close()


@gin.configurable
class EvalRunner(Runner):
    def __init__(
        self,
        max_eval_episodes=5,
        test_mode=False,
        min_interval_secs=1,
        train_base_dir=None,
        wait_for_new=False,
        timeout_secs=60,
        **kwargs,
    ):
        if max_eval_episodes < 1 or min_interval_secs <= 0 or timeout_secs < 0:
            raise ValueError("Invalid evaluation limits")
        super().__init__(**kwargs)
        self._max_eval_episodes = max_eval_episodes
        self._test_mode = test_mode
        self._min_interval_secs = min_interval_secs
        self._wait_for_new = wait_for_new
        self._timeout_secs = timeout_secs
        self._output_dir = self._base_dir / f"eval_{max_eval_episodes}"
        self._checkpoint_dir = Path(train_base_dir or self._base_dir) / "train" / "checkpoints"
        self._set_up(True)

    def run_experiment(self):
        deadline, previous = time.monotonic() + self._timeout_secs, None
        results = []
        try:
            while True:
                version, path = self._latest_checkpoint()
                if path is not None and version != previous:
                    state = self._load_checkpoint(path)
                    self._agent.eval_mode = True
                    self._env.reset_sampler()
                    self._initialize_metrics()
                    results = [self._run_one_episode()[1] for _ in range(self._max_eval_episodes)]
                    self._write_metrics(state["total_steps"], "eval")
                    (self._output_dir / f"returns_{state['total_steps']}.json").write_text(
                        json.dumps(results)
                    )
                    previous = version
                    if self._test_mode or not self._wait_for_new:
                        return results
                elif not self._wait_for_new:
                    raise FileNotFoundError(f"No checkpoint in {self._checkpoint_dir}")
                if time.monotonic() >= deadline:
                    if results:
                        return results
                    raise TimeoutError("Timed out waiting for an evaluation checkpoint")
                time.sleep(min(self._min_interval_secs, max(0, deadline - time.monotonic())))
        finally:
            self.close()
