# Copyright 2019 The RecSim Authors. Licensed under the Apache License, Version 2.0.
"""Run a small simulation or a train/evaluate experiment: python -m recsim.main."""

import argparse
from functools import partial
from importlib import import_module
from recsim.simulator.runner_lib import TrainRunner, EvalRunner, load_gin_configs


def create_agent(environment, eval_mode=False, summary_writer=None, agent_name="random", seed=0):
    common = dict(
        observation_space=environment.observation_space, action_space=environment.action_space
    )
    if agent_name == "random":
        from recsim.agents.random_agent import RandomAgent

        return RandomAgent(environment.action_space, random_seed=seed)
    if agent_name == "tabular_q":
        from recsim.agents.tabular_q_agent import TabularQAgent

        return TabularQAgent(**common, eval_mode=eval_mode, random_seed=seed)
    if agent_name == "full_slate_q":
        from recsim.agents.full_slate_q_agent import FullSlateQAgent

        return FullSlateQAgent(
            **common,
            eval_mode=eval_mode,
            summary_writer=summary_writer,
            seed=seed,
            batch_size=4,
            min_replay_history=4,
        )
    from recsim.agents.slate_decomp_q_agent import create_agent as create_slate_agent

    if agent_name == "slate_decomp_q":
        agent_name = "slate_greedy_greedy_q"
    return create_slate_agent(
        agent_name,
        **common,
        eval_mode=eval_mode,
        summary_writer=summary_writer,
        seed=seed,
        batch_size=4,
        min_replay_history=4,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", default="outputs/recsim")
    parser.add_argument("--agent_name", default="random")
    parser.add_argument(
        "--environment_name",
        choices=["interest_evolution", "interest_exploration", "long_term_satisfaction"],
        default="interest_evolution",
    )
    parser.add_argument("--mode", choices=["train", "eval", "both"], default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_candidates", type=int, default=5)
    parser.add_argument("--slate_size", type=int, default=2)
    parser.add_argument("--max_steps_per_episode", type=int, default=20)
    parser.add_argument("--max_training_steps", type=int, default=40)
    parser.add_argument("--num_iterations", type=int, default=2)
    parser.add_argument("--max_eval_episodes", type=int, default=2)
    parser.add_argument("--episode_log_file", default="")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--gin_files", action="append", default=[])
    parser.add_argument("--gin_bindings", action="append", default=[])
    args = parser.parse_args(argv)
    # Register only the selected backend before resolving Gin bindings.
    module = import_module(f"recsim.environments.{args.environment_name}")
    if args.agent_name == "full_slate_q":
        import_module("recsim.agents.full_slate_q_agent")
    elif args.agent_name == "tabular_q":
        import_module("recsim.agents.tabular_q_agent")
    elif args.agent_name not in {"random", "tabular_q"}:
        import_module("recsim.agents.slate_decomp_q_agent")
    load_gin_configs(args.gin_files, args.gin_bindings)
    config = dict(
        num_candidates=args.num_candidates,
        slate_size=args.slate_size,
        resample_documents=True,
        seed=args.seed,
    )
    factory = partial(create_agent, agent_name=args.agent_name, seed=args.seed)
    common = dict(
        base_dir=args.base_dir,
        create_agent_fn=factory,
        max_steps_per_episode=args.max_steps_per_episode,
        tensorboard=args.tensorboard,
    )
    if args.mode in ("train", "both"):
        TrainRunner(
            env=module.create_environment(config),
            max_training_steps=args.max_training_steps,
            num_iterations=args.num_iterations,
            episode_log_file=args.episode_log_file,
            **common,
        ).run_experiment()
    if args.mode in ("eval", "both"):
        returns = EvalRunner(
            env=module.create_environment(config),
            max_eval_episodes=args.max_eval_episodes,
            **common,
        ).run_experiment()
        print(f"Evaluation returns: {returns}")


if __name__ == "__main__":
    main()
