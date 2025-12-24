import os
import shutil

from dataclasses import dataclass
from typing import NamedTuple, Optional

from runner.env import make_eval_env
from runner.config import Config, EnvHyperparams, RunArgs
from runner.running_utils import (
    load_hyperparams,
    set_seeds,
    get_device,
    make_policy,
)
from shared.callbacks.eval_callback import evaluate
from shared.policy.policy import Policy
from shared.stats import EpisodesStats


@dataclass
class EvalArgs(RunArgs):
    render: bool = True
    best: bool = True
    n_envs: Optional[int] = 1
    n_episodes: int = 3
    deterministic_eval: Optional[bool] = None
    no_print_returns: bool = False
    wandb_run_path: Optional[str] = None


class Evaluation(NamedTuple):
    policy: Policy
    stats: EpisodesStats
    config: Config


class LoadedEvalSetup(NamedTuple):
    policy: Policy
    env: object
    config: Config
    model_path: str
    deterministic: bool


def load_eval_setup(args: EvalArgs, root_dir: str) -> LoadedEvalSetup:
    """
    Load a policy + evaluation env using the same logic as enjoy/evaluate, without
    actually rolling out evaluation episodes.
    """
    if args.wandb_run_path:
        import wandb

        api = wandb.Api()
        run = api.run(args.wandb_run_path)
        hyperparams = run.config

        args.algo = hyperparams["algo"]
        args.env = hyperparams["env"]
        args.seed = hyperparams.get("seed", None)
        args.use_deterministic_algorithms = hyperparams.get(
            "use_deterministic_algorithms", True
        )

        config = Config(args, hyperparams, root_dir)
        model_path = config.model_dir_path(best=args.best, downloaded=True)

        model_archive_name = config.model_dir_name(best=args.best, extension=".zip")
        run.file(model_archive_name).download()
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        shutil.unpack_archive(model_archive_name, model_path)
        os.remove(model_archive_name)
    else:
        hyperparams = load_hyperparams(args.algo, args.env, root_dir)

        config = Config(args, hyperparams, root_dir)
        model_path = config.model_dir_path(best=args.best)

    print(args)

    # set_seeds(args.seed, args.use_deterministic_algorithms)
    env = make_eval_env(
        config,
        EnvHyperparams(**config.env_hyperparams),
        override_n_envs=args.n_envs,
        render=args.render,
        normalize_load_path=model_path,
    )

    # just call reset, and save image to check if env is working. 
    # import imageio.v2 as imageio
    # from datetime import datetime
    # import sys
    # # from procgen.env import ProcgenEnv
    # from procgen import ProcgenGym3Env
    # import numpy as np
    # start_level = np.random.randint(0, 1000000)
    # print(f"start_level: {start_level}")
    # env = ProcgenGym3Env(num=1, env_name="coinrun", start_level=start_level)
    # # env = ProcgenEnv(num_envs=1, env_name="coinrun", start_level=start_level, use_generated_assets=True)
    # # obs = env.reset()
    # rew, obs, first = env.observe()
    # # print(f"obs shape: {obs.shape}") # (32, 3, 64, 64)
    # imageio.imwrite(f"obs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", obs['rgb'].squeeze())
    # # imageio.imwrite(f"obs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", obs[0].transpose(1, 2, 0))
    # print(f"saved obs image to obs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    # sys.exit(0)

    device = get_device(config.device, env)  # type: ignore[arg-type]
    policy = make_policy(
        args.algo,
        env,  # type: ignore[arg-type]
        device,
        load_path=model_path,
        **config.policy_hyperparams,
    ).eval()

    deterministic = (
        args.deterministic_eval
        if args.deterministic_eval is not None
        else config.eval_params.get("deterministic", True)
    )
    return LoadedEvalSetup(
        policy=policy,
        env=env,
        config=config,
        model_path=model_path,
        deterministic=deterministic,
    )


def evaluate_model(args: EvalArgs, root_dir: str) -> Evaluation:
    setup = load_eval_setup(args, root_dir)
    return Evaluation(
        setup.policy,
        evaluate(
            setup.env,  # type: ignore[arg-type]
            setup.policy,
            args.n_episodes,
            render=args.render,
            deterministic=setup.deterministic,
            print_returns=not args.no_print_returns,
        ),
        setup.config,
    )
