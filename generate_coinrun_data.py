"""generate_coinrun_data.py
Generate a dataset of CoinRun episodes by rolling out a pretrained RL agent.

Episodes are saved in the same chunked `.array_record` format used by:
`dreamer4-jax-private/coinrun_data/generate_coinrun_dataset.py`.

Example:
  python generate_coinrun_data.py \
    --wandb-run-path=sgoodfriend/rl-algo-impls-benchmarks/vmjd3amn \
    --output-dir=/ephemeral/datasets/coinrun_agent_episodes
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tqdm import tqdm

from runner.evaluate import EvalArgs, load_eval_setup
from runner.running_utils import base_parser


def _quota(total: int, num_workers: int, worker_id: int) -> int:
    if num_workers <= 0:
        raise ValueError(f"num_workers must be > 0, got {num_workers}")
    if not (0 <= worker_id < num_workers):
        raise ValueError(f"worker_id must be in [0, {num_workers - 1}], got {worker_id}")
    base = total // num_workers
    rem = total % num_workers
    return base + (1 if worker_id < rem else 0)


def _as_hwc_uint8(frame: np.ndarray) -> np.ndarray:
    """
    Convert a single frame to HWC uint8.
    Expected input shapes:
    - HWC
    - CHW
    """
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    if frame.ndim != 3:
        raise ValueError(f"Expected a single frame with 3 dims, got shape {frame.shape}")

    # CHW -> HWC
    if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))

    return frame


def _get_num_actions(env: Any) -> int:
    space = getattr(env, "action_space", None)
    n = getattr(space, "n", None)
    if n is None:
        raise ValueError(f"Unsupported action_space for num_actions: {space}")
    return int(n)


def save_chunks(
    file_idx: int,
    chunks_per_file: int,
    output_dir: str,
    obs_chunks: List[np.ndarray],
    file_prefix: str = "",
    act_chunks: Optional[List[np.ndarray]] = None,
    rew_chunks: Optional[List[np.ndarray]] = None,
) -> Tuple[List[Dict[str, Any]], int, List[np.ndarray], Optional[List[np.ndarray]], Optional[List[np.ndarray]]]:
    """
    Writes `.array_record` files. Format matches dreamer4-jax-private/coinrun_data/generate_coinrun_dataset.py.
    """
    try:
        from array_record.python.array_record_module import (  # type: ignore[import-not-found]
            ArrayRecordWriter,
        )
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Missing dependency `array_record`. Install it (and its deps) to write .array_record datasets."
        ) from e

    os.makedirs(output_dir, exist_ok=True)

    metadata: List[Dict[str, Any]] = []
    while len(obs_chunks) >= chunks_per_file:
        chunk_batch = obs_chunks[:chunks_per_file]
        obs_chunks = obs_chunks[chunks_per_file:]

        act_chunk_batch: Optional[List[np.ndarray]] = None
        if act_chunks:
            act_chunk_batch = act_chunks[:chunks_per_file]
            act_chunks = act_chunks[chunks_per_file:]

        rew_chunk_batch: Optional[List[np.ndarray]] = None
        if rew_chunks:
            rew_chunk_batch = rew_chunks[:chunks_per_file]
            rew_chunks = rew_chunks[chunks_per_file:]

        episode_path = os.path.join(
            output_dir, f"{file_prefix}data_{file_idx:06d}.array_record"
        )
        writer = ArrayRecordWriter(str(episode_path), "group_size:1")
        seq_lens: List[int] = []
        for idx, chunk in enumerate(chunk_batch):
            seq_len = int(chunk.shape[0])
            seq_lens.append(seq_len)
            chunk_record: Dict[str, Any] = {
                "raw_video": chunk.tobytes(),
                "sequence_length": seq_len,
            }
            if act_chunk_batch is not None:
                assert len(chunk) == len(
                    act_chunk_batch[idx]
                ), f"Observation data length and action sequence length do not match: {len(chunk)} != {len(act_chunk_batch[idx])}"
                chunk_record["actions"] = act_chunk_batch[idx]
            if rew_chunk_batch is not None:
                assert len(chunk) == len(
                    rew_chunk_batch[idx]
                ), f"Observation data length and reward sequence length do not match: {len(chunk)} != {len(rew_chunk_batch[idx])}"
                chunk_record["rewards"] = rew_chunk_batch[idx]
            writer.write(pickle.dumps(chunk_record))
        writer.close()
        file_idx += 1
        metadata.append(
            {
                "path": episode_path,
                "num_chunks": len(chunk_batch),
                "avg_seq_len": float(np.mean(seq_lens)) if seq_lens else 0.0,
            }
        )
        print(f"Created {episode_path} with {len(chunk_batch)} video chunks")

    return metadata, file_idx, obs_chunks, act_chunks, rew_chunks


@dataclass
class Args:
    # Dataset sizing
    num_episodes_train: int = 90_000
    num_episodes_val: int = 4500
    num_episodes_test: int = 4500

    # Output
    output_dir: str = "datasets/coinrun_agent_episodes"

    # Episode / chunking
    min_episode_length: int = 32
    max_episode_length: int = 1_000
    chunk_size: int = 160
    chunks_per_file: int = 100

    # Repro / rollout
    seed: int = 0
    n_envs: int = 1


def _generate_split(
    split_name: str,
    num_episodes: int,
    setup: Any,
    args: Args,
    file_prefix: str,
) -> List[Dict[str, Any]]:
    env = setup.env
    policy = setup.policy
    num_actions = _get_num_actions(env)
    dummy_action_value = num_actions
    dummy_reward_value = np.nan

    episode_idx = 0
    episode_metadata: List[Dict[str, Any]] = []

    obs_chunks: List[np.ndarray] = []
    act_chunks: List[np.ndarray] = []
    rew_chunks: List[np.ndarray] = []
    file_idx = 0

    output_dir_split = os.path.join(args.output_dir, split_name)

    # Initialize vectorized episode buffers.
    obs = env.reset()
    # --- MINIMAL PROCGEN RESET DEBUG ---
    # import sys
    # print(f"[debug] cli/args seed={args.seed} split={split_name}")

    # # unwrap wrappers to try to find the underlying procgen gym3 env that has `.options`
    # _e = env
    # _chain = []
    # while True:
    #     _chain.append(type(_e).__name__)
    #     if hasattr(_e, "env"):
    #         _e = _e.env
    #     else:
    #         break
    # print("[debug] wrapper chain:", " -> ".join(_chain))

    # _opts = getattr(_e, "options", None)
    # if isinstance(_opts, dict):
    #     print("[debug] procgen options subset:", {
    #         k: _opts.get(k)
    #         for k in ["env_name", "rand_seed", "num_levels", "start_level", "use_sequential_levels", "distribution_mode"]
    #         if k in _opts
    #     })
    # else:
    #     print("[debug] could not find procgen .options dict on unwrapped env")

    # # assuming you already have `_e` from your unwrap code pointing to ProcgenGym3Env
    # _opts = getattr(_e, "options", None)
    # if isinstance(_opts, dict):
    #     print("[debug] restrict_themes:", _opts.get("restrict_themes"))
    #     print("[debug] use_backgrounds:", _opts.get("use_backgrounds"))
    #     print("[debug] use_generated_assets:", _opts.get("use_generated_assets"))
    #     print("[debug] use_monochrome_assets:", _opts.get("use_monochrome_assets"))

    # # often contains level seeds / other metadata if procgen exposes them
    # if hasattr(_e, "get_info"):
    #     infos = _e.get_info()
    #     print("[debug] get_info keys[0]:", sorted(list(infos[0].keys())) if infos else None)
    #     # If you see something like level_seed, print a few:
    #     for k in ["level_seed", "seed", "theme", "level", "start_level"]:
    #         if infos and k in infos[0]:
    #             print("[debug]", k, [infos[i].get(k) for i in range(min(8, len(infos)))])

    # # dump observations from reset
    # obs_np = np.asarray(obs)
    # n_envs = int(getattr(env, "num_envs", obs_np.shape[0]))
    # print(f"[debug] obs shape={obs_np.shape} dtype={obs_np.dtype} inferred n_envs={n_envs}")

    # # Save debug outputs per worker (file_prefix is like "w00_" or "w01_")
    # worker_subdir = file_prefix.rstrip("_") if file_prefix else "single_worker"
    # dbg_dir = os.path.join(output_dir_split, "_debug_reset", worker_subdir)
    # os.makedirs(dbg_dir, exist_ok=True)

    # # save raw batch too (exactly what your code sees)
    # np.savez_compressed(os.path.join(dbg_dir, "reset_obs.npz"), obs=obs_np)

    # # write per-env pngs (handles CHW from TransposeImageObservation)
    # import imageio.v2 as imageio
    # for i in range(min(n_envs, 16)):
    #     img = obs_np[i]
    #     if img.ndim == 3 and img.shape[0] in (1, 3):  # CHW -> HWC
    #         img = np.transpose(img, (1, 2, 0))
    #     imageio.imwrite(os.path.join(dbg_dir, f"reset_env{i:02d}.png"), img.astype(np.uint8))

    # print(f"[debug] worker={worker_subdir} wrote {min(n_envs, 16)} pngs + reset_obs.npz to {dbg_dir}")

    # print("[debug] exiting after reset debug")
    # sys.exit(0)
    # --- END MINIMAL PROCGEN RESET DEBUG ---

    if not isinstance(obs, np.ndarray):
        raise ValueError(f"Expected env.reset() to return np.ndarray, got {type(obs)}")
    n_envs = int(getattr(env, "num_envs", obs.shape[0]))
    if obs.shape[0] != n_envs:
        raise ValueError(f"Expected obs leading dim to be n_envs={n_envs}, got {obs.shape}")

    # Dreamer indexing per env:
    # - store states s[0..T] (T+1 frames)
    # - actions[0] is dummy, rewards[0] is NaN
    observations_seq: List[List[np.ndarray]] = [
        [_as_hwc_uint8(obs[i])[None, ...]] for i in range(n_envs)
    ]
    actions_seq: List[List[np.ndarray]] = [
        [np.array([dummy_action_value], dtype=np.int64)] for _ in range(n_envs)
    ]
    rewards_seq: List[List[np.ndarray]] = [
        [np.array([dummy_reward_value], dtype=np.float32)] for _ in range(n_envs)
    ]
    episode_obs_chunks: List[List[List[np.ndarray]]] = [[] for _ in range(n_envs)]
    episode_act_chunks: List[List[List[np.ndarray]]] = [[] for _ in range(n_envs)]
    episode_rew_chunks: List[List[List[np.ndarray]]] = [[] for _ in range(n_envs)]
    steps: List[int] = [0 for _ in range(n_envs)]

    # Progress / stats (accepted episodes only).
    accepted_steps_sum = 0
    rejected_episodes = 0
    successful_episodes = 0
    pbar = tqdm(total=num_episodes, desc=f"Generating {split_name}", unit="ep")

    # Roll out asynchronously: each env index i is its own episode stream.
    while episode_idx < num_episodes:
        act = policy.act(obs, deterministic=False)
        obs_next, rew, done, infos = env.step(act)

        if not isinstance(obs_next, np.ndarray):
            raise ValueError(
                f"Expected env.step() to return obs as np.ndarray, got {type(obs_next)}"
            )
        if obs_next.shape[0] != n_envs:
            raise ValueError(
                f"Expected obs_next leading dim n_envs={n_envs}, got {obs_next.shape}"
            )

        # Normalize done to a boolean vector of shape (n_envs,).
        done_vec = np.asarray(done, dtype=bool).reshape(-1)
        if done_vec.shape[0] != n_envs:
            raise ValueError(f"Expected done to have shape ({n_envs},), got {done_vec.shape}")

        # Normalize infos to a list[dict] of length n_envs if possible.
        infos_list: List[dict]
        if isinstance(infos, list):
            infos_list = infos  # type: ignore[assignment]
        else:
            # Some envs may return a dict or other structure; fall back to empty dicts.
            infos_list = [{} for _ in range(n_envs)]

        for i in range(n_envs):
            # For SB3 VecEnv, the terminal observation is often stored in infos[i].
            next_obs_i = obs_next[i]
            if done_vec[i] and i < len(infos_list):
                terminal_obs = infos_list[i].get("terminal_observation", None)
                if terminal_obs is not None:
                    next_obs_i = terminal_obs

            # Append transition (a_t, r_t, s_{t+1}) to per-env buffers.
            a_i = np.asarray(act[i]).copy()
            if a_i.shape == ():
                a_i = np.array([a_i.item()], dtype=np.int64)
            r_i = np.asarray(rew[i], dtype=np.float32).copy()
            if r_i.shape == ():
                r_i = np.array([r_i.item()], dtype=np.float32)

            actions_seq[i].append(a_i)
            rewards_seq[i].append(r_i)
            observations_seq[i].append(_as_hwc_uint8(next_obs_i)[None, ...])
            steps[i] += 1

            if not (
                len(observations_seq[i]) == len(actions_seq[i]) == len(rewards_seq[i])
            ):
                raise RuntimeError(
                    "Invariant violated: observations/actions/rewards must have equal length. "
                    f"env={i} obs={len(observations_seq[i])} act={len(actions_seq[i])} rew={len(rewards_seq[i])}"
                )

            # Flush fixed-size chunks.
            if len(observations_seq[i]) == args.chunk_size:
                episode_obs_chunks[i].append(observations_seq[i])
                episode_act_chunks[i].append(actions_seq[i])
                episode_rew_chunks[i].append(rewards_seq[i])
                # Overlap last state: next chunk starts at the current state.
                last_obs = observations_seq[i][-1]
                observations_seq[i] = [last_obs]
                actions_seq[i] = [np.array([dummy_action_value], dtype=np.int64)]
                rewards_seq[i] = [np.array([dummy_reward_value], dtype=np.float32)]

        # Finalize any envs that ended this step.
        for i in range(n_envs):
            if not done_vec[i]:
                continue

            episode_len = steps[i]
            if episode_len >= args.min_episode_length:
                # Only save a leftover chunk if it includes at least one transition
                # (i.e., at least 2 states).
                if len(observations_seq[i]) >= 2:
                    if len(observations_seq[i]) < args.chunk_size:
                        print(
                            f"Warning: Inconsistent chunk_sizes. Episode has {len(observations_seq[i])} frames, "
                            f"which is smaller than the requested chunk_size: {args.chunk_size}. "
                            "This might lead to performance degradation during training."
                        )
                    episode_obs_chunks[i].append(observations_seq[i])
                    episode_act_chunks[i].append(actions_seq[i])
                    episode_rew_chunks[i].append(rewards_seq[i])

                obs_chunks_data = [
                    np.concatenate(seq, axis=0).astype(np.uint8)
                    for seq in episode_obs_chunks[i]
                ]
                act_chunks_data = [
                    np.concatenate(act_seq, axis=0) for act_seq in episode_act_chunks[i]
                ]
                rew_chunks_data = [
                    np.concatenate(rew_seq, axis=0) for rew_seq in episode_rew_chunks[i]
                ]

                obs_chunks.extend(obs_chunks_data)
                act_chunks.extend(act_chunks_data)
                rew_chunks.extend(rew_chunks_data)

                # Check if episode was successful (reward == 10)
                # episode_rew_chunks[i] already contains all rewards (including rewards_seq[i] appended at line 371)
                if episode_rew_chunks[i]:
                    all_rewards = [np.concatenate(rew_seq) for rew_seq in episode_rew_chunks[i]]
                    episode_rewards = np.concatenate(all_rewards)
                    is_successful = np.any(episode_rewards == 10.0)
                    if is_successful:
                        successful_episodes += 1

                ep_metadata, file_idx, obs_chunks, act_chunks, rew_chunks = save_chunks(
                    file_idx,
                    args.chunks_per_file,
                    output_dir_split,
                    obs_chunks,
                    file_prefix=file_prefix,
                    act_chunks=act_chunks,
                    rew_chunks=rew_chunks,
                )
                episode_metadata.extend(ep_metadata)

                print(f"Episode {episode_idx} completed, length: {episode_len}.")
                episode_idx += 1
                accepted_steps_sum += int(episode_len)
                avg_len = accepted_steps_sum / max(episode_idx, 1)
                success_rate = (successful_episodes / episode_idx * 100) if episode_idx > 0 else 0.0
                pbar.update(1)
                pbar.set_postfix(
                    avg_episode_len=f"{avg_len:.1f}",
                    rejected=rejected_episodes,
                    successful=successful_episodes,
                    success_rate=f"{success_rate:.1f}%"
                )
            else:
                print(f"Episode too short ({episode_len}), resampling...")
                rejected_episodes += 1
                avg_len = accepted_steps_sum / max(episode_idx, 1) if episode_idx > 0 else 0.0
                success_rate = (successful_episodes / episode_idx * 100) if episode_idx > 0 else 0.0
                pbar.set_postfix(
                    avg_episode_len=f"{avg_len:.1f}",
                    rejected=rejected_episodes,
                    successful=successful_episodes,
                    success_rate=f"{success_rate:.1f}%"
                )

            # Start a new episode for this env from the current observation.
            # Many vector envs auto-reset and return the next episode's initial obs
            # as obs_next[i] when done=True.
            observations_seq[i] = [_as_hwc_uint8(obs_next[i])[None, ...]]
            actions_seq[i] = [np.array([dummy_action_value], dtype=np.int64)]
            rewards_seq[i] = [np.array([dummy_reward_value], dtype=np.float32)]
            episode_obs_chunks[i] = []
            episode_act_chunks[i] = []
            episode_rew_chunks[i] = []
            steps[i] = 0

            if episode_idx >= num_episodes:
                break

        obs = obs_next

    pbar.close()

    avg_episode_len = accepted_steps_sum / max(num_episodes, 1)
    success_rate = (successful_episodes / num_episodes * 100) if num_episodes > 0 else 0.0
    print(
        f"Split {split_name}: generated {num_episodes} episodes "
        f"(rejected {rejected_episodes}, successful {successful_episodes}, "
        f"success_rate={success_rate:.2f}%), avg_episode_len={avg_episode_len:.2f}"
    )

    if len(obs_chunks) > 0:
        print(
            f"Warning: Dropping {len(obs_chunks)} chunks for consistent number of chunks per file.",
            "Consider changing the chunk_size and chunks_per_file parameters to prevent data-loss.",
        )

    print(f"Done generating {split_name} split")
    return episode_metadata, successful_episodes


def main() -> None:
    # Use base_parser to stay consistent with enjoy.py / EvalArgs
    parser = base_parser(multiple=False)
    parser.add_argument("--render", default=False, type=bool)
    parser.add_argument("--best", default=True, type=bool)
    parser.add_argument("--n-envs", dest="n_envs", default=1, type=int)
    parser.add_argument("--deterministic-eval", default=None, type=bool)
    parser.add_argument("--wandb-run-path", default=None, type=str)

    # Dataset options
    parser.add_argument("--output-dir", default=Args.output_dir, type=str)
    parser.add_argument("--num-episodes-train", default=Args.num_episodes_train, type=int)
    parser.add_argument("--num-episodes-val", default=Args.num_episodes_val, type=int)
    parser.add_argument("--num-episodes-test", default=Args.num_episodes_test, type=int)
    parser.add_argument("--total-episodes-train", default=None, type=int)
    parser.add_argument("--total-episodes-val", default=None, type=int)
    parser.add_argument("--total-episodes-test", default=None, type=int)
    parser.add_argument("--num-workers", default=1, type=int)
    parser.add_argument("--worker-id", default=0, type=int)
    parser.add_argument("--file-prefix", default=None, type=str)
    parser.add_argument("--min-episode-length", default=Args.min_episode_length, type=int)
    parser.add_argument("--max-episode-length", default=Args.max_episode_length, type=int)
    parser.add_argument("--chunk-size", default=Args.chunk_size, type=int)
    parser.add_argument("--chunks-per-file", default=Args.chunks_per_file, type=int)

    # Defaults
    parser.set_defaults(seed=Args.seed)
    parser.set_defaults(algo=["ppo"])
    cli = parser.parse_args()

    # Normalize base_parser list args
    cli.algo = cli.algo[0]
    cli.env = cli.env[0]

    num_workers = int(cli.num_workers)
    worker_id = int(cli.worker_id)
    if num_workers <= 0:
        raise ValueError(f"--num-workers must be > 0, got {num_workers}")
    if not (0 <= worker_id < num_workers):
        raise ValueError(f"--worker-id must be in [0, {num_workers - 1}], got {worker_id}")

    total_train = int(cli.total_episodes_train) if cli.total_episodes_train is not None else int(cli.num_episodes_train)
    total_val = int(cli.total_episodes_val) if cli.total_episodes_val is not None else int(cli.num_episodes_val)
    total_test = int(cli.total_episodes_test) if cli.total_episodes_test is not None else int(cli.num_episodes_test)

    num_episodes_train = _quota(total_train, num_workers, worker_id)
    num_episodes_val = _quota(total_val, num_workers, worker_id)
    num_episodes_test = _quota(total_test, num_workers, worker_id)

    file_prefix = str(cli.file_prefix) if cli.file_prefix is not None else f"w{worker_id:02d}_"

    args = Args(
        num_episodes_train=num_episodes_train,
        num_episodes_val=num_episodes_val,
        num_episodes_test=num_episodes_test,
        output_dir=str(cli.output_dir),
        min_episode_length=int(cli.min_episode_length),
        max_episode_length=int(cli.max_episode_length),
        chunk_size=int(cli.chunk_size),
        chunks_per_file=int(cli.chunks_per_file),
        seed=int(cli.seed + worker_id),
        n_envs=int(cli.n_envs),
    )

    assert (
        args.max_episode_length >= args.min_episode_length
    ), "Maximum episode length must be greater than or equal to minimum episode length."
    if args.min_episode_length < args.chunk_size:
        print(
            "Warning: Minimum episode length is smaller than chunk size. "
            "Note that episodes shorter than the chunk size will be discarded."
        )

    np.random.seed(args.seed + worker_id)

    # Filter cli args to only include EvalArgs fields
    eval_args_dict = {
        "algo": cli.algo,
        "env": cli.env,
        "seed": cli.seed,
        "use_deterministic_algorithms": getattr(cli, "use_deterministic_algorithms", True),
        "render": cli.render,
        "best": cli.best,
        "n_envs": cli.n_envs,
        "n_episodes": getattr(cli, "n_episodes", 3),  # Not used by this script, but EvalArgs requires it
        "deterministic_eval": cli.deterministic_eval,
        "no_print_returns": getattr(cli, "no_print_returns", False),
        "wandb_run_path": cli.wandb_run_path,
    }
    eval_args = EvalArgs(**eval_args_dict)
    setup = load_eval_setup(eval_args, os.path.dirname(__file__))
    num_actions = _get_num_actions(setup.env)

    # Force stochastic rollouts (per user request), regardless of config.
    print("Rolling out policy stochastically (deterministic=False).")

    train_episode_metadata, train_successful = _generate_split("train", args.num_episodes_train, setup, args, file_prefix=file_prefix)
    val_episode_metadata, val_successful = _generate_split("val", args.num_episodes_val, setup, args, file_prefix=file_prefix)
    test_episode_metadata, test_successful = _generate_split("test", args.num_episodes_test, setup, args, file_prefix=file_prefix)

    metadata: Dict[str, Any] = {
        "env": "coinrun",
        "algo": setup.config.algo,
        "wandb_run_path": eval_args.wandb_run_path,
        "best": bool(eval_args.best),
        "num_workers": num_workers,
        "worker_id": worker_id,
        "file_prefix": file_prefix,
        "total_episodes_train": total_train,
        "total_episodes_val": total_val,
        "total_episodes_test": total_test,
        "successful_episodes_train": train_successful,
        "successful_episodes_val": val_successful,
        "successful_episodes_test": test_successful,
        "success_rate_train": float(train_successful / args.num_episodes_train * 100) if args.num_episodes_train > 0 else 0.0,
        "success_rate_val": float(val_successful / args.num_episodes_val * 100) if args.num_episodes_val > 0 else 0.0,
        "success_rate_test": float(test_successful / args.num_episodes_test * 100) if args.num_episodes_test > 0 else 0.0,
        "n_envs": int(args.n_envs),
        "num_actions": num_actions,
        "dummy_action": num_actions,
        "dummy_reward": "nan",
        "num_episodes_train": args.num_episodes_train,
        "num_episodes_val": args.num_episodes_val,
        "num_episodes_test": args.num_episodes_test,
        "avg_episode_len_train": float(
            np.mean([ep["avg_seq_len"] for ep in train_episode_metadata])
        )
        if train_episode_metadata
        else 0.0,
        "avg_episode_len_val": float(
            np.mean([ep["avg_seq_len"] for ep in val_episode_metadata])
        )
        if val_episode_metadata
        else 0.0,
        "avg_episode_len_test": float(
            np.mean([ep["avg_seq_len"] for ep in test_episode_metadata])
        )
        if test_episode_metadata
        else 0.0,
        "min_episode_length": args.min_episode_length,
        "max_episode_length": args.max_episode_length,
        "chunk_size": args.chunk_size,
        "chunks_per_file": args.chunks_per_file,
        "frame_layout": "HWC_uint8",
        "indexing": "dreamer_t+1",
        "episode_metadata_train": train_episode_metadata,
        "episode_metadata_val": val_episode_metadata,
        "episode_metadata_test": test_episode_metadata,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    metadata_name = (
        f"metadata_worker_{worker_id:02d}.json" if num_workers > 1 else "metadata.json"
    )
    with open(os.path.join(args.output_dir, metadata_name), "w") as f:
        json.dump(metadata, f)

    print("Done generating dataset.")


if __name__ == "__main__":
    main()