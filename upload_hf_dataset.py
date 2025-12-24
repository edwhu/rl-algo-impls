"""
upload_hf_dataset.py

Upload a local folder to a Hugging Face Hub repo (typically a dataset repo).

This is a thin, editable wrapper around `huggingface_hub.upload_folder()` that:
- Lets you select include/exclude patterns
- Can do a dry-run to show what would be uploaded
- Preserves folder structure by default (train/val/test, etc.)

Example (your CoinRun hard agent episodes):

python3 upload_hf_dataset.py \
  --folder-path /ephemeral/datasets/coinrun_hard_agent_episodes \
  --repo-id edwhu/coinrun_hard_agent \
  --repo-type dataset \
  --allow "train/**/*.array_record" \
  --allow "val/**/*.array_record" \
  --allow "test/**/*.array_record" \
  --allow "metadata*.json"
"""

from __future__ import annotations

import argparse
import os
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Iterable, List, Sequence


def _list_all_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*") if p.is_file()]


def _matches_any_pattern(rel_posix: str, patterns: Sequence[str]) -> bool:
    return any(fnmatchcase(rel_posix, pat) for pat in patterns)


def _collect_matches(root: Path, patterns: Sequence[str]) -> List[Path]:
    """
    Match files using the same style of matching as huggingface_hub allow/ignore patterns
    (fnmatch-style against POSIX paths relative to `root`).
    """
    if not patterns:
        return _list_all_files(root)

    out: List[Path] = []
    for p in _list_all_files(root):
        rel = p.relative_to(root).as_posix()
        if _matches_any_pattern(rel, patterns):
            out.append(p)
    return out


def _filter_excludes(paths: Iterable[Path], exclude_patterns: Sequence[str]) -> List[Path]:
    if not exclude_patterns:
        return list(paths)
    excluded: List[Path] = []
    out: List[Path] = []
    for p in paths:
        rel = p.as_posix()
        hit = _matches_any_pattern(rel, exclude_patterns)
        if hit:
            excluded.append(p)
        else:
            out.append(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a folder to Hugging Face Hub.")
    parser.add_argument(
        "--folder-path",
        required=True,
        type=str,
        help="Local folder to upload (e.g. /ephemeral/datasets/coinrun_hard_agent_episodes).",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        type=str,
        help="Hub repo id, e.g. edwhu/coinrun_hard_agent",
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
        type=str,
        choices=["dataset", "model", "space"],
        help="Hub repo type (default: dataset).",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        type=str,
        help="Optional subdirectory inside the repo to place uploads (e.g. 'data').",
    )
    parser.add_argument(
        "--large",
        action="store_true",
        help=(
            "Use HfApi().upload_large_folder(...) instead of upload_folder(). "
            "Recommended for many files / large uploads."
        ),
    )
    parser.add_argument(
        "--num-workers",
        default=16,
        type=int,
        help="Parallel workers for --large mode (default: 16).",
    )
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        help=(
            "Glob pattern(s) to include, relative to folder-path. "
            "Repeat flag to add multiple. If omitted, uploads everything under folder-path."
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob pattern(s) to exclude, relative to folder-path. Repeatable.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload dataset files",
        type=str,
        help="Commit message shown on the Hub.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded, but do not upload.",
    )
    parser.add_argument(
        "--list-limit",
        default=50,
        type=int,
        help="Max files to print in dry-run mode (default: 50).",
    )
    args = parser.parse_args()

    root = Path(args.folder_path).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"--folder-path does not exist or is not a directory: {root}")

    allow_patterns: List[str] = list(args.allow)
    exclude_patterns: List[str] = list(args.exclude)

    if allow_patterns:
        candidates = _collect_matches(root, allow_patterns)
    else:
        candidates = _list_all_files(root)

    # Convert to paths relative to root for exclude matching + printing.
    rel_candidates = [p.relative_to(root) for p in candidates]
    rel_filtered = _filter_excludes(rel_candidates, exclude_patterns)

    print("Upload config:")
    print(f"  folder_path = {root}")
    print(f"  repo_id     = {args.repo_id}")
    print(f"  repo_type   = {args.repo_type}")
    print(f"  path_in_repo= {args.path_in_repo}")
    print(f"  allow       = {allow_patterns if allow_patterns else ['<ALL FILES>']}")
    print(f"  exclude     = {exclude_patterns if exclude_patterns else ['<NONE>']}")
    print(f"  num_files   = {len(rel_filtered)}")

    if args.dry_run:
        print("\nDry-run: showing files that would be uploaded:")
        shown = 0
        for p in sorted(rel_filtered):
            print(f"  - {p.as_posix()}")
            shown += 1
            if shown >= int(args.list_limit):
                remaining = len(rel_filtered) - shown
                if remaining > 0:
                    print(f"  ... ({remaining} more)")
                break
        print("\nDry-run complete (no upload performed).")
        return

    # Import only when actually uploading (keeps dry-run lightweight).
    if args.large:
        if args.path_in_repo:
            raise SystemExit(
                "--large mode does not support --path-in-repo. "
                "To upload into a subfolder, create that subfolder structure locally inside --folder-path."
            )
        if args.commit_message and args.commit_message != "Upload dataset files":
            print(
                "Note: --large mode does not support a custom --commit-message "
                "(upload_large_folder may create multiple commits). Ignoring --commit-message."
            )

        from huggingface_hub import HfApi

        api = HfApi()
        # NOTE: upload_large_folder does not currently return a CommitInfo object.
        api.upload_large_folder(
            repo_id=str(args.repo_id),
            repo_type=str(args.repo_type),
            folder_path=str(root),
            allow_patterns=allow_patterns if allow_patterns else None,
            ignore_patterns=exclude_patterns if exclude_patterns else None,
            num_workers=int(args.num_workers),
        )
        print("\nUpload complete (large-folder mode).")
    else:
        from huggingface_hub import upload_folder

        # huggingface_hub will use cached login (HF_TOKEN / ~/.cache/huggingface / etc.)
        commit_info = upload_folder(
            folder_path=str(root),
            repo_id=str(args.repo_id),
            repo_type=str(args.repo_type),
            path_in_repo=args.path_in_repo,
            allow_patterns=allow_patterns if allow_patterns else None,
            ignore_patterns=exclude_patterns if exclude_patterns else None,
            commit_message=str(args.commit_message),
        )

        print("\nUpload complete.")
        commit_url = getattr(commit_info, "commit_url", None)
        commit_oid = getattr(commit_info, "oid", None)
        if commit_url:
            print(f"  commit_url: {commit_url}")
        if commit_oid:
            print(f"  commit_oid: {commit_oid}")


if __name__ == "__main__":
    # Avoid HF telemetry env warnings if user wants quiet logs.
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    main()
