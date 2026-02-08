#!/usr/bin/env python3
"""rlbot - A GitHub bot for managing PRs in the torchrl repository.

Triggered by PR comments starting with `@rlbot`. Supports commands:
  - merge: Merge a PR (or ghstack) using ghstack land or gh pr merge
  - rebase: Rebase a PR onto a target branch

Inspired by PyTorch's @pytorchbot.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def gh(*args: str, **kwargs) -> subprocess.CompletedProcess:
    """Run a ``gh`` CLI command and return the result."""
    return subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )


def git(*args: str, **kwargs) -> subprocess.CompletedProcess:
    """Run a ``git`` command and return the result."""
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )


def post_comment(repo: str, pr_number: int, body: str) -> None:
    """Post a comment on the given PR."""
    gh("pr", "comment", str(pr_number), "--repo", repo, "--body", body)


def react_to_comment(repo: str, comment_id: int, reaction: str) -> None:
    """Add a reaction to a comment."""
    try:
        gh(
            "api",
            f"repos/{repo}/issues/comments/{comment_id}/reactions",
            "-f",
            f"content={reaction}",
            "--silent",
        )
    except subprocess.CalledProcessError:
        # Non-critical – don't fail if the reaction can't be added
        pass


def get_pr_info(repo: str, pr_number: int) -> dict:
    """Fetch PR metadata via ``gh``."""
    result = gh(
        "pr",
        "view",
        str(pr_number),
        "--repo",
        repo,
        "--json",
        "headRefName,baseRefName,author,title,url,labels,mergeable,reviewDecision",
    )
    return json.loads(result.stdout)


def check_write_permission(repo: str, username: str) -> bool:
    """Return True if *username* has write (or higher) permission on *repo*."""
    try:
        result = gh(
            "api",
            f"repos/{repo}/collaborators/{username}/permission",
        )
        data = json.loads(result.stdout)
        return data.get("permission") in ("admin", "maintain", "write")
    except subprocess.CalledProcessError:
        return False


def is_ghstack_pr(head_branch: str) -> bool:
    """Detect whether the PR was created by ghstack."""
    return bool(re.match(r"^gh/[^/]+/\d+/head$", head_branch))


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@dataclass
class CommandContext:
    repo: str
    pr_number: int
    comment_id: int
    comment_author: str
    pr_info: dict = field(default_factory=dict)


def cmd_merge(ctx: CommandContext, args: argparse.Namespace) -> None:
    """Handle ``@rlbot merge``."""
    pr = ctx.pr_info
    head = pr["headRefName"]

    # Permission gate
    if not check_write_permission(ctx.repo, ctx.comment_author):
        post_comment(
            ctx.repo,
            ctx.pr_number,
            f"@{ctx.comment_author} you don't have write permission on this repository. "
            "Only collaborators with write access can merge PRs.",
        )
        return

    # Approval gate (unless forced)
    if not args.force:
        decision = pr.get("reviewDecision", "")
        if decision != "APPROVED":
            post_comment(
                ctx.repo,
                ctx.pr_number,
                f"@{ctx.comment_author} this PR has not been approved yet "
                f"(current status: **{decision or 'REVIEW_REQUIRED'}**). "
                "Use `@rlbot merge -f 'reason'` to force merge.",
            )
            return

    if is_ghstack_pr(head):
        _merge_ghstack(ctx, args)
    else:
        _merge_regular(ctx, args)


def _merge_ghstack(ctx: CommandContext, args: argparse.Namespace) -> None:
    """Merge a ghstack PR using ``ghstack land``."""
    pr = ctx.pr_info
    url = pr["url"]

    post_comment(
        ctx.repo,
        ctx.pr_number,
        f"Merging ghstack PR via `ghstack land` (requested by @{ctx.comment_author}).\n\n"
        + (f"Force reason: {args.force}\n" if args.force else ""),
    )

    try:
        subprocess.run(
            ["ghstack", "land", url],
            capture_output=True,
            text=True,
            check=True,
        )
        post_comment(ctx.repo, ctx.pr_number, "ghstack land completed successfully.")
    except subprocess.CalledProcessError as exc:
        post_comment(
            ctx.repo,
            ctx.pr_number,
            f"ghstack land **failed**.\n\n```\n{exc.stderr or exc.stdout}\n```",
        )
        sys.exit(1)


def _merge_regular(ctx: CommandContext, args: argparse.Namespace) -> None:
    """Merge a regular (non-ghstack) PR using ``gh pr merge``."""
    merge_args = [
        "pr",
        "merge",
        str(ctx.pr_number),
        "--repo",
        ctx.repo,
        "--squash",
        "--delete-branch",
    ]

    msg_parts = [f"Merging PR (requested by @{ctx.comment_author})."]
    if args.force:
        msg_parts.append(f"Force reason: {args.force}")
        merge_args.append("--admin")

    post_comment(ctx.repo, ctx.pr_number, "\n".join(msg_parts))

    try:
        gh(*merge_args)
        post_comment(ctx.repo, ctx.pr_number, "PR merged successfully.")
    except subprocess.CalledProcessError as exc:
        post_comment(
            ctx.repo,
            ctx.pr_number,
            f"Merge **failed**.\n\n```\n{exc.stderr or exc.stdout}\n```",
        )
        sys.exit(1)


def cmd_rebase(ctx: CommandContext, args: argparse.Namespace) -> None:
    """Handle ``@rlbot rebase``."""
    pr = ctx.pr_info
    head = pr["headRefName"]
    target_branch = args.branch

    # Permission gate
    if not check_write_permission(ctx.repo, ctx.comment_author):
        post_comment(
            ctx.repo,
            ctx.pr_number,
            f"@{ctx.comment_author} you don't have write permission on this repository. "
            "Only collaborators with write access can rebase PRs.",
        )
        return

    post_comment(
        ctx.repo,
        ctx.pr_number,
        f"Rebasing `{head}` onto `{target_branch}` (requested by @{ctx.comment_author}).",
    )

    try:
        git("fetch", "origin", target_branch)
        git("fetch", "origin", head)
        git("checkout", head)
        git("rebase", f"origin/{target_branch}")
        git("push", "origin", head, "--force-with-lease")
        post_comment(
            ctx.repo,
            ctx.pr_number,
            f"Rebase onto `{target_branch}` completed successfully.",
        )
    except subprocess.CalledProcessError as exc:
        # Abort any in-progress rebase
        try:
            git("rebase", "--abort")
        except subprocess.CalledProcessError:
            pass
        post_comment(
            ctx.repo,
            ctx.pr_number,
            f"Rebase **failed**.\n\n```\n{exc.stderr or exc.stdout}\n```",
        )
        sys.exit(1)


def cmd_help(ctx: CommandContext, _args: argparse.Namespace) -> None:
    """Handle ``@rlbot help``."""
    post_comment(ctx.repo, ctx.pr_number, HELP_TEXT)


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

HELP_TEXT = """\
## @rlbot Help

```
usage: @rlbot {merge,rebase,help}
```

### `merge`
Merge a PR. For ghstack PRs, uses `ghstack land`; otherwise uses `gh pr merge --squash`.

```
@rlbot merge [-f MESSAGE]
```

| Flag | Description |
|------|-------------|
| `-f`, `--force` | Force merge with a reason (bypasses approval check, uses `--admin`) |

### `rebase`
Rebase the PR branch onto a target branch.

```
@rlbot rebase [-b BRANCH]
```

| Flag | Description |
|------|-------------|
| `-b`, `--branch` | Target branch (default: `main`) |

### `help`
Show this help message.

```
@rlbot help
```
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="@rlbot", add_help=False)
    sub = parser.add_subparsers(dest="command")

    # merge
    merge_p = sub.add_parser("merge", add_help=False)
    merge_p.add_argument(
        "-f",
        "--force",
        type=str,
        default=None,
        help="Force merge with a reason (bypasses approval gate)",
    )

    # rebase
    rebase_p = sub.add_parser("rebase", add_help=False)
    rebase_p.add_argument(
        "-b",
        "--branch",
        type=str,
        default="main",
        help="Branch to rebase onto (default: main)",
    )

    # help
    sub.add_parser("help", add_help=False)

    return parser


COMMAND_HANDLERS = {
    "merge": cmd_merge,
    "rebase": cmd_rebase,
    "help": cmd_help,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_command(comment_body: str) -> list[str] | None:
    """Extract the @rlbot command tokens from a comment body.

    Scans for the first line that starts with ``@rlbot`` (ignoring leading
    whitespace) and returns the tokens after ``@rlbot``.
    """
    for line in comment_body.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("@rlbot"):
            # Remove the "@rlbot" prefix and tokenise the rest
            rest = stripped[len("@rlbot") :].strip()
            if not rest:
                return []
            return rest.split()
    return None


def main() -> None:
    # Read event payload
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        sys.stderr.write(
            "GITHUB_EVENT_PATH not set – are you running inside GitHub Actions?\n"
        )
        sys.exit(1)

    with open(event_path) as f:
        event = json.load(f)

    repo = os.environ["GITHUB_REPOSITORY"]

    comment_body = event["comment"]["body"]
    comment_id = event["comment"]["id"]
    comment_author = event["comment"]["user"]["login"]
    pr_number = event["issue"]["number"]

    tokens = parse_command(comment_body)
    if tokens is None:
        # No @rlbot command found – nothing to do
        return

    # Acknowledge the command
    react_to_comment(repo, comment_id, "+1")

    parser = build_parser()

    if not tokens:
        # Bare "@rlbot" with no command
        ctx = CommandContext(
            repo=repo,
            pr_number=pr_number,
            comment_id=comment_id,
            comment_author=comment_author,
        )
        cmd_help(ctx, argparse.Namespace())
        return

    try:
        args = parser.parse_args(tokens)
    except SystemExit:
        ctx = CommandContext(
            repo=repo,
            pr_number=pr_number,
            comment_id=comment_id,
            comment_author=comment_author,
        )
        post_comment(
            repo,
            pr_number,
            f"@{comment_author} I couldn't parse that command. "
            f"Run `@rlbot help` to see available commands.\n\n"
            f"Input: `@rlbot {' '.join(tokens)}`",
        )
        return

    handler = COMMAND_HANDLERS.get(args.command)
    if handler is None:
        ctx = CommandContext(
            repo=repo,
            pr_number=pr_number,
            comment_id=comment_id,
            comment_author=comment_author,
        )
        cmd_help(ctx, args)
        return

    pr_info = get_pr_info(repo, pr_number)

    ctx = CommandContext(
        repo=repo,
        pr_number=pr_number,
        comment_id=comment_id,
        comment_author=comment_author,
        pr_info=pr_info,
    )

    handler(ctx, args)


if __name__ == "__main__":
    main()
