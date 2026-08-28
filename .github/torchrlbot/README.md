# <img src="https://raw.githubusercontent.com/pytorch/rl/main/.github/torchrlbot/icon.png" width="30"> @torchrlbot

A GitHub bot for managing Pull Requests in the **torchrl** repository.
Inspired by PyTorch's [@pytorchbot](https://github.com/pytorch/pytorch/wiki/Bot-commands).

## How it works

`@torchrlbot` is powered by a GitHub Actions workflow (`.github/workflows/torchrlbot.yml`)
that triggers on PR comments containing `@torchrlbot`. A Python script
(`.github/torchrlbot/torchrlbot.py`) parses the command and executes the appropriate
action.

## Usage

Add a comment on any PR starting with `@torchrlbot` followed by a command:

```
@torchrlbot <command> [options]
```

### Commands

#### `merge`

Merge a PR. If the PR was created with **ghstack**, the bot uses `ghstack land`;
otherwise it performs a squash merge via `gh pr merge --squash`.

```
@torchrlbot merge [-f MESSAGE]
```

| Flag | Description |
|------|-------------|
| `-f`, `--force` `MESSAGE` | Force merge with a reason. Bypasses the approval check and uses `--admin` for non-ghstack PRs. |

**Examples:**

```
@torchrlbot merge
@torchrlbot merge -f 'Trivial doc fix, tests passing'
```

**Requirements:**
- The commenter must have **write** (or higher) permission on the repository.
- The PR must be **approved** unless `-f` is used.
- **Admins and maintainers** can merge without approval (the approval gate is
  skipped automatically).

#### `rebase`

Rebase the PR branch onto a target branch (default: `main`).

```
@torchrlbot rebase [-b BRANCH]
```

| Flag | Description |
|------|-------------|
| `-b`, `--branch` `BRANCH` | Target branch to rebase onto (default: `main`). |

**Examples:**

```
@torchrlbot rebase
@torchrlbot rebase -b release/0.6
```

**Requirements:**
- The commenter must have **write** (or higher) permission on the repository.
- The PR branch must be in this repository. The workflow token is scoped to
  `pytorch/rl` and cannot push rebased commits to a contributor's fork.

#### `reviewer`

Request reviews from one or more repository collaborators.

```
@torchrlbot reviewer @user1 @user2
```

Reviewers can be space- or comma-separated, with or without a leading `@`.

**Examples:**

```
@torchrlbot reviewer @vmoens
@torchrlbot reviewer vmoens,albertbou92
```

**Requirements:**
- The commenter must be the **PR author** or have **write** (or higher)
  permission on the repository.
- Requested reviewers must be collaborators on the repository.
- The PR author is skipped automatically (GitHub does not allow self-review).

#### `help`

Display the help message with all available commands.

```
@torchrlbot help
```

## Permissions

The `merge` and `rebase` commands require the commenter to have **write** access
to the repository. The `reviewer` command can also be run by the PR author, so
external contributors can request a review on their own PR without being able to
modify other contributors' PRs. The bot checks permissions via the GitHub API
before executing any action.

The workflow grants only the permissions used by its commands:

- `contents: write` for same-repository rebases, branch deletion, and merges.
- `pull-requests: write` for PR metadata, merges, and review requests.
- `issues: write` for command acknowledgements and status comments.

The token belongs to the GitHub Actions app and is scoped to `pytorch/rl`; it
does not impersonate the commenter. Consequently, it cannot rebase fork branches,
and `merge --force` can only bypass repository rules that explicitly allow the
GitHub Actions app to bypass them.

## Architecture

```
.github/
├── torchrlbot/
│   ├── torchrlbot.py   # Command parser and handler
│   ├── icon.png        # Bot icon (64x64)
│   └── README.md       # This file
└── workflows/
    └── torchrlbot.yml  # GitHub Actions workflow (issue_comment trigger)
```

The workflow:

1. **Trigger**: An `issue_comment` event fires when someone comments on a PR.
2. **Filter**: The job only runs if the comment is on a PR and contains `@torchrlbot`.
3. **Parse**: `torchrlbot.py` reads the GitHub event payload, extracts the command
   line from the comment, and parses it with `argparse`.
4. **Validate**: The bot checks that the commenter is authorized for the command
   and that the PR meets any required conditions (e.g., approval status for
   merge).
5. **Execute**: The appropriate handler runs (`ghstack land`, `gh pr merge`,
   `git rebase`, etc.) and posts status comments back on the PR.

## Adding new commands

1. Add a handler function `cmd_<name>(ctx, args)` in `torchrlbot.py`.
2. Register a subparser in `build_parser()`.
3. Add the handler to the `COMMAND_HANDLERS` dict.
4. Update `HELP_TEXT` and this README.

## Secrets and tokens

The bot uses `GITHUB_TOKEN` (automatically provided by GitHub Actions) for all
API calls. No additional secrets are required for basic operation.

For `ghstack land`, the token is passed via `GH_TOKEN` which `ghstack` picks up
automatically.
