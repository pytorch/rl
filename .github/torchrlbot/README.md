# <img src="https://raw.githubusercontent.com/pytorch/rl/main/docs/source/_static/img/icon.png" width="30"> @torchrlbot

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

#### `help`

Display the help message with all available commands.

```
@torchrlbot help
```

## Permissions

All commands require the commenter to have **write** access to the repository.
This prevents external contributors from triggering merges or rebases. The bot
checks permissions via the GitHub API before executing any action.

## Architecture

```
.github/
├── torchrlbot/
│   ├── torchrlbot.py   # Command parser and handler
│   └── README.md       # This file
└── workflows/
    └── torchrlbot.yml  # GitHub Actions workflow (issue_comment trigger)
```

The workflow:

1. **Trigger**: An `issue_comment` event fires when someone comments on a PR.
2. **Filter**: The job only runs if the comment is on a PR and contains `@torchrlbot`.
3. **Parse**: `torchrlbot.py` reads the GitHub event payload, extracts the command
   line from the comment, and parses it with `argparse`.
4. **Validate**: The bot checks that the commenter has write permission and that
   the PR meets any required conditions (e.g., approval status for merge).
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
