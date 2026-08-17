from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).parents[2]
_BOT_PATH = _REPO_ROOT / ".github" / "torchrlbot" / "torchrlbot.py"


def _load_torchrlbot():
    spec = importlib.util.spec_from_file_location("torchrlbot", _BOT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


torchrlbot = _load_torchrlbot()


def _context(*, comment_author: str, pr_author: str = "contributor", **pr_info):
    return torchrlbot.CommandContext(
        repo="pytorch/rl",
        pr_number=123,
        comment_id=456,
        comment_author=comment_author,
        pr_info={"author": {"login": pr_author}, **pr_info},
    )


def test_pr_author_can_request_review(monkeypatch):
    gh_calls = []
    comments = []
    ctx = _context(comment_author="Contributor", pr_author="contributor")

    def unexpected_permission_check(*_args):
        raise AssertionError("PR authors should not need repository write access")

    monkeypatch.setattr(
        torchrlbot, "check_write_permission", unexpected_permission_check
    )
    monkeypatch.setattr(
        torchrlbot, "gh", lambda *args, **kwargs: gh_calls.append((args, kwargs))
    )
    monkeypatch.setattr(
        torchrlbot,
        "post_comment",
        lambda *args: comments.append(args),
    )

    torchrlbot.cmd_reviewer(ctx, argparse.Namespace(reviewers=["maintainer"]))

    assert gh_calls == [
        (
            (
                "api",
                "repos/pytorch/rl/pulls/123/requested_reviewers",
                "--method",
                "POST",
                "--input",
                "-",
            ),
            {"input": json.dumps({"reviewers": ["maintainer"]})},
        )
    ]
    assert comments[-1][-1] == (
        "Requested review from @maintainer (requested by @Contributor)."
    )


def test_write_collaborator_can_request_review(monkeypatch):
    gh_calls = []
    ctx = _context(comment_author="maintainer")

    monkeypatch.setattr(torchrlbot, "check_write_permission", lambda *_args: True)
    monkeypatch.setattr(
        torchrlbot, "gh", lambda *args, **kwargs: gh_calls.append((args, kwargs))
    )
    monkeypatch.setattr(torchrlbot, "post_comment", lambda *_args: None)

    torchrlbot.cmd_reviewer(ctx, argparse.Namespace(reviewers=["reviewer"]))

    assert len(gh_calls) == 1


def test_pr_author_is_not_requested_case_insensitively(monkeypatch):
    gh_calls = []
    comments = []
    ctx = _context(comment_author="maintainer", pr_author="Contributor")

    monkeypatch.setattr(torchrlbot, "check_write_permission", lambda *_args: True)
    monkeypatch.setattr(
        torchrlbot, "gh", lambda *args, **kwargs: gh_calls.append((args, kwargs))
    )
    monkeypatch.setattr(
        torchrlbot,
        "post_comment",
        lambda *args: comments.append(args),
    )

    torchrlbot.cmd_reviewer(
        ctx,
        argparse.Namespace(reviewers=["contributor", "reviewer"]),
    )

    assert json.loads(gh_calls[0][1]["input"]) == {"reviewers": ["reviewer"]}
    assert "skipped @Contributor" in comments[-1][-1]


def test_unrelated_contributor_cannot_request_review(monkeypatch):
    gh_calls = []
    comments = []
    ctx = _context(comment_author="unrelated-user")

    monkeypatch.setattr(torchrlbot, "check_write_permission", lambda *_args: False)
    monkeypatch.setattr(
        torchrlbot, "gh", lambda *args, **kwargs: gh_calls.append((args, kwargs))
    )
    monkeypatch.setattr(
        torchrlbot,
        "post_comment",
        lambda *args: comments.append(args),
    )

    torchrlbot.cmd_reviewer(ctx, argparse.Namespace(reviewers=["maintainer"]))

    assert not gh_calls
    assert "Only the PR author or collaborators with write access" in comments[-1][-1]


def test_rebase_rejects_fork_before_running_git(monkeypatch):
    comments = []
    ctx = _context(
        comment_author="maintainer",
        headRefName="topic",
        isCrossRepository=True,
        headRepository={"nameWithOwner": "contributor/rl"},
    )

    monkeypatch.setattr(torchrlbot, "check_write_permission", lambda *_args: True)
    monkeypatch.setattr(
        torchrlbot,
        "git",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("fork rebases must not invoke git")
        ),
    )
    monkeypatch.setattr(
        torchrlbot,
        "post_comment",
        lambda *args: comments.append(args),
    )

    torchrlbot.cmd_rebase(ctx, argparse.Namespace(branch="main"))

    assert "cannot push to forks" in comments[-1][-1]
    assert "contributor/rl:topic" in comments[-1][-1]


if __name__ == "__main__":
    pytest.main([__file__])
