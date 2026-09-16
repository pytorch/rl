# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Where a MuJoCo model comes from: a local file, a GitHub repository, a collection.

A model source resolves to the path of one XML on disk, fetching and caching
the files it needs when ``download=True`` allows network access.
:class:`~torchrl.envs.MujocoModelEnv` loads any source, and
:class:`~torchrl.envs.MenagerieEnv` wraps the curated
:class:`~torchrl.envs.MenagerieModelSource`.
"""

from __future__ import annotations

import os
import re
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol, runtime_checkable

from torchrl._utils import logger as torchrl_logger

GITHUB_MODELS_ROOT = "~/.cache/torchrl/github_models"
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


@runtime_checkable
class ModelSource(Protocol):
    """Anything that resolves to the path of a MuJoCo XML.

    ``download`` is the explicit permission for network access: a source that
    would have to fetch files raises ``FileNotFoundError`` when it is
    ``False``, naming what to do instead.
    """

    def resolve(self, *, download: bool = False) -> Path:
        ...


def resolve_model_source(
    source: ModelSource | str | Path, *, download: bool = False
) -> Path:
    """Resolve a model source, or the path of a local XML, to an existing file.

    Args:
        source (ModelSource, str or Path): a source, or the XML itself.

    Keyword Args:
        download (bool, optional): whether the source may fetch files.
            Defaults to ``False``.

    Returns:
        The absolute path to the XML.
    """
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"MuJoCo model XML not found: {path}.")
        return path.resolve()
    return source.resolve(download=download)


def _github_commit(repo: str, revision: str) -> str:
    """Resolve a branch, tag or short SHA to a full commit SHA through the GitHub API."""
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/commits/{urllib.parse.quote(revision)}",
        headers={"Accept": "application/vnd.github.sha", "User-Agent": "torchrl"},
    )
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request) as response:
        return response.read().decode("utf-8").strip()


def _download_github_tree(repo: str, commit: str, target: Path) -> Path:
    """Fetch the repository tree at ``commit`` into ``target``.

    The archive is extracted next to the target and moved into place
    atomically, so a concurrent caller either finds the complete tree or
    fetches it itself.
    """
    root = target.parent
    root.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo}/archive/{commit}.zip"
    torchrl_logger.info("Downloading %s at %s to %s", repo, commit[:9], target)
    with TemporaryDirectory(prefix=".download-", dir=root) as tmp:
        archive = Path(tmp) / "archive.zip"
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
        trees = [path for path in Path(tmp).iterdir() if path.is_dir()]
        if len(trees) != 1:
            raise RuntimeError(
                f"Expected one top-level directory in {url}, got "
                f"{[path.name for path in trees]}."
            )
        try:
            trees[0].replace(target)
        except OSError:
            if not target.is_dir():
                raise
    return target


@dataclass(frozen=True)
class GitHubModelSource:
    """A MuJoCo model in a GitHub repository, pinned to one revision.

    Robot repositories put their XML at the root or several directories deep,
    keep meshes in sibling directories and often ship several scenes, so the
    source names everything instead of guessing: ``repo`` as ``"owner/name"``,
    ``revision`` (a commit SHA, tag or branch) and ``entry``, the
    repository-relative path of the XML to load. The whole repository tree at
    that revision is cached under ``root/owner/name/<commit>``, so relative
    includes and assets resolve as in a checkout. A tag or branch is resolved
    to its commit through the GitHub API on the first ``download=True`` and
    that mapping is cached next to the tree, so the source resolves offline
    afterwards and keeps pointing at the same commit until the cache entry is
    removed.

    Args:
        repo (str): ``"owner/name"`` of the repository.
        revision (str): commit SHA, tag or branch to pin. A full 40-character
            SHA needs no API call.
        entry (str): repository-relative path of the XML, for example
            ``"ur3e.xml"`` or ``"models/scene.xml"``.
        root (str or Path, optional): cache directory holding one
            ``owner/name/<commit>`` tree per revision. Defaults to
            ``~/.cache/torchrl/github_models``.

    Examples:
        >>> from torchrl.envs import GitHubModelSource, MujocoModelEnv
        >>> source = GitHubModelSource(
        ...     "SouthColumn76/universal_robots_ur3e",
        ...     revision="5f042ffca6b5885fd18f5448e17b71ab46274fa3",
        ...     entry="ur3e.xml",
        ... )
        >>> source.commit()
        '5f042ffca6b5885fd18f5448e17b71ab46274fa3'
        >>> source.resolve(download=True).name  # doctest: +SKIP
        'ur3e.xml'
        >>> env = MujocoModelEnv(source, download=True, seed=0)  # doctest: +SKIP
        >>> env.reset()["qpos"].shape  # doctest: +SKIP
        torch.Size([1, 6])
    """

    repo: str
    revision: str
    entry: str
    root: str | Path | None = None

    def __post_init__(self):
        owner, _, name = self.repo.partition("/")
        if not owner or not name or "/" in name:
            raise ValueError(f"repo must be 'owner/name', got {self.repo!r}.")
        entry = Path(self.entry)
        if entry.is_absolute() or ".." in entry.parts or not entry.parts:
            raise ValueError(
                f"entry must be a repository-relative path, got {self.entry!r}."
            )

    @property
    def cache_dir(self) -> Path:
        """The ``owner/name`` directory holding the cached trees of this repository."""
        root = GITHUB_MODELS_ROOT if self.root is None else self.root
        return Path(root).expanduser() / self.repo

    def commit(self, *, download: bool = False) -> str:
        """The full commit SHA of ``revision``.

        A full SHA is returned as is; a tag, branch or short SHA is read from
        the cache, or resolved through the GitHub API when ``download`` is
        ``True``.
        """
        if _FULL_SHA.match(self.revision):
            return self.revision
        ref_file = self.cache_dir / "refs" / urllib.parse.quote(self.revision, safe="")
        if ref_file.is_file():
            return ref_file.read_text().strip()
        if not download:
            raise FileNotFoundError(
                f"{self.repo}@{self.revision} has not been resolved to a commit "
                "yet; pass download=True to resolve it through the GitHub API."
            )
        sha = _github_commit(self.repo, self.revision)
        if not _FULL_SHA.match(sha):
            raise RuntimeError(
                f"GitHub returned {sha!r} for {self.repo}@{self.revision}."
            )
        ref_file.parent.mkdir(parents=True, exist_ok=True)
        ref_file.write_text(sha)
        return sha

    def resolve(self, *, download: bool = False) -> Path:
        """The path of ``entry`` inside the cached tree, fetching the tree if allowed.

        Raises:
            FileNotFoundError: if the revision or the tree is not cached and
                ``download`` is ``False``, or if the tree has no ``entry``;
                the message then lists the XML files the tree contains.
        """
        commit = self.commit(download=download)
        tree = self.cache_dir / commit
        if not tree.is_dir():
            if not download:
                raise FileNotFoundError(
                    f"{self.repo}@{commit[:9]} is not cached under {tree.parent}; "
                    "pass download=True to fetch it."
                )
            _download_github_tree(self.repo, commit, tree)
        xml = tree / self.entry
        if not xml.is_file():
            candidates = sorted(
                str(path.relative_to(tree)) for path in tree.rglob("*.xml")
            )
            raise FileNotFoundError(
                f"{self.repo}@{commit[:9]} has no {self.entry!r}; XML files in "
                f"the tree: {candidates}."
            )
        return xml.resolve()
