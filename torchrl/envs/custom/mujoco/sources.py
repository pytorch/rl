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
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

from torchrl._utils import logger as torchrl_logger

GITHUB_MODELS_ROOT = os.path.join(
    os.path.expanduser("~"), ".cache", "torchrl", "github_models"
)
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_REPO_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")
_TIMEOUT = 60.0


class ModelSource(Protocol):
    """Anything that resolves to the path of a MuJoCo XML.

    ``download`` is the explicit permission for network access: a source that
    would have to fetch files raises ``FileNotFoundError`` when it is
    ``False``, naming what to do instead.
    """

    def resolve(self, *, download: bool = False) -> Path:
        ...


def _resolve_model_source(
    source: ModelSource | str | Path, *, download: bool = False
) -> Path | str:
    """Resolve a source to an existing XML; local paths and URLs pass through.

    A local path must exist; an ``http(s)`` URL is returned as is for
    :class:`~torchrl.envs.MujocoEnv` to fetch, which needs a self-contained XML.
    """
    if isinstance(source, str) and source.startswith(("http://", "https://")):
        return source
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"MuJoCo model XML not found: {path}.")
        return path.resolve()
    if not callable(getattr(source, "resolve", None)):
        raise TypeError(
            "Expected the path of an XML or a model source with a "
            f"resolve(download=...) method, got {type(source).__name__}."
        )
    return source.resolve(download=download)


def _github_request(
    url: str, accept: str, *, token: str | None = None
) -> urllib.request.Request:
    request = urllib.request.Request(
        url, headers={"Accept": accept, "User-Agent": "torchrl"}
    )
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    return request


def _github_open(url: str, accept: str, what: str, *, token: str | None = None):
    """Open a GitHub URL, turning HTTP errors into messages that name the request."""
    try:
        return urllib.request.urlopen(
            _github_request(url, accept, token=token), timeout=_TIMEOUT
        )
    except urllib.error.HTTPError as err:
        hint = ""
        if err.code in (401, 403):
            hint = (
                " GITHUB_TOKEN is set but rejected; unset or renew it."
                if token
                else " Set GITHUB_TOKEN for a private repository or a higher rate limit."
            )
        raise FileNotFoundError(f"{what}: HTTP {err.code} from {url}.{hint}") from err


def _github_commit(repo: str, revision: str) -> str:
    """Resolve a branch, tag or short SHA to a full commit SHA through the GitHub API."""
    url = f"https://api.github.com/repos/{repo}/commits/{urllib.parse.quote(revision)}"
    with _github_open(
        url,
        "application/vnd.github.sha",
        f"Could not resolve {repo}@{revision}",
        token=os.environ.get("GITHUB_TOKEN"),
    ) as response:
        return response.read().decode("utf-8").strip()


def _download_github_tree(repo: str, commit: str, target: Path) -> Path:
    """Fetch the repository tree at ``commit`` into ``target``.

    The plain archive URL serves public repositories without credentials;
    when it answers 404 and ``GITHUB_TOKEN`` is set, the API archive endpoint
    is tried with the token for a private one. The archive is extracted next
    to the target and moved into place atomically, so a concurrent caller
    either finds the complete tree or fetches it itself.
    """
    root = target.parent
    root.mkdir(parents=True, exist_ok=True)
    what = f"Could not download {repo}@{commit[:9]}"
    torchrl_logger.info("Downloading %s at %s to %s", repo, commit[:9], target)
    with TemporaryDirectory(prefix=".download-", dir=root) as tmp:
        archive = Path(tmp) / "archive.zip"
        try:
            response = _github_open(
                f"https://github.com/{repo}/archive/{commit}.zip", "*/*", what
            )
        except FileNotFoundError:
            token = os.environ.get("GITHUB_TOKEN")
            if not token:
                raise
            response = _github_open(
                f"https://api.github.com/repos/{repo}/zipball/{commit}",
                "application/vnd.github+json",
                what,
                token=token,
            )
        with response, open(archive, "wb") as handle:
            shutil.copyfileobj(response, handle)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
        trees = [path for path in Path(tmp).iterdir() if path.is_dir()]
        if len(trees) != 1:
            raise RuntimeError(
                f"{what}: expected one top-level directory in the archive, got "
                f"{[path.name for path in trees]}."
            )
        try:
            trees[0].replace(target)
        except OSError:
            if not target.is_dir():
                raise
    return target


def _write_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}-", delete=False
    )
    with handle:
        handle.write(text)
    os.replace(handle.name, path)


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
    removed. Public repositories need no credentials; ``GITHUB_TOKEN`` is
    sent to the API for the revision lookup and, when the public archive
    answers 404, for the archive of a private repository.

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
        segments = self.repo.split("/")
        if len(segments) != 2 or any(
            not _REPO_SEGMENT.match(segment) or segment in (".", "..")
            for segment in segments
        ):
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
            cached = ref_file.read_text().strip()
            if _FULL_SHA.match(cached):
                return cached
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
        _write_atomically(ref_file, sha)
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
