# TorchRL Release Agent Prompt

This document provides instructions for an AI assistant to help automate TorchRL releases.

## Prerequisites

Before starting a release, ensure you have:
- Write access to the repository
- Ability to create branches, tags, and pull requests
- Access to view GitHub Actions workflow runs

## Input Parameters

Collect the following information from the user:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `version_tag` | The version to release | `v0.11.0` |
| `release_type` | Major (0.x.0) or minor (0.x.y) release | `major` or `minor` |
| `pytorch_release` | PyTorch release branch to build against | `release/2.8` |
| `previous_version` | Previous release tag (for release notes) | `v0.10.0` |

---

## Step 1: Analyze Commits for Release Notes

### For Major Releases (e.g., 0.10.x → 0.11.0)

Get commits from the last major release:

```bash
# Find the last major release tag
git fetch --tags
git log v0.10.0..HEAD --oneline --no-merges
```

### For Minor Releases (e.g., 0.11.0 → 0.11.1)

Get commits from the last release:

```bash
git log v0.11.0..HEAD --oneline --no-merges
```

**Important: PR Selection for Minor Releases**

When selecting PRs for a minor release, follow this decision flow:

1. **If labeled `user-facing`** → **Exclude** (only for major releases)
2. **If labeled `non-user-facing` or `Suitable for minor`** → **Include**
3. **If neither label is present** → **Assess yourself** based on the changes

Labels:
- `user-facing` - API changes, new features, or public interface changes
- `non-user-facing` - Internal changes, bug fixes, refactoring
- `Suitable for minor` - Explicitly marked as safe for minor releases

To filter PRs:
```bash
# Find PRs explicitly safe for minor release
gh pr list --label "non-user-facing" --state merged --json number,title
gh pr list --label "Suitable for minor" --state merged --json number,title

# Check labels on a specific PR
gh pr view <PR_NUMBER> --json labels --jq '.labels[].name'
```

For unlabeled PRs, review the changes and determine if they affect the public API or just internal implementation.

### Critical: Don't Miss ghstack Commits

**The biggest pitfall in release notes is only looking at commits with PR numbers.** Many of the most significant features are merged via ghstack and have NO PR number in the commit message. Always analyze both:

1. **PR commits**: `git log v0.10.0..v0.11.0 --oneline | grep "(#"`
2. **ghstack commits**: `git log v0.10.0..v0.11.0 --oneline | grep -v "(#"`

ghstack commits often represent major feature work (e.g., Dreamer overhaul, weight sync schemes, new algorithms like DAPO/CISPO).

### Handling ghstack Commits

Many commits are merged via ghstack and do NOT have PR numbers in the commit message. These often contain major features:

```bash
# Find ghstack commits (no PR number)
git log v0.10.0..v0.11.0 --oneline | grep -v "(#"

# Search for specific features by keyword
git log v0.10.0..v0.11.0 --oneline | grep -i dreamer
git log v0.10.0..v0.11.0 --oneline | grep -i "weight.*sync\|sync.*scheme"
git log v0.10.0..v0.11.0 --oneline | grep -i collector
```

Include these in release notes using their short SHA (e.g., `cc917bae`) instead of a PR number. Group related ghstack commits by feature area.

### Categorize Commits

Review each commit and categorize into:

1. **Features** - New functionality
   - Look for: "add", "implement", "support", "new"

2. **Bug Fixes** - Issue corrections
   - Look for: "fix", "bug", "issue", "patch", "correct"

3. **Breaking Changes** - API changes that may break existing code
   - Look for: "breaking", "remove", "deprecate", "rename" (of public APIs)
   - Check for removed or renamed public functions/classes
   - Check for changed function signatures

4. **Deprecations** - Features marked for future removal
   - Look for: "deprecate", "warn"

5. **Performance** - Speed or memory improvements
   - Look for: "perf", "optimize", "speed", "memory", "faster"

6. **Documentation** - Doc improvements (usually not included in notes)

7. **CI/Infrastructure** - Build/test changes (usually not included in notes)

### Generate Human-Readable Summaries

Transform commit messages into user-friendly descriptions:

**Bad:** `fix: handle edge case in TensorDict.to_dict() when nested (#1234)`
**Good:** `Fixed an issue where TensorDict.to_dict() would fail with deeply nested structures`

**Bad:** `feat: add support for bf16 in memmap (#1235)`
**Good:** `Added bfloat16 support for memory-mapped tensors`

---

## Step 2: Draft Release Notes

Present the following template to the user for review:

```markdown
## TorchRL {version_tag}

### Highlights

<!-- 2-3 sentence summary of the most important changes -->

### Breaking Changes

<!-- List any breaking changes. If none, write "No breaking changes in this release." -->

### Deprecations

<!-- List any new deprecations. If none, remove this section. -->

### Features

- Feature description ([#PR_NUMBER](link))
- ...

### Bug Fixes

- Fix description ([#PR_NUMBER](link))
- ...

### Performance

- Performance improvement description ([#PR_NUMBER](link))
- ...

### Contributors

Thanks to all contributors:
<!-- List first-time contributors especially -->
```

### Release Notes Format Requirements

- **Author handles**: Every entry must include the lead author's GitHub handle (e.g., `@vmoens`)
- **Explanations**: For PRs/commits with >10 lines changed, add 1-3 lines of explanation
- **Metadata**: Use `gh pr view <PR> --json author,additions,deletions` to get PR details
- **ghstack commits**: Use `git show --stat <sha>` to check lines changed
- **File storage**: Save release notes to `RELEASE_NOTES_v0.X.Y.md` in the repo root to preserve context

### Release Notes Best Practices

1. **Create the file early** - Save to `RELEASE_NOTES_v0.X.Y.md` as you work. This preserves context and allows iterative refinement.

2. **Group by feature, not by commit** - Multiple ghstack commits often belong to the same feature. Group them under a single feature heading (e.g., "Dreamer World Model Improvements").

3. **Prioritize by impact** - Use `git show --stat <sha>` to see lines changed. Features with 100+ lines deserve their own section in highlights.

4. **Check the sota-implementations/** - Major algorithm updates often land here. Check for new or significantly updated implementations.

5. **Verify docs build before updating stable** - The docs workflow must complete and upload the version folder to gh-pages before you can update the stable symlink.

**Important:** Wait for user approval before proceeding.

---

## Step 3: Update Version Files

Update version in **all 3 required locations**:

| File | Variable/Content | Example |
|------|------------------|---------|
| `version.txt` | Version string | `0.11.0` |
| `.github/scripts/td_script.sh` | `export TORCHRL_BUILD_VERSION=` | `export TORCHRL_BUILD_VERSION=0.11.0` |
| `.github/scripts/version_script.bat` | `set TORCHRL_BUILD_VERSION=` | `set TORCHRL_BUILD_VERSION=0.11.0` |

### Commands to update all files:

```bash
# 1. Root version.txt
echo "{version_without_v}" > version.txt

# 2. td_script.sh (Linux/macOS builds)
sed -i 's/^export TORCHRL_BUILD_VERSION=.*/export TORCHRL_BUILD_VERSION={version_without_v}/' .github/scripts/td_script.sh

# 3. version_script.bat (Windows builds) - IMPORTANT: Don't forget this one!
sed -i 's/^set TORCHRL_BUILD_VERSION=.*/set TORCHRL_BUILD_VERSION={version_without_v}/' .github/scripts/version_script.bat
```

**Note:** The release workflow includes sanity checks that verify all 3 files have matching versions. If any file is missed, the release will fail at the sanity check step.

### For Major Releases Only: Update pyproject.toml

Update tensordict version constraint if major version changed:
```toml
# For 0.11.x release, update to:
"tensordict>=0.11.0,<0.12.0",
```

---

## Step 4: Commit Version Changes on the main branch (Major releases only)

Check that the version hasn't been bumped yet!
If not:
```bash
git checkout -b bump-v{version} origin/main
git add version.txt .github/scripts/td_script.sh .github/scripts/version_script.bat pyproject.toml
git commit -m "Bump version to {version_without_v}"
gh pr create -t "Bump version to {version_without_v}" -b ""
```
Then merge the PR.

---

## Step 5: Create Release Branch

```bash
# Create release branch from main
git checkout main
git pull origin main
git checkout -b release/{major.minor}

# Example: for v0.11.0 (Must always contain version, major and minor)
git checkout -b release/0.11.0
```

---

## Step 6: Commit Version Changes (Minor releases only)

On minors, the version is bumped locally on the release branch.

```bash
git add version.txt .github/scripts/td_script.sh .github/scripts/version_script.bat
git commit -m "Bump version to {version_without_v}"
```

---

## Step 7: Create and Push Tag

```bash
# Create annotated tag
git tag -a {version_tag} -m "TorchRL {version_tag}"

# Push branch and tag
git push origin release/{major.minor}
git push origin {version_tag}
```

---

## Step 8: Trigger Release Workflow

The release workflow is triggered manually via workflow_dispatch (not automatically on tag push).

To trigger:

1. Go to Actions → Release workflow
2. Click "Run workflow"
3. Fill in:
   - `tag`: The version tag (e.g., `v0.11.0`)
   - `pytorch_release`: The PyTorch branch (e.g., `release/2.8`)
   - `dry_run`: Check for testing without publishing

---

## Step 9: Create Draft GitHub Release

If not using the automated workflow, create manually:

```bash
gh release create {version_tag} \
  --draft \
  --title "TorchRL {version_tag}" \
  --notes-file RELEASE_NOTES.md
```

---

## Step 10: Monitor Workflow

Watch the release workflow for:

1. **Sanity checks** - Verify all version files match
2. **Wheel builds** - All platforms should succeed (builds run sequentially)
3. **Docs update** - Stable symlink updated
4. **Release creation** - Draft created with wheels

---

## Post-Release Manual Steps

### 1. Review Draft Release Notes

- Go to GitHub Releases
- Find the draft release for `{version_tag}`
- Edit the release notes with the content prepared in Step 2
- Add any last-minute changes or acknowledgments

### 2. Publish the Release

- Once satisfied with release notes, click "Publish release"
- This makes the release public and notifies watchers

### 3. Approve PyPI Publishing

- Go to the release workflow run
- Find the "Publish to PyPI" job waiting for approval
- Review the wheels to be published
- Approve the environment to proceed

### 4. Verify PyPI Upload

```bash
# Check that the package is available
pip index versions torchrl

# Or install and verify
pip install torchrl=={version_without_v}
python -c "import torchrl; print(torchrl.__version__)"
```

### 5. Verify Documentation

- Check https://pytorch.org/rl/stable/ points to the new version
- Verify the version selector includes the new version

### 6. Announce the Release

Consider announcing on:
- PyTorch forums
- Twitter/X
- Discord/Slack channels
- Project mailing lists

---

## Troubleshooting

### Version Mismatch Errors

If sanity checks fail due to version mismatch:
```bash
# Check all 3 version files
cat version.txt
grep "^export TORCHRL_BUILD_VERSION=" .github/scripts/td_script.sh
grep "^set TORCHRL_BUILD_VERSION=" .github/scripts/version_script.bat
```

All 3 files must have the same version. If any is wrong, update it and re-run the release.

**Common mistake:** Forgetting to update `version_script.bat` causes Windows wheels to be built with the wrong version.

### Wheel Build Failures

Check the individual build workflow logs. Common issues:
- PyTorch version compatibility
- Missing dependencies
- Platform-specific compilation errors
- **Wrong version in Windows builds**: Check `version_script.bat` has correct `TORCHRL_BUILD_VERSION`

### Docs Update Failure

If the version folder doesn't exist on gh-pages:
1. Trigger the docs workflow manually for the release tag
2. Wait for docs to build and upload
3. Re-run the release workflow or manually update stable symlink

### PyPI Upload Failure

If OIDC authentication fails:
1. Verify the repository is configured as a trusted publisher on PyPI
2. Check the workflow permissions include `id-token: write`
3. Ensure the environment name matches PyPI configuration (`pypi`)

### If tag already exists:
```bash
# Delete local and remote tag (use with caution!)
git tag -d v<VERSION>
git push origin :refs/tags/v<VERSION>
```

### If release branch already exists:
```bash
# Delete and recreate (use with caution!)
git push origin --delete release/<VERSION>
```

### gh-pages Push Rejected

If `git push` to gh-pages is rejected due to remote updates (common when CI updates gh-pages in parallel):
```bash
git fetch origin gh-pages
git rebase origin/gh-pages
git push origin HEAD:gh-pages
```

---

## Environment Setup for PyPI Trusted Publishing

Before the first release, configure PyPI trusted publishing:

1. Go to https://pypi.org/manage/project/torchrl/settings/publishing/
2. Add a new trusted publisher:
   - Owner: `pytorch`
   - Repository: `rl`
   - Workflow name: `release.yml`
   - Environment name: `pypi`

---

## Quick Reference Commands

```bash
# Check current version
cat version.txt

# List recent tags
git tag --sort=-creatordate | head -10

# View commits since last release
git log $(git describe --tags --abbrev=0)..HEAD --oneline

# Verify tag exists and is annotated
git show {version_tag}

# Check workflow status
gh run list --workflow=release.yml

# Download release artifacts
gh release download {version_tag} --dir ./release-artifacts
```

---

## Summary Template

After completing all steps, provide this summary to the user:

```
## Release v<VERSION> Preparation Complete

### Completed Steps:
- [x] Analyzed commits from <LAST_RELEASE> to HEAD
- [x] Wrote release notes (categorized: X features, Y bug fixes, Z breaking changes)
- [x] Created release branch: release/<VERSION>
- [x] Updated version files:
  - version.txt: <VERSION>
  - td_script.sh: TORCHRL_BUILD_VERSION=<VERSION>
  - version_script.bat: TORCHRL_BUILD_VERSION=<VERSION>
  - pyproject.toml: tensordict constraint (if major release)
- [x] Created and pushed tag: v<VERSION>
- [x] Created draft GitHub release with release notes
- [x] Triggered release workflow

### Manual Steps Required:
1. Monitor build workflow: <WORKFLOW_URL>
2. Review draft release: https://github.com/pytorch/rl/releases/tag/v<VERSION>
3. Publish release when all builds pass
4. Approve PyPI publish in workflow (requires environment approval)

### Important URLs:
- Release workflow: <WORKFLOW_RUN_URL>
- Draft release: https://github.com/pytorch/rl/releases/tag/v<VERSION>
- PyPI project: https://pypi.org/project/torchrl/
```

---

## Version Naming Convention

- **Major releases**: `v0.11.0`, `v0.12.0` - New features, may have breaking changes
- **Minor/Patch releases**: `v0.11.1`, `v0.11.2` - Bug fixes only, no new features or user-facing changes
- **Release candidates**: `v0.11.0-rc1` - Pre-release testing

**Note:** PRs labeled `user-facing` must only be included in major releases, never in minor/patch releases.

## TensorDict Version Compatibility

TorchRL and TensorDict versions must match in major version:
- TorchRL `0.11.x` requires TensorDict `>=0.11.0,<0.12.0`
- TorchRL `0.12.x` requires TensorDict `>=0.12.0,<0.13.0`

Ensure TensorDict is released first, or coordinate releases.

## TensorDict Installation in Builds

The build scripts automatically determine how to install tensordict based on the branch:

| Branch/Context | TensorDict Source | Reason |
|----------------|-------------------|--------|
| `release/*` branches | PyPI (stable) | Release builds should use stable tensordict |
| Release tags (`v0.11.0`) | PyPI (stable) | Same as release branches |
| `main` branch | Git (latest dev) | Development builds need latest tensordict |
| Pull requests | Git (latest dev) | Test with latest tensordict changes |
| `nightly` branch | Git (latest dev) | Nightly builds use bleeding edge |

### Manual Override

When triggering the release workflow manually, you can choose the tensordict source:
- `auto` (default): Uses branch-based detection as described above
- `stable`: Forces installation from PyPI
- `git`: Forces installation from git

**Note**: The manual override is passed to build workflows, but due to GitHub Actions limitations with reusable workflows, the primary mechanism is branch-based detection in the build scripts.

## Documentation Updates

The release workflow automatically updates gh-pages:
- `stable` symlink is updated to point to the new version folder
- `versions.html` is updated to mark the new version as "(stable release)"

If the docs for the new version folder don't exist yet, they will be built and deployed by the `docs.yml` workflow when the tag is pushed.

### Manual gh-pages Updates

If you need to manually update gh-pages (e.g., update stable symlink, fix versions.html):

```bash
# Create worktree for gh-pages (avoids switching branches)
git worktree add .worktrees/gh-pages gh-pages

# Make changes in the worktree
cd .worktrees/gh-pages

# Update stable symlink to new version
rm -f stable && ln -sf 0.11 stable

# Update versions.html - remove old stable marker, add new one
sed -i 's/ (stable release)//g' versions.html
sed -i 's|href="0.11/">v0.11|href="0.11/">v0.11 (stable release)|g' versions.html

# Add new version entry if needed (insert after "main (unstable)" line)
# Check versions.html to see the format

# Commit and push
git add -A && git commit -m "Update stable to 0.11"
git push origin HEAD:gh-pages

# Cleanup worktree
cd ../..
git worktree remove .worktrees/gh-pages
```

**Note:** If `rsync` fails with "read-only variable" error when copying docs, use `cp -R` instead.

---

## Common Pitfalls

### Release Notes

1. **Only looking at PR commits** - The biggest mistake. ghstack commits have no PR numbers but often contain the most significant features. Always run both `grep "(#"` and `grep -v "(#"` on the commit log.

2. **Missing related commits** - A feature like "Dreamer improvements" may span 10+ commits. Search by keyword (`grep -i dreamer`) to find all related work.

3. **Not checking lines changed** - A commit titled "[BugFix] Minor fix" might actually be 300 lines of important refactoring. Use `git show --stat` to verify.

4. **Forgetting author attribution** - Every entry needs `@username`. For ghstack commits, check `git log --format='%an %ae' <sha>` to find the author.

### Documentation Updates

1. **Updating stable before docs exist** - The docs workflow must complete first. Check that the version folder exists on gh-pages before updating the stable symlink.

2. **Not adding version to versions.html** - When manually updating gh-pages, remember to add a new `<li>` entry for the version in `versions.html`, not just update the stable symlink.

### General Process

1. **Working directly on gh-pages branch** - Use `git worktree` instead. It's cleaner and avoids accidentally committing to the wrong branch.

2. **Not verifying the release branch exists** - Before tagging, ensure the release branch (e.g., `release/0.11.0`) is pushed and has all necessary commits.

3. **Assuming CI passed** - Always monitor the docs/release workflows. Check the actual logs if something seems off.
