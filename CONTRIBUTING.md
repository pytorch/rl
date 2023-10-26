# Contributing to torchrl
We want to make contributing to this project as easy and transparent as
possible.

## Installing the library
Install the library as suggested in the README. For advanced features,
it is preferable to install the nightly built of pytorch.

TorchRL and TensorDict are being developed jointly. Stable releases are tied
together such that you can safely install the latest version of both libraries.
For TorchRL development, we recommend installing tensordict nightly

```
pip install tensordict-nightly
```
or the git version of the library:
```
pip install git+https://github.com/pytorch/tensordict
```

Once cloned, make sure you install torchrl in develop mode by running
```
python setup.py develop
```
in your shell.

If the generation of this artifact in MacOs M1 doesn't work correctly or in the execution the message
`(mach-o file, but is an incompatible architecture (have 'x86_64', need 'arm64e'))` appears, then try

```
ARCHFLAGS="-arch arm64" python setup.py develop
```

## Formatting your code
**Type annotation**

TorchRL is not strongly-typed, i.e. we do not enforce type hints, neither do we check that the ones that are present are valid. We rely on type hints purely for documentary purposes. Although this might change in the future, there is currently no need for this to be enforced at the moment.

**Linting**

Before your PR is ready, you'll probably want your code to be checked. This can be done easily by installing
```
pip install pre-commit
```
and running
```
pre-commit run --all-files
```
from within the torchrl cloned directory.

You can also install [pre-commit hooks](https://pre-commit.com/) (using `pre-commit install`
). You can disable the check by appending `-n` to your commit command: `git commit -m <commit message> -n`

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite and the documentation pass.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

When submitting a PR, we encourage you to link it to the related issue (if any) and add some tags to it.

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License
By contributing to rl, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
