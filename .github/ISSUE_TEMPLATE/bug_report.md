---
name: Bug report
about: Create a report to help us improve
title: "[BUG]"
labels: ["bug"]
assignees: vmoens

---

## Describe the bug

A clear and concise description of what the bug is.

## To Reproduce

Steps to reproduce the behavior.

Please try to provide a minimal example to reproduce the bug. Error messages and stack traces are also helpful.

Please use the markdown code blocks for both code and stack traces.

```python
import torchrl
```

```bash
Traceback (most recent call last):
  File ... 
```

## Expected behavior

A clear and concise description of what you expected to happen.

## Screenshots

If applicable, add screenshots to help explain your problem.

## System info

Describe the characteristic of your environment:
 * Describe how the library was installed (pip, source, ...)
 * Python version
 * Versions of any other relevant libraries

```python
import torchrl, numpy, sys
print(torchrl.__version__, numpy.__version__, sys.version, sys.platform)
```

## Additional context

Add any other context about the problem here.

## Reason and Possible fixes

If you know or suspect the reason for this bug, paste the code lines and suggest modifications.

## Checklist

- [ ] I have checked that there is no similar issue in the repo (**required**)
- [ ] I have read the [documentation](https://github.com/facebookresearch/rl/tree/main/docs/) (**required**)
- [ ] I have provided a minimal working example to reproduce the bug (**required**)
