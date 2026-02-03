# Scripts

This directory contains utility scripts for repository maintenance.

## check-sphinx-section-underline

Validates and auto-fixes Sphinx/ReST section underline lengths in documentation files.

**Usage:**
```bash
# Check files
python3 scripts/check-sphinx-section-underline docs/source/**/*.rst

# Check and fix
python3 scripts/check-sphinx-section-underline --fix docs/source/**/*.rst
```

This script is automatically run by pre-commit on all `.rst` files in the `docs/` directory.



