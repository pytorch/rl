# Release Notes - v0.10.1

This patch release includes bug fixes, type annotation improvements, and CI enhancements cherry-picked from main.

## Bug Fixes

- **#3168** - @vmoens - [BugFix] AttributeError in accept_remote_rref_udf_invocation
  - Fixed AttributeError in RPC utilities when decorating classes with remote RRef invocation by handling None values in getattr calls

## Features

- **#3174** - @vmoens - [Feature] Named dims in Composite
  - Added support for named dimensions in Composite specs, enabling better integration with PyTorch's named tensors

- **#3214** - Louis Faury - [Feature] Composite specs can create named tensors with 'zero' and 'rand'
  - Extended Composite specs to properly propagate names when creating tensors using `zero()` and `rand()` methods

## Type Annotations & Documentation

- @vmoens - [Typing] Edit wrongfully set str type annotations
  - Fixed incorrect string type annotations across 19 files

- **#3175** - @vmoens - [Versioning] Fix doc versioning
  - Fixed documentation versioning issues

## CI/Build Improvements

- **#3200** - @vmoens - [CI] Use pip install
  - Updated CI workflows to use pip install across 41 files

- @vmoens - [CI] Fix missing librhash0 in doc CI
  - Added missing librhash0 dependency in documentation CI

- @vmoens - [CI] Fix benchmarks for LLMs
  - Fixed LLM benchmark CI configurations

- **#3222** - @vmoens - [CI] Upgrade doc python version
  - Upgraded Python version in documentation build workflows and added vLLM plugin entry point for FP32 overrides

