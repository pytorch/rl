# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Task-specific policy architectures built from TorchRL modules."""

from .microduck_policy import MicroDuckSkillPolicy, MicroDuckSkills

__all__ = ["MicroDuckSkillPolicy", "MicroDuckSkills"]
