# Vendored from SimpleVLA-RL (https://github.com/PRIME-RL/SimpleVLA-RL),
# path verl/utils/vla_utils/openvla_oft, commit
# 7c51662df27b586f9e8a1ab35fcf849f2b8852f9. MIT License, Copyright (c) 2025
# PRIME-RL.
#
# This is the OpenVLA-OFT *token* variant used by the SimpleVLA-RL SFT
# checkpoints (HF: Haozhan72/*): parallel decoding and action chunking with
# the continuous L1 head reverted to discrete action tokens emitted by the
# language-model head. The vanilla `openvla-7b` remote code cannot load these
# checkpoints, hence the vendoring. Files are kept verbatim except
# constants.py, where the implicit (and, for LIBERO, buggy) command-line
# platform sniffing is replaced by an explicit ROBOT_PLATFORM environment
# variable defaulting to LIBERO -- see the note there.
