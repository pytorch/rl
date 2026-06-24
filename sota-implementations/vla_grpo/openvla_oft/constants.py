"""
Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.
"""
import os
from enum import Enum

# Llama 2 token constants
IGNORE_INDEX = -100
ACTION_TOKEN_BEGIN_IDX = 31743
STOP_INDEX = 2  # '</s>'


# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
LIBERO_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 8,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

ALOHA_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 25,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

ALOHA_CONSTANTS_12chunk = {
    "NUM_ACTIONS_CHUNK": 12,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

ALOHA_CONSTANTS_8chunk = {
    "NUM_ACTIONS_CHUNK": 8,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

ALOHA_CONSTANTS_6chunk = {
    "NUM_ACTIONS_CHUNK": 6,
    "ACTION_DIM": 14,
    "PROPRIO_DIM": 14,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

BRIDGE_CONSTANTS = {
    "NUM_ACTIONS_CHUNK": 5,
    "ACTION_DIM": 7,
    "PROPRIO_DIM": 7,
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}


# --- torchrl vendor note -----------------------------------------------------
# Upstream selects the platform by sniffing sys.argv, with a bug that maps
# "libero" in the command line to the ALOHA constants, and silently defaults
# to ALOHA. Vendored for the LIBERO recipe, the selection is explicit: set the
# ROBOT_PLATFORM environment variable (default: "LIBERO").
ROBOT_PLATFORM = os.environ.get("ROBOT_PLATFORM", "LIBERO").upper()
_PLATFORM_CONSTANTS = {
    "LIBERO": LIBERO_CONSTANTS,
    "ALOHA": ALOHA_CONSTANTS,
    "ALOHA_12": ALOHA_CONSTANTS_12chunk,
    "ALOHA_8": ALOHA_CONSTANTS_8chunk,
    "ALOHA_6": ALOHA_CONSTANTS_6chunk,
    "BRIDGE": BRIDGE_CONSTANTS,
}
if ROBOT_PLATFORM not in _PLATFORM_CONSTANTS:
    raise ValueError(
        f"Unknown ROBOT_PLATFORM {ROBOT_PLATFORM!r}; "
        f"expected one of {sorted(_PLATFORM_CONSTANTS)}."
    )
constants = _PLATFORM_CONSTANTS[ROBOT_PLATFORM]

# Assign constants to global variables
NUM_ACTIONS_CHUNK = constants["NUM_ACTIONS_CHUNK"]
ACTION_DIM = constants["ACTION_DIM"]
PROPRIO_DIM = constants["PROPRIO_DIM"]
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]


