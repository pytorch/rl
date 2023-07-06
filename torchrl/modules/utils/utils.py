from torchrl.data.tensor_specs import (
    BinaryDiscreteTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    MultiOneHotDiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
    TensorSpec,
)

ACTION_SPACE_MAP = {}
ACTION_SPACE_MAP[OneHotDiscreteTensorSpec] = "one_hot"
ACTION_SPACE_MAP[MultiOneHotDiscreteTensorSpec] = "mult_one_hot"
ACTION_SPACE_MAP[BinaryDiscreteTensorSpec] = "binary"
ACTION_SPACE_MAP[DiscreteTensorSpec] = "categorical"
ACTION_SPACE_MAP["one_hot"] = "one_hot"
ACTION_SPACE_MAP["one-hot"] = "one_hot"
ACTION_SPACE_MAP["mult_one_hot"] = "mult_one_hot"
ACTION_SPACE_MAP["mult-one-hot"] = "mult_one_hot"
ACTION_SPACE_MAP["multi_one_hot"] = "mult_one_hot"
ACTION_SPACE_MAP["multi-one-hot"] = "mult_one_hot"
ACTION_SPACE_MAP["binary"] = "binary"
ACTION_SPACE_MAP["categorical"] = "categorical"
# TODO for the future ;)
# ACTION_SPACE_MAP[MultiDiscreteTensorSpec] = "multi_categorical"
# ACTION_SPACE_MAP["multi_categorical"] = "multi_categorical"
# ACTION_SPACE_MAP["multi-categorical"] = "multi_categorical"
# ACTION_SPACE_MAP["multi_discrete"] = "multi_categorical"
# ACTION_SPACE_MAP["multi-discrete"] = "multi_categorical"


def _find_action_space(action_space):
    if isinstance(action_space, TensorSpec):
        if isinstance(action_space, CompositeSpec):
            if "action" in action_space.keys():
                _key = "action"
            else:
                # the first key is the action
                for _key in action_space.keys(True, True):
                    if isinstance(_key, tuple) and _key[-1] == "action":
                        break
                else:
                    raise KeyError
            action_space = action_space[_key]
        action_space = type(action_space)
    try:
        action_space = ACTION_SPACE_MAP[action_space]
    except KeyError:
        raise ValueError(
            f"action_space was not specified/not compatible and could not be retrieved from the value network. Got action_space={action_space}."
        )
    return action_space
