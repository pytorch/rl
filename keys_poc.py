import warnings
from dataclasses import dataclass, asdict

from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey
from torchrl.objectives import LossModule, ValueEstimators


class GAE(TensorDictModuleBase):
    @dataclass
    class _AcceptedKeys:
        advantage: NestedKey = "advantage"
        value: NestedKey = "state_value"
        reward: NestedKey = "reward"

    def __init__(self, value_network):
        super().__init__()
        self.key_list = self._AcceptedKeys()
        # here all keys have default values, which means one cannot rely here on the tensordict keys used there
        # the following code (from the actual ValueEstimatorBase) would raise always an exception
        if (
            hasattr(value_network, "out_keys")
            and self.keys.value_key not in value_network.out_keys
        ):
            raise KeyError(
                f"value key '{self.keys.value_key}' not found in value network out_keys."
            )

    @property
    def keys(self):
        return self._keys

    @keys.setter
    def keys(self, value):
        if not isinstance(value, type(self._AcceptedKeys)):
            raise ValueError
        self._keys = value

    def set_keys(self, **kw):
        for key, val in kw.items():
            if not isinstance(val, NestedKey):
                raise ValueError
        conf = asdict(self.keys)
        conf.update(kw)
        self._keys = self._AcceptedKeys(**conf)

    @property
    def in_keys(self):
        return [self.keys.value, self.keys.reward]

    @property
    def out_keys(self):
        return [self.keys.advantage]


class PPOLoss(LossModule):
    @dataclass
    class _AcceptedKeys:
        advantage: NestedKey = "advantage"
        value: NestedKey = "state_value"
        other: NestedKey = "other"

    def __init__(self, advantage_key=None):
        super().__init__()
        if advantage_key is not None:
            warnings.warn("deprecated", category=DeprecationWarning)
            self.set_keys(advantage=advantage_key)

    def make_value_estimator(
        self,
        value_type: ValueEstimators = None,
        **hyperparams
        ):
        self.advantage_module = GAE()
        keys = asdict(self.advantage_module.keys)

        # update keys since module prevails
        for key in list(keys.keys()):
            keys[key] = getattr(self.keys, key)
        self.advantage_module.set_keys(**keys)

    @property
    def keys(self):
        return self._keys

    @keys.setter
    def keys(self, value):
        if not isinstance(value, type(self._AcceptedKeys)):
            raise ValueError
        self._keys = value

    def set_keys(self, **kw):
        for key, val in kw.items():
            if not isinstance(val, NestedKey):
                raise ValueError
            # add custom checks here
            if key == "value":
                if key not in self.value_network.out_keys:
                    raise RuntimeError("unexpected")
        conf = asdict(self.keys)
        conf.update(kw)
        self._keys = self._AcceptedKeys(**conf)

    @property
    def in_keys(self):
        return [*self.advantage_module.in_keys, self.keys.other]

    @property
    def out_keys(self):
        return ["loss_actor"]

