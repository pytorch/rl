import numpy as np
from tensordict.tensordict import TensorDict, TensorDictBase

from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec, UnboundedDiscreteTensorSpec
from torchrl.envs import EnvBase
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
import torch

X, y = fetch_openml('adult', version=1, return_X_y=True)

def _get_data(dataset_name):
    if dataset_name in ['adult_num', 'adult_onehot']:
        X, y = fetch_openml('adult', version=1, return_X_y=True)
        is_NaN = X.isna()
        row_has_NaN = is_NaN.any(axis=1)
        X = X[~row_has_NaN]
        # y = y[~row_has_NaN]
        y = X["occupation"]
        X = X.drop(["occupation"],axis=1)
        cat_ix = X.select_dtypes(include=['category']).columns
        num_ix = X.select_dtypes(include=['int64', 'float64']).columns
        encoder = LabelEncoder()
        # now apply the transformation to all the columns:
        for col in cat_ix:
            X[col] = encoder.fit_transform(X[col])
        y = encoder.fit_transform(y)
        if dataset_name == 'adult_onehot':
            cat_features = OneHotEncoder(sparse=False).fit_transform(X[cat_ix])
            num_features = StandardScaler().fit_transform(X[num_ix])
            X = np.concatenate((num_features, cat_features), axis=1)
        else:
            X = StandardScaler().fit_transform(X)
    elif dataset_name in ['mushroom_num', 'mushroom_onehot']:
        X, y = fetch_openml('mushroom', version=1, return_X_y=True)
        encoder = LabelEncoder()
        # now apply the transformation to all the columns:
        for col in X.columns:
            X[col] = encoder.fit_transform(X[col])
        # X = X.drop(["veil-type"],axis=1)
        y = encoder.fit_transform(y)
        if dataset_name == 'mushroom_onehot':
            X = OneHotEncoder(sparse=False).fit_transform(X)
        else:
            X = StandardScaler().fit_transform(X)
    elif dataset_name == 'covertype':
        # https://www.openml.org/d/150
        # there are some 0/1 features -> consider just numeric
        X, y = fetch_openml('covertype', version=3, return_X_y=True)
        X = StandardScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(y)
    elif dataset_name == 'shuttle':
        # https://www.openml.org/d/40685
        # all numeric, no missing values
        X, y = fetch_openml('shuttle', version=1, return_X_y=True)
        X = StandardScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(y)
    elif dataset_name == 'magic':
        # https://www.openml.org/d/1120
        # all numeric, no missing values
        X, y = fetch_openml('MagicTelescope', version=1, return_X_y=True)
        X = StandardScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(y)
    else:
        raise RuntimeError('Dataset does not exist')
    return TensorDict({"X": X, "y": y}, X.shape[:1])

def make_composite_from_td(td):
    # custom funtion to convert a tensordict in a similar spec structure
    # of unbounded values.
    composite = CompositeSpec(
        {
            key: make_composite_from_td(tensor)
            if isinstance(tensor, TensorDictBase)
            else UnboundedContinuousTensorSpec(
                dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
            ) if tensor.dtype in (torch.float16, torch.float32, torch.float64) else
            UnboundedDiscreteTensorSpec(
                dtype=tensor.dtype, device=tensor.device, shape=tensor.shape
            )
            for key, tensor in td.items()
        },
        shape=td.shape,
    )
    return composite


class OpenMLEnv(EnvBase):
    def __init__(self, dataset_name, device="cpu", batch_size=None):
        if batch_size is None:
            batch_size = [1]
        self.dataset_name = dataset_name
        self._data = _get_data(dataset_name)
        super().__init__(device=device, batch_size=batch_size)
        self.observation_spec = make_composite_from_td(self._data[:self.batch_size.numel()])
        self.action_spec = self.observation_spec["y"].clone()
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(*self.batch_size, 1))

    def _reset(self, tensordict):
        r_id = torch.randint(self._data.shape[0], (self.batch_size.numel(),))
        data = self._data[r_id]
        return data

    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        action = tensordict["action"]
        reward = (action == tensordict["y"]).float().unsqueeze(-1)
        done = torch.ones_like(reward, dtype=torch.bool)
        return TensorDict({
            "done": done,
            "reward": reward,
            "X": tensordict["X"],
            "y": tensordict["y"],
        }, self.batch_size)

    def _set_seed(self, seed):
        self.rng = torch.random.manual_seed(seed)
