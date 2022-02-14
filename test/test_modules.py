from numbers import Number

import pytest
import torch
from torch import nn

from torchrl.data import TensorDict
from torchrl.data.tensor_specs import OneHotDiscreteTensorSpec
from torchrl.modules import QValueActor, ActorValueOperator
from torchrl.modules.models import *


@pytest.mark.parametrize("in_features", [3, 10, None])
@pytest.mark.parametrize("out_features", [3, (3, 10)])
@pytest.mark.parametrize("depth, num_cells", [(3, 32), (None, (32, 32, 32))])
@pytest.mark.parametrize("activation_kwargs", [{"inplace": True}, {}])
@pytest.mark.parametrize(
    "norm_class, norm_kwargs",
    [(nn.LazyBatchNorm1d, {}), (nn.BatchNorm1d, {"num_features": 32})],
)
@pytest.mark.parametrize("bias_last_layer", [True, False])
@pytest.mark.parametrize("single_bias_last_layer", [True, False])
@pytest.mark.parametrize("layer_class", [nn.Linear, NoisyLinear])
def test_mlp(
    in_features,
    out_features,
    depth,
    num_cells,
    activation_kwargs,
    bias_last_layer,
    norm_class,
    norm_kwargs,
    single_bias_last_layer,
    layer_class,
    seed=0,
):
    torch.manual_seed(seed)
    batch = 2
    mlp = MLP(
        in_features=in_features,
        out_features=out_features,
        depth=depth,
        num_cells=num_cells,
        activation_class=nn.ReLU,
        activation_kwargs=activation_kwargs,
        norm_class=norm_class,
        norm_kwargs=norm_kwargs,
        bias_last_layer=bias_last_layer,
        single_bias_last_layer=False,
        layer_class=layer_class,
    )
    if in_features is None:
        in_features = 5
    x = torch.randn(batch, in_features)
    y = mlp(x)
    out_features = [out_features] if isinstance(out_features, Number) else out_features
    assert y.shape == torch.Size([batch, *out_features])


@pytest.mark.parametrize(
    "layer_class",
    [
        NoisyLinear,
        NoisyLazyLinear,
    ],
)
def test_noisy(layer_class, seed=0):
    torch.manual_seed(seed)
    l = layer_class(3, 4)
    x = torch.randn(10, 3)
    y1 = l(x)
    l.reset_noise()
    y2 = l(x)
    y3 = l(x)
    torch.testing.assert_allclose(y2, y3)
    with pytest.raises(AssertionError):
        torch.testing.assert_allclose(y1, y2)


def test_value_based_policy():
    torch.manual_seed(0)
    obs_dim = 4
    action_dim = 5
    action_spec = OneHotDiscreteTensorSpec(action_dim)

    def make_net():
        net = MLP(in_features=obs_dim, out_features=action_dim, depth=2)
        for l in net.modules():
            if hasattr(l, "bias") and l.bias is not None:
                l.bias.data.zero_()
        return net

    actor = QValueActor(action_spec, mapping_operator=make_net(), safe=True)
    obs = torch.zeros(2, obs_dim)
    td = TensorDict(batch_size=[2], source={"observation": obs})
    action = actor(td).get("action")
    assert (action.sum(-1) == 1).all()

    actor = QValueActor(action_spec, mapping_operator=make_net(), safe=False)
    obs = torch.randn(2, obs_dim)
    td = TensorDict(batch_size=[2], source={"observation": obs})
    action = actor(td).get("action")
    assert (action.sum(-1) == 1).all()

    actor = QValueActor(action_spec, mapping_operator=make_net(), safe=False)
    obs = torch.zeros(2, obs_dim)
    td = TensorDict(batch_size=[2], source={"observation": obs})
    action = actor(td).get("action")
    with pytest.raises(AssertionError):
        assert (action.sum(-1) == 1).all()


def test_actorcritic():
    spec = None
    in_keys = ["obs"]
    common_mapping_operator = nn.Linear(3, 4)
    policy_operator = nn.Linear(4, 5)
    value_operator = nn.Linear(4, 1)
    op = ActorValueOperator(
        spec=spec,
        in_keys=in_keys,
        common_mapping_operator=common_mapping_operator,
        policy_operator=policy_operator,
        value_operator=value_operator,
    )
    td = TensorDict(
        source={"obs": torch.randn(4, 3)},
        batch_size=[
            4,
        ],
    )
    td_total = op(td.clone())
    policy_op = op.get_policy_operator()
    td_policy = policy_op(td.clone())
    value_op = op.get_value_operator()
    td_value = value_op(td)
    torch.testing.assert_allclose(td_total.get("action"), td_policy.get("action"))
    torch.testing.assert_allclose(
        td_total.get("action_log_prob"), td_policy.get("action_log_prob")
    )
    torch.testing.assert_allclose(
        td_total.get("state_value"), td_value.get("state_value")
    )

    value_params = set(
        list(op.value_po.parameters()) + list(op.mapping_operator.parameters())
    )
    value_params2 = set(value_op.parameters())
    assert len(value_params.difference(value_params2)) == 0 and len(
        value_params.intersection(value_params2)
    ) == len(value_params)

    policy_params = set(
        list(op.policy_po.parameters()) + list(op.mapping_operator.parameters())
    )
    policy_params2 = set(policy_op.parameters())
    assert len(policy_params.difference(policy_params2)) == 0 and len(
        policy_params.intersection(policy_params2)
    ) == len(policy_params)


if __name__ == "__main__":
    pytest.main([__file__])
