# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torchrl.modules.tensordict_module._rnn_precision import (
    get_recurrent_matmul_precision,
    RecurrentMatmulPrecision,
    RecurrentMatmulPrecisionUserMode,
    set_recurrent_matmul_precision,
)
from torchrl.modules.tensordict_module.actors import (
    Actor,
    ActorCriticOperator,
    ActorCriticWrapper,
    ActorValueOperator,
    DecisionTransformerInferenceWrapper,
    DiffusionActor,
    DistributionalQValueActor,
    DistributionalQValueHook,
    DistributionalQValueModule,
    LMHeadActorValueOperator,
    MultiStepActorWrapper,
    ProbabilisticActor,
    QValueActor,
    QValueHook,
    QValueModule,
    TanhModule,
    ValueOperator,
)
from torchrl.modules.tensordict_module.common import SafeModule, VmapModule
from torchrl.modules.tensordict_module.exploration import (
    AdditiveGaussianModule,
    AdditiveGaussianWrapper,
    EGreedyModule,
    EGreedyWrapper,
    OrnsteinUhlenbeckProcessModule,
    OrnsteinUhlenbeckProcessWrapper,
    RandomPolicy,
)
from torchrl.modules.tensordict_module.probabilistic import (
    SafeProbabilisticModule,
    SafeProbabilisticTensorDictSequential,
)
from torchrl.modules.tensordict_module.rnn import (
    canonicalize_rnn_subset,
    GRU,
    GRUCell,
    GRUModule,
    LSTM,
    LSTMCell,
    LSTMModule,
    recurrent_mode,
    set_recurrent_mode,
)
from torchrl.modules.tensordict_module.sequence import SafeSequential
from torchrl.modules.tensordict_module.world_models import WorldModel, WorldModelWrapper

__all__ = [
    "Actor",
    "ActorCriticOperator",
    "ActorCriticWrapper",
    "ActorValueOperator",
    "DecisionTransformerInferenceWrapper",
    "DiffusionActor",
    "DistributionalQValueActor",
    "DistributionalQValueHook",
    "DistributionalQValueModule",
    "LMHeadActorValueOperator",
    "MultiStepActorWrapper",
    "ProbabilisticActor",
    "QValueActor",
    "QValueHook",
    "QValueModule",
    "TanhModule",
    "ValueOperator",
    "SafeModule",
    "VmapModule",
    "AdditiveGaussianModule",
    "AdditiveGaussianWrapper",
    "EGreedyModule",
    "EGreedyWrapper",
    "RandomPolicy",
    "OrnsteinUhlenbeckProcessModule",
    "OrnsteinUhlenbeckProcessWrapper",
    "SafeProbabilisticModule",
    "SafeProbabilisticTensorDictSequential",
    "canonicalize_rnn_subset",
    "GRU",
    "GRUCell",
    "GRUModule",
    "LSTM",
    "LSTMCell",
    "LSTMModule",
    "RecurrentMatmulPrecision",
    "RecurrentMatmulPrecisionUserMode",
    "get_recurrent_matmul_precision",
    "recurrent_mode",
    "set_recurrent_matmul_precision",
    "set_recurrent_mode",
    "SafeSequential",
    "WorldModel",
    "WorldModelWrapper",
]
