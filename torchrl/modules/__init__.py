# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from .distributions import (
    Delta,
    distributions_maps,
    IndependentNormal,
    NormalParamWrapper,
    OneHotCategorical,
    TanhDelta,
    TanhNormal,
    TruncatedNormal,
)

# from .functional_modules import (
#     FunctionalModule,
#     FunctionalModuleWithBuffers,
#     extract_weights,
#     extract_buffers,
# )
from .models import (
    ConvNet,
    DdpgCnnActor,
    DdpgCnnQNet,
    DdpgMlpActor,
    DdpgMlpQNet,
    DistributionalDQNnet,
    DreamerActor,
    DuelingCnnDQNet,
    LSTMNet,
    MLP,
    NoisyLazyLinear,
    NoisyLinear,
    ObsDecoder,
    ObsEncoder,
    reset_noise,
    RSSMPosterior,
    RSSMPrior,
    Squeeze2dLayer,
    SqueezeLayer,
)
from .tensordict_module import (
    Actor,
    ActorCriticOperator,
    ActorCriticWrapper,
    ActorValueOperator,
    AdditiveGaussianWrapper,
    DistributionalQValueActor,
    EGreedyWrapper,
    OrnsteinUhlenbeckProcessWrapper,
    ProbabilisticActor,
    QValueActor,
    SafeModule,
    SafeProbabilisticModule,
    SafeSequential,
    ValueOperator,
    WorldModelWrapper,
)
from .planners import CEMPlanner, MPCPlannerBase  # usort:skip
