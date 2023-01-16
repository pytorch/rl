# -*- coding: utf-8 -*-
"""
TensorDictModule
============================
We recommand reading the TensorDict tutorial before going through this one.
"""
##############################################################################
# For a convenient usage of the ``TensorDict`` class with ``nn.Module``,
# :obj:`tensordict` provides an interface between the two named ``TensorDictModule``.
# The ``TensorDictModule`` class is an ``nn.Module`` that takes a
# ``TensorDict`` as input when called.
# It is up to the user to define the keys to be read as input and output.
#
# TensorDictModule by examples
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

###############################################################################
# Example 1: Simple usage
# --------------------------------------
# We have a ``TensorDict`` with 2 entries ``"a"`` and ``"b"`` but only the
# value associated with ``"a"`` has to be read by the network.

tensordict = TensorDict(
    {"a": torch.randn(5, 3), "b": torch.zeros(5, 4, 3)},
    batch_size=[5],
)
linear = TensorDictModule(nn.Linear(3, 10), in_keys=["a"], out_keys=["a_out"])
linear(tensordict)
assert (tensordict.get("b") == 0).all()
print(tensordict)

###############################################################################
# Example 2: Multiple inputs
# --------------------------------------
# Suppose we have a slightly more complex network that takes 2 entries and
# averages them into a single output tensor. To make a ``TensorDictModule``
# instance read multiple input values, one must register them in the
# ``in_keys`` keyword argument of the constructor.


class MergeLinear(nn.Module):
    def __init__(self, in_1, in_2, out):
        super().__init__()
        self.linear_1 = nn.Linear(in_1, out)
        self.linear_2 = nn.Linear(in_2, out)

    def forward(self, x_1, x_2):
        return (self.linear_1(x_1) + self.linear_2(x_2)) / 2


###############################################################################

tensordict = TensorDict(
    {
        "a": torch.randn(5, 3),
        "b": torch.randn(5, 4),
    },
    batch_size=[5],
)

mergelinear = TensorDictModule(
    MergeLinear(3, 4, 10), in_keys=["a", "b"], out_keys=["output"]
)

mergelinear(tensordict)

###############################################################################
# Example 3: Multiple outputs
# --------------------------------------
# Similarly, ``TensorDictModule`` not only supports multiple inputs but also
# multiple outputs. To make a ``TensorDictModule`` instance write to multiple
# output values, one must register them in the ``out_keys`` keyword argument
# of the constructor.


class MultiHeadLinear(nn.Module):
    def __init__(self, in_1, out_1, out_2):
        super().__init__()
        self.linear_1 = nn.Linear(in_1, out_1)
        self.linear_2 = nn.Linear(in_1, out_2)

    def forward(self, x):
        return self.linear_1(x), self.linear_2(x)


###############################################################################

tensordict = TensorDict({"a": torch.randn(5, 3)}, batch_size=[5])

splitlinear = TensorDictModule(
    MultiHeadLinear(3, 4, 10),
    in_keys=["a"],
    out_keys=["output_1", "output_2"],
)
splitlinear(tensordict)

###############################################################################
# When having multiple input keys and output keys, make sure they match the
# order in the module.
#
# ``TensorDictModule`` can work with ``TensorDict`` instances that contain
# more tensors than what the ``in_keys`` attribute indicates.
#
# Unless a ``vmap`` operator is used, the ``TensorDict`` is modified in-place.
#
# **Ignoring some outputs**
#
# Note that it is possible to avoid writing some of the tensors to the
# ``TensorDict`` output, using ``"_"`` in ``out_keys``.
#
# Example 4: Combining multiple ``TensorDictModule`` with ``TensorDictSequential``
# ----------------------------------------------------------------------------------
# To combine multiple ``TensorDictModule`` instances, we can use
# ``TensorDictSequential``. We create a list where each ``TensorDictModule`` must
# be executed sequentially. ``TensorDictSequential`` will read and write keys to the
# tensordict following the sequence of modules provided.
#
# We can also gather the inputs needed by ``TensorDictSequential`` with the
# ``in_keys`` property, and the outputs keys are found at the ``out_keys`` attribute.

tensordict = TensorDict({"a": torch.randn(5, 3)}, batch_size=[5])

splitlinear = TensorDictModule(
    MultiHeadLinear(3, 4, 10),
    in_keys=["a"],
    out_keys=["output_1", "output_2"],
)
mergelinear = TensorDictModule(
    MergeLinear(4, 10, 13),
    in_keys=["output_1", "output_2"],
    out_keys=["output"],
)

split_and_merge_linear = TensorDictSequential(splitlinear, mergelinear)

assert split_and_merge_linear(tensordict)["output"].shape == torch.Size([5, 13])

###############################################################################
# Example 5: Compatibility with functorch
# -----------------------------------------
# tensordict.nn is compatible with functorch. It also comes with its own functional
# utilities. Let us have a look:

import functorch

tensordict = TensorDict({"a": torch.randn(5, 3)}, batch_size=[5])

splitlinear = TensorDictModule(
    MultiHeadLinear(3, 4, 10),
    in_keys=["a"],
    out_keys=["output_1", "output_2"],
)
func, params, buffers = functorch.make_functional_with_buffers(splitlinear)
print(func(params, buffers, tensordict))

###############################################################################
# This can be used with the vmap operator. For example, we use 3 replicas of the
# params and buffers and execute a vectorized map over these for a single batch
# of data:

params_expand = [p.expand(3, *p.shape) for p in params]
buffers_expand = [p.expand(3, *p.shape) for p in buffers]
print(functorch.vmap(func, (0, 0, None))(params_expand, buffers_expand, tensordict))

###############################################################################
# We can also use the native :obj:`get_functional()` function from tensordict.nn,
# which modifies the module to make it accept the parameters as regular inputs:

from tensordict.nn import make_functional

tensordict = TensorDict({"a": torch.randn(5, 3)}, batch_size=[5])
num_models = 10
model = TensorDictModule(nn.Linear(3, 4), in_keys=["a"], out_keys=["output"])
params = make_functional(model)
# we stack two groups of parameters to show the vmap usage:
params = torch.stack([params, params.apply(lambda x: torch.zeros_like(x))], 0)
result_td = functorch.vmap(model, (None, 0))(tensordict, params)
print("the output tensordict shape is: ", result_td.shape)


from tensordict.nn import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
)

###############################################################################
# Do's and don't with TensorDictModule
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Don't use ``nn.Sequence``, similar to ``nn.Module``, it would break features
# such as ``functorch`` compatibility. Do use ``TensorDictSequential`` instead.
#
# Don't assign the output tensordict to a new variable, as the output
# tensordict is just the input modified in-place:
#
#   tensordict = module(tensordict)  # ok!
#
#   tensordict_out = module(tensordict)  # don't!
#
# ``ProbabilisticTensorDictModule``
# ----------------------------------
# ``ProbabilisticTensorDictModule`` is a non-parametric module representing a
# probability distribution. Distribution parameters are read from tensordict
# input, and the output is written to an output tensordict. The output is
# sampled given some rule, specified by the input ``default_interaction_mode``
# argument and the ``exploration_mode()`` global function. If they conflict,
# the context manager precedes.
#
# It can be wired together with a ``TensorDictModule`` that returns
# a tensordict updated with the distribution parameters using
# ``ProbabilisticTensorDictSequential``. This is a special case of
# ``TensorDictSequential`` that terminates in a
# ``ProbabilisticTensorDictModule``.
#
# ``ProbabilisticTensorDictModule`` is responsible for constructing the
# distribution (through the ``get_dist()`` method) and/or sampling from this
# distribution (through a regular ``__call__()`` to the module). The same
# ``get_dist()`` method is exposed on ``ProbabilisticTensorDictSequential.
#
# One can find the parameters in the output tensordict as well as the log
# probability if needed.

from torchrl.modules import NormalParamWrapper, TanhNormal

td = TensorDict({"input": torch.randn(3, 4), "hidden": torch.randn(3, 8)}, [3])
net = NormalParamWrapper(torch.nn.GRUCell(4, 8))
module = TensorDictModule(net, in_keys=["input", "hidden"], out_keys=["loc", "scale"])
td_module = ProbabilisticTensorDictSequential(
    module,
    ProbabilisticTensorDictModule(
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        return_log_prob=True,
    ),
)
print(f"TensorDict before going through module: {td}")
td_module(td)
print(f"TensorDict after going through module now as keys action, loc and scale: {td}")

###############################################################################
# ``Actor``
# ------------------------------
# Actor inherits from ``TensorDictModule`` and comes with a default value
# for ``out_keys`` of ``["action"]``.
#
# ``ProbabilisticActor``
# ------------------------------
# General class for probabilistic actors in RL that inherits from
# ``ProbabilisticTensorDictModule``. Similarly to ``Actor``, it comes with
# default values for the ``out_keys`` (``["action"]``).
#
# ``ActorCriticOperator``
# ------------------------------
# Similarly, ``ActorCriticOperator`` inherits from ``TensorDictSequentialand``
# wraps both an actor network and a value Network.
#
# ``ActorCriticOperator`` will first compute the action from the actor and
# then the value according to this action.

from torchrl.modules import (
    ActorCriticOperator,
    MLP,
    NormalParamWrapper,
    TanhNormal,
    ValueOperator,
)
from torchrl.modules.tensordict_module import ProbabilisticActor

module_hidden = torch.nn.Linear(4, 4)
td_module_hidden = TensorDictModule(
    module=module_hidden,
    in_keys=["observation"],
    out_keys=["hidden"],
)
module_action = NormalParamWrapper(torch.nn.Linear(4, 8))
module_action = TensorDictModule(
    module_action, in_keys=["hidden"], out_keys=["loc", "scale"]
)
td_module_action = ProbabilisticActor(
    module=module_action,
    in_keys=["loc", "scale"],
    out_keys=["action"],
    distribution_class=TanhNormal,
    return_log_prob=True,
)
module_value = MLP(in_features=8, out_features=1, num_cells=[])
td_module_value = ValueOperator(
    module=module_value,
    in_keys=["hidden", "action"],
    out_keys=["state_action_value"],
)
td_module = ActorCriticOperator(td_module_hidden, td_module_action, td_module_value)
td = TensorDict({"observation": torch.randn(3, 4)}, [3])
print(td)
td_clone = td_module(td.clone())
print(td_clone)
td_clone = td_module.get_policy_operator()(td.clone())
print(f"Policy: {td_clone}")  # no value
td_clone = td_module.get_critic_operator()(td.clone())
print(f"Critic: {td_clone}")  # no action

###############################################################################
# Other blocks exist such as:
#
# - The ``ValueOperator`` which is a general class for value functions in RL.
# - The ``ActorCriticWrapper`` which wraps together an actor and a value model
#   that do not share a common observation embedding network.
# - The ``ActorValueOperator`` which wraps together an actor and a value model
#   that share a common observation embedding network.
#
# Showcase: Implementing a transformer using TensorDictModule
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To demonstrate the flexibility of ``TensorDictModule``, we are going to
# create a transformer that reads ``TensorDict`` objects using ``TensorDictModule``.
#
# The following figure shows the classical transformer architecture
# (Vaswani et al, 2017).
#
# .. image:: /reference/generated/tutorials/media/transformer.png
#    :alt: The transformer png
#
# We have let the positional encoders aside for simplicity.
#
# Let's re-write the classical transformers blocks:


class TokensToQKV(nn.Module):
    def __init__(self, to_dim, from_dim, latent_dim):
        super().__init__()
        self.q = nn.Linear(to_dim, latent_dim)
        self.k = nn.Linear(from_dim, latent_dim)
        self.v = nn.Linear(from_dim, latent_dim)

    def forward(self, X_to, X_from):
        Q = self.q(X_to)
        K = self.k(X_from)
        V = self.v(X_from)
        return Q, K, V


class SplitHeads(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, Q, K, V):
        batch_size, to_num, latent_dim = Q.shape
        _, from_num, _ = K.shape
        d_tensor = latent_dim // self.num_heads
        Q = Q.reshape(batch_size, to_num, self.num_heads, d_tensor).transpose(1, 2)
        K = K.reshape(batch_size, from_num, self.num_heads, d_tensor).transpose(1, 2)
        V = V.reshape(batch_size, from_num, self.num_heads, d_tensor).transpose(1, 2)
        return Q, K, V


class Attention(nn.Module):
    def __init__(self, latent_dim, to_dim):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Linear(latent_dim, to_dim)

    def forward(self, Q, K, V):
        batch_size, n_heads, to_num, d_in = Q.shape
        attn = self.softmax(Q @ K.transpose(2, 3) / d_in)
        out = attn @ V
        out = self.out(out.transpose(1, 2).reshape(batch_size, to_num, n_heads * d_in))
        return out, attn


class SkipLayerNorm(nn.Module):
    def __init__(self, to_len, to_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm((to_len, to_dim))

    def forward(self, x_0, x_1):
        return self.layer_norm(x_0 + x_1)


class FFN(nn.Module):
    def __init__(self, to_dim, hidden_dim, dropout_rate=0.2):
        super().__init__()
        self.FFN = nn.Sequential(
            nn.Linear(to_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, to_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, X):
        return self.FFN(X)


class AttentionBlock(nn.Module):
    def __init__(self, to_dim, to_len, from_dim, latent_dim, num_heads):
        super().__init__()
        self.tokens_to_qkv = TokensToQKV(to_dim, from_dim, latent_dim)
        self.split_heads = SplitHeads(num_heads)
        self.attention = Attention(latent_dim, to_dim)
        self.skip = SkipLayerNorm(to_len, to_dim)

    def forward(self, X_to, X_from):
        Q, K, V = self.tokens_to_qkv(X_to, X_from)
        Q, K, V = self.split_heads(Q, K, V)
        out, attention = self.attention(Q, K, V)
        out = self.skip(X_to, out)
        return out


class EncoderTransformerBlock(nn.Module):
    def __init__(self, to_dim, to_len, latent_dim, num_heads):
        super().__init__()
        self.attention_block = AttentionBlock(
            to_dim, to_len, to_dim, latent_dim, num_heads
        )
        self.FFN = FFN(to_dim, 4 * to_dim)
        self.skip = SkipLayerNorm(to_len, to_dim)

    def forward(self, X_to):
        X_to = self.attention_block(X_to, X_to)
        X_out = self.FFN(X_to)
        return self.skip(X_out, X_to)


class DecoderTransformerBlock(nn.Module):
    def __init__(self, to_dim, to_len, from_dim, latent_dim, num_heads):
        super().__init__()
        self.attention_block = AttentionBlock(
            to_dim, to_len, from_dim, latent_dim, num_heads
        )
        self.encoder_block = EncoderTransformerBlock(
            to_dim, to_len, latent_dim, num_heads
        )

    def forward(self, X_to, X_from):
        X_to = self.attention_block(X_to, X_from)
        X_to = self.encoder_block(X_to)
        return X_to


class TransformerEncoder(nn.Module):
    def __init__(self, num_blocks, to_dim, to_len, latent_dim, num_heads):
        super().__init__()
        self.encoder = nn.ModuleList(
            [
                EncoderTransformerBlock(to_dim, to_len, latent_dim, num_heads)
                for i in range(num_blocks)
            ]
        )

    def forward(self, X_to):
        for i in range(len(self.encoder)):
            X_to = self.encoder[i](X_to)
        return X_to


class TransformerDecoder(nn.Module):
    def __init__(self, num_blocks, to_dim, to_len, from_dim, latent_dim, num_heads):
        super().__init__()
        self.decoder = nn.ModuleList(
            [
                DecoderTransformerBlock(to_dim, to_len, from_dim, latent_dim, num_heads)
                for i in range(num_blocks)
            ]
        )

    def forward(self, X_to, X_from):
        for i in range(len(self.decoder)):
            X_to = self.decoder[i](X_to, X_from)
        return X_to


class Transformer(nn.Module):
    def __init__(
        self, num_blocks, to_dim, to_len, from_dim, from_len, latent_dim, num_heads
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            num_blocks, to_dim, to_len, latent_dim, num_heads
        )
        self.decoder = TransformerDecoder(
            num_blocks, from_dim, from_len, to_dim, latent_dim, num_heads
        )

    def forward(self, X_to, X_from):
        X_to = self.encoder(X_to)
        X_out = self.decoder(X_from, X_to)
        return X_out


###############################################################################
# We first create the ``AttentionBlockTensorDict``, the attention block using
# ``TensorDictModule`` and ``TensorDictSequential``.
#
# The wiring operation that connects the modules to each other requires us
# to indicate which key each of them must read and write. Unlike
# ``nn.Sequence``, a ``TensorDictSequential`` can read/write more than one
# input/output. Moreover, its components inputs need not be identical to the
# previous layers outputs, allowing us to code complicated neural architecture.


class AttentionBlockTensorDict(TensorDictSequential):
    def __init__(
        self,
        to_name,
        from_name,
        to_dim,
        to_len,
        from_dim,
        latent_dim,
        num_heads,
    ):
        super().__init__(
            TensorDictModule(
                TokensToQKV(to_dim, from_dim, latent_dim),
                in_keys=[to_name, from_name],
                out_keys=["Q", "K", "V"],
            ),
            TensorDictModule(
                SplitHeads(num_heads),
                in_keys=["Q", "K", "V"],
                out_keys=["Q", "K", "V"],
            ),
            TensorDictModule(
                Attention(latent_dim, to_dim),
                in_keys=["Q", "K", "V"],
                out_keys=["X_out", "Attn"],
            ),
            TensorDictModule(
                SkipLayerNorm(to_len, to_dim),
                in_keys=[to_name, "X_out"],
                out_keys=[to_name],
            ),
        )


###############################################################################
# We build the encoder and decoder blocks that will be part of the transformer
# thanks to ``TensorDictModule``.


class TransformerBlockEncoderTensorDict(TensorDictSequential):
    def __init__(
        self,
        to_name,
        from_name,
        to_dim,
        to_len,
        from_dim,
        latent_dim,
        num_heads,
    ):
        super().__init__(
            AttentionBlockTensorDict(
                to_name,
                from_name,
                to_dim,
                to_len,
                from_dim,
                latent_dim,
                num_heads,
            ),
            TensorDictModule(
                FFN(to_dim, 4 * to_dim),
                in_keys=[to_name],
                out_keys=["X_out"],
            ),
            TensorDictModule(
                SkipLayerNorm(to_len, to_dim),
                in_keys=[to_name, "X_out"],
                out_keys=[to_name],
            ),
        )


class TransformerBlockDecoderTensorDict(TensorDictSequential):
    def __init__(
        self,
        to_name,
        from_name,
        to_dim,
        to_len,
        from_dim,
        latent_dim,
        num_heads,
    ):
        super().__init__(
            AttentionBlockTensorDict(
                to_name,
                to_name,
                to_dim,
                to_len,
                to_dim,
                latent_dim,
                num_heads,
            ),
            TransformerBlockEncoderTensorDict(
                to_name,
                from_name,
                to_dim,
                to_len,
                from_dim,
                latent_dim,
                num_heads,
            ),
        )


###############################################################################
# We create the transformer encoder and decoder.
#
# For an encoder, we just need to take the same tokens for both queries,
# keys and values.
#
# For a decoder, we now can extract info from ``X_from`` into ``X_to``.
# ``X_from`` will map to queries whereas ``X_from`` will map to keys and values.


class TransformerEncoderTensorDict(TensorDictSequential):
    def __init__(
        self,
        num_blocks,
        to_name,
        from_name,
        to_dim,
        to_len,
        from_dim,
        latent_dim,
        num_heads,
    ):
        super().__init__(
            *[
                TransformerBlockEncoderTensorDict(
                    to_name,
                    from_name,
                    to_dim,
                    to_len,
                    from_dim,
                    latent_dim,
                    num_heads,
                )
                for _ in range(num_blocks)
            ]
        )


class TransformerDecoderTensorDict(TensorDictSequential):
    def __init__(
        self,
        num_blocks,
        to_name,
        from_name,
        to_dim,
        to_len,
        from_dim,
        latent_dim,
        num_heads,
    ):
        super().__init__(
            *[
                TransformerBlockDecoderTensorDict(
                    to_name,
                    from_name,
                    to_dim,
                    to_len,
                    from_dim,
                    latent_dim,
                    num_heads,
                )
                for _ in range(num_blocks)
            ]
        )


class TransformerTensorDict(TensorDictSequential):
    def __init__(
        self,
        num_blocks,
        to_name,
        from_name,
        to_dim,
        to_len,
        from_dim,
        from_len,
        latent_dim,
        num_heads,
    ):
        super().__init__(
            TransformerEncoderTensorDict(
                num_blocks,
                to_name,
                to_name,
                to_dim,
                to_len,
                to_dim,
                latent_dim,
                num_heads,
            ),
            TransformerDecoderTensorDict(
                num_blocks,
                from_name,
                to_name,
                from_dim,
                from_len,
                to_dim,
                latent_dim,
                num_heads,
            ),
        )


###############################################################################
# We now test our new ``TransformerTensorDict``.

to_dim = 5
from_dim = 6
latent_dim = 10
to_len = 3
from_len = 10
batch_size = 8
num_heads = 2
num_blocks = 6

tokens = TensorDict(
    {
        "X_encode": torch.randn(batch_size, to_len, to_dim),
        "X_decode": torch.randn(batch_size, from_len, from_dim),
    },
    batch_size=[batch_size],
)

transformer = TransformerTensorDict(
    num_blocks,
    "X_encode",
    "X_decode",
    to_dim,
    to_len,
    from_dim,
    from_len,
    latent_dim,
    num_heads,
)

transformer(tokens)
tokens

###############################################################################
# We've achieved to create a transformer with ``TensorDictModule``. This
# shows that ``TensorDictModule`` is a flexible module that can implement
# complex operarations.
#
# Benchmarking
# ------------------------------

###############################################################################

to_dim = 5
from_dim = 6
latent_dim = 10
to_len = 3
from_len = 10
batch_size = 8
num_heads = 2
num_blocks = 6

###############################################################################

td_tokens = TensorDict(
    {
        "X_encode": torch.randn(batch_size, to_len, to_dim),
        "X_decode": torch.randn(batch_size, from_len, from_dim),
    },
    batch_size=[batch_size],
)

###############################################################################

X_encode = torch.randn(batch_size, to_len, to_dim)
X_decode = torch.randn(batch_size, from_len, from_dim)

###############################################################################

tdtransformer = TransformerTensorDict(
    num_blocks,
    "X_encode",
    "X_decode",
    to_dim,
    to_len,
    from_dim,
    from_len,
    latent_dim,
    num_heads,
)

###############################################################################

transformer = Transformer(
    num_blocks, to_dim, to_len, from_dim, from_len, latent_dim, num_heads
)

###############################################################################
# **Inference Time**

import time

###############################################################################

t1 = time.time()
tokens = tdtransformer(td_tokens)
t2 = time.time()
print("Execution time:", t2 - t1, "seconds")

###############################################################################

t3 = time.time()
X_out = transformer(X_encode, X_decode)
t4 = time.time()
print("Execution time:", t4 - t3, "seconds")

###############################################################################
# We can see on this minimal example that the overhead introduced by
# ``TensorDictModule`` is marginal.
#
# Have fun with TensorDictModule!
