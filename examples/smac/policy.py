from torch import nn, Tensor

class MaskedLogitPolicy(nn.Module):
    def __init__(self, policy_module):
        super().__init__()
        self.policy_module = policy_module

    def forward(self, *inputs):
        *inputs, mask = inputs
        outputs = self.policy_module(*inputs)
        if isinstance(outputs, Tensor):
            outputs = (outputs,)
        # first output is logits
        outputs[0].masked_fill_(mask.expand_as(outputs[0]), -float("inf"))
        return tuple(outputs)

