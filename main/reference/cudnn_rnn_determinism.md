Note

If the following conditions are not met, the backward pass will use a slower but more
memory efficient implementation:

- The input is a [`PackedSequence`](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence)
- The input is not batch first
- `dropout != 0`
- `training == True`