.. note::
    If the following conditions are not met, the backward pass will use a slower but more
    memory efficient implementation:
    
    * The input is a :class:`~torch.nn.utils.rnn.PackedSequence`
    * The input is not batch first
    * ``dropout != 0``
    * ``training == True``
