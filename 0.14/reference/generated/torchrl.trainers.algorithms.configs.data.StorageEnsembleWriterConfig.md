# torchrl.trainers.algorithms.configs.data.StorageEnsembleWriterConfig

*class*torchrl.trainers.algorithms.configs.data.StorageEnsembleWriterConfig(*_partial_: bool = False*, *_target_: str = 'torchrl.trainers.algorithms.configs.data._make_writer_ensemble'*, *writers: list[~typing.Any] = <factory>*, *p: ~typing.Any = None*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/data.html#StorageEnsembleWriterConfig)

Hydra configuration for [`WriterEnsemble`](torchrl.data.replay_buffers.WriterEnsemble.html#torchrl.data.replay_buffers.WriterEnsemble).

This name is a historical typo for
`WriterEnsembleConfig`.
The Config is kept so existing Hydra group references do not vanish
without a deprecation cycle.

Fields match `WriterEnsembleConfig`
(`writers`, `p`). `WriterEnsemble.__init__` only accepts `*writers`;
`p` is stored for Config parity and is not a constructor argument.