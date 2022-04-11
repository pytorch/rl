.. _gettingstarted:

Getting started
===============

TorchRL aims to be a modular, primitive-first library.
Components are designed to be as self-contained as can be, and it is up to the user to create each component to make
them interact together in a training script.
To get a sense of what it means, one can glance at the (`examples`)[examples]  directory.

In general, an online RL training script requires the following elements:

- an environment building function
- a policy (and accompanying modules, e.g. value function etc.)
- a data collector
- loss module and scheduler/update
- an optimizer
- an agent that makes those component interact

Optionally, a replay buffer can also be provided to the agent.

In general, each of the front-end TorchRL API modules reads and writes `TensorDict` instances. This enables a simple
and generic high-level API of the training loop, where all the data passed from an object to another is packed in a
`TensorDict` instance:

.. image:: ./rl_pipeline.png
  :width: 400

In offline RL settings, there is no environment where to deploy the policy and a regular data collector suffies.
However, agents will expect the data collector to return a `TensorDict` instance.
This can be easily implemented as follows:

.. code:: python

    from torchrl.data import TensorDict

    dataset = make_dataset(...)

    class DatasetTDWrapper:
        def __init__(self, dataset, keys):
            self.dataset = dataset
            self.keys = keys

        def _to_td(self, values):
            return TensorDict({key: value for key, value in zip(self.keys, values)})

        def __getitem__(self, item):
            values = self.dataset[item]
            return self._to_td(values)

        def __iter__(self):
            for values in self.dataset:
                yield self._to_td(values)

        def __len__(self):
            return len(self.dataset)

    def collate_fn(tensordicts):
        return torch.stack(list(tensordicts), 0)

    dataloader = torch.utils.data.DataLoader(
        DatasetTDWrapper(dataset, keys=['key1', ...])),
        collate_fn=collate_fn, ...)


