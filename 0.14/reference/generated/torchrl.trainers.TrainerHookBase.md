# TrainerHookBase

*class*torchrl.trainers.TrainerHookBase[[source]](../../_modules/torchrl/trainers/trainers.html#TrainerHookBase)

An abstract hooking class for torchrl Trainer class.

*abstract*register(*trainer: [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)*, *name: str*)[[source]](../../_modules/torchrl/trainers/trainers.html#TrainerHookBase.register)

Registers the hook in the trainer at a default location.

Parameters:

- **trainer** ([*Trainer*](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)) - the trainer where the hook must be registered.
- **name** (*str*) - the name of the hook.

Note

To register the hook at another location than the default, use
`register_op()`.