# ValueEstimators

*class*torchrl.objectives.ValueEstimators(*value*, *names=None*, ***, *module=None*, *qualname=None*, *type=None*, *start=1*, *boundary=None*)[[source]](../../_modules/torchrl/objectives/utils.html#ValueEstimators)

Value function enumerator for custom-built estimators.

Allows for a flexible usage of various value functions when the loss module
allows it.

Examples

```
>>> dqn_loss = DQNLoss(actor)
>>> dqn_loss.make_value_estimator(ValueEstimators.TD0, gamma=0.9)
```