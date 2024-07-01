## Functional calls in loss modules

When profiling the loss modules, one can observe that a consdierable overehad is introduced by the
functional calls that are made over the actor and value networks.
It therefore legitimate to ask whether this couldn't be simplified by trying to remove these calls
whenever it is possible to do so.

Unfortunately, functional calls help torchrl deal with several edge cases:

- Shared parameters between actor and value networks: in these cases, we typically want to backpropagate
  through the actor loss only for the part of the parameters that is shared. Using functional calls
  allows us to detach the parameters of the common backbone within the value network in a way that
  does not impact this network directly. Consider the case where we detach the parameters within the
  backbone during the computation of the value network loss: either we have copied the backbone structure
  (which is generally not possible, some nn.Modules cannot be deepcopied) and do the detach() in that copy only,
  or we're stuck with a complex solution where we would need to keep some operations out of the compuational
  graph based on whether the parameters are shared between actor and value network.
  
  To elaborate this point, assume we want to build a loss class that takes two input networks, a value model and
  an actor: `ddpg = SomeLoss(actor, value)`. Assume also that actor and value are build as such:
  ```python
  actor = TensorDictSequential(backbone, actor_head)
  value = TensorDictSequential(backbone, value_head)
  loss = SomeLoss(actor, value)
  ```
  We wish to include the backbone parameters only in the graph of the actor loss, not the value.
  If we were not using TensorDict and just registering the actor and value in the module (
  `loss.actor = actor`
  ), we would have
  duplicated copies of the parameters:
  ```python
  param_dict = dict(loss.named_parameters())
  assert param_dict["actor.modules[0].sone_param"] is not param_dict["value.modules[0].sone_param"] # Fails!
  ```
  This is annoying. We can register the parameters using TensorDict and "hide" the actor and value network to
  solve this issue
  ```python
  loss.__dict__["actor"] = actor
  loss.__dict__["value"] = value
  loss.actor_params = TensorDictParams(TensorDict.from_module(actor))
  value_params = TensorDict.from_module(actor)
  # Remove duplicates
  set_actor_params = set(loss.actor_params.values(True, True))
  for key, value in list(value_params.items()):
      if value in set_actor_params:
         value_params.set(key, value.data)
  loss.value_params = value_params
  ```
  Now we have made sure that the value parameters and the actor parameters were not overlapping, even though
  the backbone is still shared between the two.
  If we replace these value parameters (detached and not detached) in the value network, we will overwrite
  the parameters of the backbone, which we don't want to do as it is shared with the actor. We could also
  deepcopy the value network structure and place the value params in it, but some modules cannot be deepcopied.
  An alternative option is to keep a list of the overlapping parameters and detach them (calling `requires_grad_(False)`)
  when the value loss is called but that isn't necessarily faster or safer than a functional call.

- If more than one model configuration is to be run (eg. we have two or more copies of the value network
  parameters to optimize), we have the following options:
  - Ask the user to provide all copies of the model, run them in a loop and optimize each set of params
    idependently.
  - Pass one copy of the model, deepcopy it, resample in every copy, run in a loop
  - Isolate the parameters, resample each copy, call the module with vmap + the set of parameters.
  The third solution is the most efficient for large models because it vectorizes the operations (another 
  gain will come during calls to `optimizer.step()` where math operations can be fused).

- Using TorchRL's functional API, we have put in isolation the parameters of each module as well as their
  targets in a way that makes it easy to represent them in parallel. For instance, since tensordicts can be updated
  in-place with point-wise operations, we can do
  ```python
  loss_module.target_value_params.mul_(eps).add_(loss_module.value_params.data.mul(1-eps))
  # equivalently
  loss_module.target_value_params *= eps
  loss_module.target_value_params += loss_module.value_params.data * (1-eps)
  ```
  and this will work even if there are more than one parameter configuration etc.
