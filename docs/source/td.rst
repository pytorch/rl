.. td:

TensorDicts
===========

TensorDict is a new tensor container introduced in TorchRL. Its main purpose is
to seamlessly carry an undefined number of named tensors across objects, functions
or even processes. Let us take the following example: In RL, we typically have
actors that take actions based on (a set of) observations:

.. code:: python

  >>> action = actor(observation)


where ``action`` and ``observation`` are tensors. Sometimes, it might be the case that
the actor also returns the log-probability of the action under the policy:

.. code:: python

  >>> action, action_log_prob = actor(observation)


In general, if one wants to build a generic data collector for different scenarios,
it is impossible to encompass in advance every single use case that might occur.
For instance, it is common also for the actor to return an RNN state. The
environment's behaviour too can't be generally forecast: it might occur that
more than one observation is returned and used by the policy etc. One solution
could be to cast the outputs to hierarchical structures (tuple, list, dictionaries)
and many libraries already offer such tools.

In TorchRL, we make the following observations:

* modules (envirnoments, actors, replay buffers, batchers, data collectors) may
  return an arbitrary number of tensors;
* Working with regular tuple or unnamed data structures can be both confusing an
  dangerous;
* Output *dtypes* can vary across tensors, but in general a single operation will
  return tensors on the *same device*   and on the *same memory location*
  (physical memory, shared memory, RAM, cuda);
* In general, algebraic operations are meaningless over a bag of tensors collected
  at a point in time. For instance, it would make little sense to perform an addition
  over ``action`` and ``action_log_prob`` at the same time;

Typical use case
----------------

In many RL settings, executing a step in the environment is a task executed on CPU with little or no possiblity of vectorization. For most DL solutions, the actor step is ideally executed on GPU, where we would gain in executing the step once rather than repeating the same query across environments. The most natural solution is therefore to execute all the environment step queries at once (either serially or in parallel), collect the outputs (observations, reward, done state and possibly others), and then query an actor step.

.. code:: python

  >>> def rollout():
  ...     observations = some_reset_operation()
  ...     out_action = []
  ...     out_obs = []
  ...     out_reward = []
  ...     out_done = []
  ...     while some_condition():
  ...        action = actor(observations)
  ...        observations = []
  ...        rewards = []
  ...        dones = []
  ...        for env in env_list:
  ...            obs, reward, done, *others = env.step(action[i])
  ...            observations.append(obs)
  ...            rewards.append(reward)
  ...            dones.append(done)
  ...        out_obs.append(torch.stack(observations, 0))
  ...        out_action.append(action)
  ...        out_reward.append(torch.stack(rewards, 0))
  ...        out_done.append(torch.stack(dones, 0))
  ...     out_obs = torch.stack(out_obs, 0)
  ...     out_action = torch.stack(out_action, 0)
  ...     out_reward = torch.stack(out_reward, 0)
  ...     out_done = torch.stack(out_done, 0)
  ...     return out_obs, out_action, out_reward, out_done


(we simplified a bit the collection loop as the first observation is not collected
at the end of the function, ideally we would need to get it too). This kind
of loop quicly becomes tedious to read. It is also hard to generalise it to every
single use case (such as those detailed above). We propose to sugar-coat this code by using ``TensorDict``:

.. code:: python

  >>> def rollout():
  ...     out = []
  ...     tensordict = some_reset_operation()
  ...     while some_condition():
  ...         tensordict = actor(tensordict)
  ...         for env in env_list:
  ...             tensordict[i] = env(tensordict[i])
  ...         out.append(tensordict.clone())
  ...     return torch.stack(out, 0)


This loop has two advantages:

#. It encompasses almost (if not) all the use cases one could think of with the
   usual single agent training loop. Of course it requires the user to code the
   behaviour of the actor and environment when they receive a tensordict, but
   in most cases a simple wrapper will do the trick:
   .. code:: python

     >>> class ActorWrapper(ActorClass):
     ...     def __init__(self, actor):
     ...         self.actor = actor
     ...
     ...     def forward(self, tensordict):
     ...         observation = tensordict.get("observation")
     ...         action = self.actor(observation)
     ...         tensordict.set("action", action)
     ...         return tensordict
#. It is easier to work with its output. Recall that data collection step is
   supposed to be independent of the training loop, in the sense that it should
   be the same whether a PPO (which requires an ``action_log_prob`` tensor) or a DQN
   (which doesn't) algorithm is used for instance.

TensorDict properties
---------------------

Before going in more details in the description of what a ``TensorDict`` can be
used for, let us pause for a moment and describe the properties of this class.

In general, a ``TensorDict`` can be characterized by

#. Its set of key-value pairs (``tensordict.items()`` or ``tensordict.to_dict()``);
#. Its batch size (``tensordict.batch_size``) which returns a ``torch.Size``
   object with the common first dimensions of all the tensors it contains
   (importantly, this is not inferred automatically by looking at the tensors,
   but has to be set manually when creating the ``TensorDict``:
   ``tensordict = TensorDict(source=source, batch_size=[K,N]))``;
#. Its device and/or memory location (cuda, cpu, shared memory, memmap numpy
   array, file on disk).

As mentioned above, it is not expected that all the items share the same dtype,
neither is it assumed that they share the same shape (besides the first
``N`` dimensions, where ``N = tensordict.batch_dims``).

A ``TensorDict`` supports many operations, such as

#. Setting a new key or overwriting an existing key: ``tensordict.set(key, value)``
   and ``tensordict.set_(key, value)``;
#. Getting the value of a key: ``tensordict.get(key)``
#. indexing / masking: ``tensordict = tensordict[idx]``, which can be done only
   in the batch dimensions (i.e. indexing a ``TensorDict`` with an empty batch
   size is not allowed, even if all of its tensors has a size that would allow it).
   Updating a tensordict at some index is also supported, provided that the keys
   and shape match: ``td[idx] = other_td``.
#. Similarly, the following operations are supported *if they comply with the
   batch size*: ``torch.cat``, ``undind(dim)``, ``view(*shape)``, ``squeeze(dim)``,
   ``expand(*shape)`` (the following methods are not yet supported but are planned:
   ``permute``, ``transpose``, ``reshape``, ``repeat``, ``flatten``);
#. Along the same line, ``tensordict.set_at_(key, value, index)`` will write
   the value at the index provided by index if it complies with the batch_size;
#. Stacking of tensordicts along a dimension using ``torch.stack``;
#. cloning: ``tensordict.clone()``
#. updating ``tensordict.update(other)``. Similarly to ``set`` and ``set_``, if
   one wants to make sure that an appropriate exception is
   raised when an existing tensor differs in nature (dtype or shape) from the input
   tensor or if a key does not exist yet, ``tensordict.update_(other)`` can also be used;
#. ``zero_``-ing or ``fill_``-ing (mainly for testing purposes);
#. placing in shared memory: ``tensordict.share_memory_()`` or writing to a memmap
   file: ``tensordict.memmap_()``;
#. Casting to device ``tensordict.to(device)`` or to another tensordict type
   (see below): ``tensordict.to(OtherTensorDictClass)`` (as can be expected,
   ``tensordict.to(dtype)`` will raise an exception);
#. Renaming keys (``tensordict.rename(old_name, new_name)``), selecting a subset
   of keys (``tensordict.select(*keys)``), excluding some keys
   (``tensordict.exclude(*keys)``) or deleting keys (``tensordict.del_``);
#. changing batch size: ``tensordict.batch_size = new_batch_size`` (provided
   the new batch size is compatible with the tensor shapes).

All algebraic operations (``__add__``, ``__div__``, ``__neg__`` etc.) are not
supported by design.

Multiprocessing and shared memory
---------------------------------

A useful property of ``TensorDict`` is that it enables easy access to shared
memory. In fact, a tensordict supports ``tensordict.share_memory_()``, which
casts all of its tensors to share memory. This is extremely handy when working
with multiple processes.

In multiprocessing, it is usually better (if possible) to re-use the same
buffer over and over when passing data from one process to the other. Also, it
is sometimes better to use a single large container (e.g. one tensor) that is
shared amongst all the processes rather than having a single buffer for each
individual process. The reason for this is that one might easily end up in a
situation where too many files are open if processes and tensors to be shared
are numerous. However, one should also keep in mind that reading and writing
from and to shared memory tensors are mildly expensive operations.

A typical data collection loop using multiprocessing using a shared ``TensorDict``
would therefore look like this (discarding reset or seeding operations for convenience):

.. code:: python

  >>> def main(num_processes, num_steps, policy, ...):
  ...     # init: create container TensorDict and tensors:
  ...     tensor_dict_container = TensorDict(source={
  ...         "observation": observation_empty_tensor,
  ...         "action": action_empty_tensor,
  ...         "reward": reward_empty_tensor,
  ...         "done": done_empty_tensor,
  ...     }, batch_size=[num_processes])
  ...     tensor_dict_container.share_memory_()
  ...     # launch processes
  ...     pipes = []
  ...     procs = []
  ...     for i in range(num_processes):
  ...         parent_pipe, child_pipe = mp.Pipe()
  ...         pipes.append(parent_pipe)
  ...         # an indexed tensordict sends the reference to the appropriate memory location to each worker, hence
  ...         # we can simply send that object to each worker for them to write to their assigned shared sub-tensor.
  ...         proc = mp.Process(target=worker_fn, args=(parent_pipe, child_pipe, tensor_dict_container[i]))
  ...         proc.daemon = True
  ...         proc.start()
  ...         child_pipe.close()
  ...         procs.append(proc)
  ...     out_tensor_dict = []
  ...     for i in range(num_steps):
  ...         # compute action and send it to workers via the shared tensor_dict (assuming policy the tensordict and the
  ...         # policy are on the same device)
  ...         policy(tensor_dict_container)
  ...         for worker_id in range(num_processes):
  ...             pipes[worker_id].send("step")
  ...             # cloning a shared tensor returns a non-shared tensor (if on CPU)
  ...         for worker_id in range(num_processes):
  ...             msg = pipes[worker_id].recv()
  ...             assert msg == "step done"
  ...         out_tensor_dict.append(tensor_dict_container.clone())
  ...     out_tendor_dict = torch.stack(out_tensor_dict, 1)
  ...     for worker_id in range(num_processes):
  ...         pipes[worker_id].send("close")
  ...     for worker_id in range(num_processes):
  ...         procs[worker_id].join()
  ...         pipes[worker_id].close()
  ...     return out_tendor_dict
  >>>
  >>> def worker_fn(parent_pipe, child_pipe, tensor_dict_container,
  ...     env_constructor_fn):
  ...   env = env_constructor_fn()  # creates env
  ...   parent_pipe.close()
  ...   while True:
  ...       cmd = child_pipe.recv()
  ...       if cmd == "step":
  ...           env.step(tensor_dict_container)  # writes the results directly onto the sharerd memory tensors
  ...           child_pipe.send("step done")
  ...       elif cmd == "close":
  ...           env.close()
  ...           child_pipe.close()
  ...           break


Here the advantage of the ``TensorDict`` class is mainly code readability:
without that, all the read and write operations should be done via tensors
passed to the worker process independently, e.g.:

.. code:: python

  >>> def main(num_processes, num_steps, policy, ...):
  ...     observation_container.share_memory_()
  ...     action_container.share_memory_()
  ...     reward_container.share_memory_()
  ...     done_container.share_memory_()
  ...     ...
  ...     for i in range(num_processes):
  ...         ...
  ...         proc = mp.Process(target=worker_fn, args=(parent_pipe, child_pipe,
  ...                                                   observation_container[i],
  ...                                                   action_container[i],
  ...                                                   reward_container[i],
  ...                                                   done_container[i]))
  ...     ...
  ...     for i in range(num_steps):
  ...         action = policy(observation_container)
  ...         ...
  ...         actions.append(action.clone())
  ...         obs.append(observation_container.clone())
  ...         rewards.append(reward_container.clone())
  ...         dones.append(done_container.clone())
  ...     # further stacking operations ...
  ...     return obs, actions, rewards, dones
  >>>
  >>>
  >>> def worker_fn(parent_pipe, child_pipe, observation_container,action_container, reward_container, done_container):
  ...     ...
  ...     while True:
  ...         ...
  ...         if cmd == "step":
  ...             _obs, _reward, _done, *_ = env.step(action_container)
  ...             observation_container.copy_(_obs)
  ...             reward_container.copy_(_reward)
  ...             done_container.copy_(_done)
  ...             ...


One can easily imagine how much code readability is impacted.
Again, this solution is not highly general, as env and policy may have very
different signatures.

Stacking TensorDicts
--------------------

We have seen that being able to 'stack' tensordicts together comes in handy
whenever we are collecting them in a loop (across trajectory steps or processes
for instance). We provide a way to make this operation in a 'lazy' manner, i.e.
display a list of ``TensorDict`` objects as being 'stacked' while keeping them
separated in memory. This is actually the default behaviour of torch.stack when
called upon a list of TensorDicts:

.. code:: python

  >>> tensor_dict_list = [TensorDict(source={'a': torch.rand(3, 4, 5)}, batch_size=[3, 4]) for i in range(10)]
  >>> td_stack = torch.stack(tensor_dict_list, dim=1)
  >>> print("stack: ", td_stack)
  >>> print("indexed stack gives original tensor_dict: ", td_stack[:, 0] is tensor_dict_list[0])


which outputs

.. code:: python

  stack: LazyStackedTensorDict(
      fields={a: Tensor(torch.Size([3, 10, 4, 5]), dtype=torch.float32)},
      batch_size=torch.Size([3, 10, 4]),
      device=cpu)
  indexed stack gives original tensor_dict: True


If one wants to 'execute' the stack command for a single key-value pair,
calling the ``get(key)`` method will work:

.. code:: python

  >>> print(td_stack.get("a"))  # prints a tensor of shape [3, 10, 4, 5]


Similarly, calling ``td_stack.clone()`` will return a ``TensorDict``
object with the stacked values stored in it.

An interesting feature of this lazy stacking is that modifications to the original
tensordicts will be reflected in the stack, and vice-versa:

.. code:: python

  >>> tds = [TensorDict(source={'a': torch.randn(3, 4)}, batch_size=[3]) for _ in range(5)]
  >>> td_stack = torch.stack(tds, 1)
  >>> print('original: ', td_stack)
  >>> td_stack.set("b", torch.zeros(3, 5, 1))
  >>> print('with new b key: ', td_stack)
  >>> print('first element of tds has the new key: ', tds[0])  #  first element of the stack has a new "b" key
  >>> for _td in tds:
  ...     _td.set("c", torch.ones(3, dtype=torch.bool))
  >>> print('stack has new c key: ', td_stack)  #  td_stack has a new "c" key
  >>> tds[0].set("d", torch.zeros(3, dtype=torch.double))
  >>> print('stack is unchanged: ', td_stack)  #  td_stack has no new key as only one of its elements has key "d"


The following output should follow:

.. code:: python

  original:  LazyStackedTensorDict(
      fields={a: Tensor(torch.Size([3, 5, 4]), dtype=torch.float32)},
      batch_size=torch.Size([3, 5]),
      device=cpu)
  with new b key:  LazyStackedTensorDict(
      fields={a: Tensor(torch.Size([3, 5, 4]), dtype=torch.float32),
          b: Tensor(torch.Size([3, 5, 1]), dtype=torch.float32)},
      batch_size=torch.Size([3, 5]),
      device=cpu)
  first element of tds has the new key:  TensorDict(
      fields={a: Tensor(torch.Size([3, 4]), dtype=torch.float32),
          b: Tensor(torch.Size([3, 1]), dtype=torch.float32)},
      shared=False,
      batch_size=torch.Size([3]))
  stack has new c key:  LazyStackedTensorDict(
      fields={a: Tensor(torch.Size([3, 5, 4]), dtype=torch.float32),
          b: Tensor(torch.Size([3, 5, 1]), dtype=torch.float32),
          c: Tensor(torch.Size([3, 5]), dtype=torch.bool)},
      batch_size=torch.Size([3, 5]),
      device=cpu)
  stack is unchanged:  LazyStackedTensorDict(
      fields={a: Tensor(torch.Size([3, 5, 4]), dtype=torch.float32),
          b: Tensor(torch.Size([3, 5, 1]), dtype=torch.float32),
          c: Tensor(torch.Size([3, 5]), dtype=torch.bool)},
      batch_size=torch.Size([3, 5]),
      device=cpu)


This feature exists to make it easy to execute operations on a list of ``TensorDict``
without needing to allocate more memory in the process.
For instance, it is common to store TensorDicts obtained at individual time steps in a replay buffer:

.. code:: python

  >>> tensor_dict_traj = []
  >>> for t in range(num_steps):
  ...     policy(tensor_dict)
  ...     env.step(temsor_dict)
  ...     tensor_dict_traj.append(tensor_dict.clone())
  >>> tensor_dict_traj = torch.stack(tensor_dict_traj, 0)
  >>> # execute some operations on tensor_dict_traj, such as multi_step rewards or other batching
  >>> replay_buffer.add(tensor_dict_traj)
  >>> batch_of_tensordicts = replay_buffer.sample(
  ...     16)  # returns a stack of 16 individual tensor_dicts that can be used to train the policy
  >>> ...  # compute loss, call backward(), optim.step() etc.
  >>> replay_buffer.update_priority(batch_of_tensordicts)


The last step update the priority assigned to each item in the sample, based on
its contribution to the loss (for instance, items with a higher MSE will be
assigned a higher priority for the next sample operation). All of this is done 'in-place',
i.e. when the loss is computed, the corresponding value in the target TensorDicts
is updated accordingly. The whole process keeps track of the original tensordict
and the number of copies in memory is kept to a minimum.

If we had stored the stack as a set of contiguous tensors, we would have had to
allocate the memory for it. Then when placing it in the replay buffer, we would
have had to either (1) index the stacked tensordict for each of its items,
keeping for each of them a reference to the original stack that would only be
cleared once the full stack is cleared:

.. code:: python

  >>> tensor_dict_traj = torch.stack(tensor_dict_traj, 0).clone()
  >>> # execute some operations on tensor_dict_traj, such as multi_step rewards or other batching
  >>> for t in range(num_steps)
  ...     replay_buffer.add(tensor_dict_traj[t])  # keeps a reference to tensor_dict_traj
  >>> sample = replay_buffer.sample(16)  # returns a stack of 16 individual tensor_dicts that can be used to train the policy


or (2) do this but create a copy of the indexed tensordicts, and allocate new
memory with this operation:

.. code:: python

  >>> tensor_dict_traj = torch.stack(tensor_dict_traj, 0).clone()
  >>> # execute some operations on tensor_dict_traj, such as multi_step rewards or other batching
  >>> for t in range(num_steps)
  ...     replay_buffer.add(tensor_dict_traj[t].clone())  # keeps a reference to tensor_dict_traj
  >>> sample = replay_buffer.sample(16)  # returns a stack of 16 individual tensor_dicts that can be used to train the policy


In practice, the efficiency of these 3 operations obviously depends on what is
executed in the hidden batcher step (i.e. how efficiently the multi-step operation would be coded).

Another instance where this feature is useful is when we are placing a stack of
tensordicts in a memory buffer:

.. code:: python

  >>> tensor_dict_traj = []
  >>> for t in range(num_steps):
  ...     policy(tensor_dict)
  ...     env.step(temsor_dict)
  ...     tensor_dict_traj.append(tensor_dict.clone())
  >>> tensor_dict_traj = torch.stack(tensor_dict_traj, 0)
  >>> # place in shared memory buffer
  >>> tensor_dict_shared_memory.copy_(tensor_dict_traj)


Under the hood, the ``copy_`` operation will iterate through the keys of the
``LazyStackedTensorDict``, execute the stack operation, copy the resulting
tensor on the shared memory correspondent and clear the stacked tensor from
memory. Again let us look at the alternative:

.. code:: python

  >>> for t in range(num_steps):
  ...     policy(tensor_dict)
  ...     env.step(temsor_dict)
  ...     # place in shared memory buffer, no stacking
  ...     tensor_dict_shared_memory[t].copy_(tensor_dict)


In this way, we access the shared memory buffer as many times as the number of
steps. In the ``test_shared.py``, we provide an example of this where it is
apparent that this solution is by far slower than the one proposed above.

Another viable solution might be to go through the stacking operation first,
then copy all the tensors in shared memory:

.. code:: python

  >>> tensor_dict_traj = []
  >>> for t in range(num_steps):
  ...     policy(tensor_dict)
  ...     env.step(temsor_dict)
  ...     tensor_dict_traj.append(tensor_dict.clone())
  >>> tensor_dict_traj = torch.stack(tensor_dict_traj, 0).clone()  # non-lazy stacking
  >>> # place in shared memory buffer
  >>> tensor_dict_shared_memory.copy_(tensor_dict_traj)


Here too, the memory occupied by the single TensorDicts collected in the
loop is cleared after the ``clone()`` method is called, hence it requires no more
memory than the first option. Finally, note that ``cat`` will call the regular
``torch.cat`` operation on every value in the input tensordicts.

Saving TensorDicts on disk
--------------------------

We also provide a ``SavedTensorDict`` class. It represents an interface to work
with tensordicts that are saved on the disk (via ``torch.save``) in a
temporary file. Creating a ``SavedTensorDict`` is fairly easy:

.. code:: python

  >>> from torchrl.data import TensorDict, SavedTensorDict
  >>> import torch
  >>> import os
  >>> import sys
  >>>
  >>> tensor_dict_list = [TensorDict(source={'a': torch.rand(3, 4, 100)}, batch_size=[3, 4]) for i in range(10)]
  >>> tensor_dict_stack = torch.stack(tensor_dict_list, 0)
  >>> tensor_dict_list_saved = tensor_dict_stack.to(SavedTensorDict)
  >>> print("saved: ", tensor_dict_list_saved)
  >>> filename = tensor_dict_list_saved.filename  #  file where the tensordict is saved
  >>>
  >>>  #  check that values match
  >>> torch.testing.assert_allclose(tensor_dict_list_saved.get("a"), tensor_dict_stack.get("a"))  #  passes
  >>> print("re-loaded: ", tensor_dict_list_saved._load())  # this should be a LazyStackedTensorDict
  >>>
  >>>  # check size of object, recursively
  >>> print("size in memory of original tensordict: ", getsize(tensor_dict_stack))  #  getsize << https://stackoverflow.com/a/30316760/4858862
  >>> print("size in memory of saved tensordict: ", getsize(tensor_dict_list_saved))  # size does not depend on the tensordict content!
  >>>
  >>> print('exists before del: ', os.path.isfile(filename))  #  True
  >>> del tensor_dict_list_saved  #  garbage collected
  >>> print('exists before del: ', os.path.isfile(filename))  #  False


which should output

.. code:: python

  SavedTensorDict(
      fields={a: Tensor(torch.Size([10, 3, 4, 100]), dtype=torch.float32)},
      batch_size=torch.Size([10, 3, 4]),
      file=/var/folders/zs/9lq15k8x61l1g0c_sf__63c80000gn/T/tmpt99po0du)
  LazyStackedTensorDict(
      fields={a: Tensor(torch.Size([10, 3, 4, 100]), dtype=torch.float32)},
      batch_size=torch.Size([10, 3, 4]),
      device=cpu)
  size in memory of original tensordict:  13632
  size in memory of saved tensordict:  7240
  exists before del:  True
  deleting
  exists before del:  False


The ``SavedTensorDict`` class relied on the ``tempfile`` library to create
the temporary files.
Note that ``tempfile.TemporaryFile`` instances aren't usually pickable,
hence serialization across processes may be an issue.
To cirumvent this, we pass only the reference to the filename string when
serializing an instance of ``SavedTensorDict`` and the clearing of this object
is left to the responsibility of the main process (this might obviously lead
to some bugs and suggestions for improvement are welcome).
``SavedTensorDict`` makes it easy to save big tensors on disk, and can be used
seamlessly in a replay buffer for instance.
As they occupy virtually no space in memory, we can store many of these in
the RB, sample them at each step of the optimization and load them in memory
when needed.

``SavedTensorDict`` saves the whole set of tensors together (it basically
serializes the original TensorDict). Alternatively, we also offer a
``tensordict.memmap_()`` functionality, that maps the tensors in the tensordict
onto a ``numpy.memmap`` array. This works pretty much like ``share_memory_()``:

.. code:: python

  >>> from torchrl.data import TensorDict, SavedTensorDict
  >>> import torch
  >>> import os
  >>> import sys
  >>>
  >>> tensor_dict_memmap = TensorDict(source={'a': torch.rand(3, 4, 100)}, batch_size=[3, 4])
  >>> tensor_dict = tensor_dict_memmap.clone()
  >>> tensor_dict_memmap = tensor_dict_memmap.memmap_()
  >>> print("saved: ", tensor_dict)
  >>> filename = tensor_dict_memmap.get("a").filename
  >>> print(type(tensor_dict_memmap.get("a")))  # MemmapTensor is the interface between numpy.memmap and torch.tensor
  >>>
  >>> # check that values match
  >>> torch.testing.assert_allclose(tensor_dict.get("a"), tensor_dict_memmap.get("a").clone())  # passes
  >>> print("re-loaded: ", tensor_dict_memmap.clone())  # this should be a LazyStackedTensorDict
  >>>
  >>> # check size of object, recursively
  >>> print("size in memory of original tensordict: ",
  ...      getsize(tensor_dict))  # getsize << https://stackoverflow.com/a/30316760/4858862
  >>> print("size in memory of saved tensordict: ",
  ...      getsize(tensor_dict_memmap))  # size does not depend on the tensordict content!
  >>> print('exists before del: ', os.path.isfile(filename))  # True
  >>> del tensor_dict_memmap  # garbage collected
  >>> print('exists before del: ', os.path.isfile(filename))  # False


which output is

.. code:: python

  saved:  TensorDict(
      fields={a: Tensor(torch.Size([3, 4, 100]), dtype=torch.float32)},
      shared=False,
      batch_size=torch.Size([3, 4]))
  <class 'torchrl.data.tensordict.memmap.MemmapTensor'>
  re-loaded:  TensorDict(
      fields={a: Tensor(torch.Size([3, 4, 100]), dtype=torch.float32)},
      shared=False,
      batch_size=torch.Size([3, 4]))
  size in memory of original tensordict:  2249
  size in memory of saved tensordict:  50162
  exists before del:  True
  exists before del:  False


We hope that this interface with ``np.memmap`` will make it easy to store
big tensors (e.g. videos in pixel-based RL) in settings where only some indices
of these tensors need to be accessed. For instance, in `Dreamer`_, one stores
entire trajectories of example behaviours as videos, then sub-samples consecutive
ranges of, say, 50 images at random to train the algorithm.
Using ``np.memmap`` allows us to do this sampling without requiring us to define
the start and stop index in advance.

.. _DREAMER: https://arxiv.org/abs/1912.01603

As of now, none of these two classes (``SavedTensorDict`` or ``MemmapTensor``)
supports ``TensorDict``s that contain tensors that require gradients.

update, update_,  __getitem__ and __setitem__
---------------------------------------------

The ``TensorDict`` class supports indexing *along the batch dimensions*:

.. code:: python

  >>> td = TensorDict(source={'a': torch.randn(3, 4, 5)}, batch_size=[3, 4])
  >>> print(td[:, 2])  # passes
  >>> print(td[:, :, 2])  # raises an exception


Similarly, one can update a tensordict using the ``__setindex__`` functionality:

.. code:: python

  >>> td[:, 0] = TensorDict(souce={'a', torch.randn(3, 5)}, batch_size=[3])


One can update a tensordict with *new* keys and/or replace keys by using ``update()``:

.. code:: python

  >>> td = TensorDict(source={'a': torch.randn(3, 4, 5)}, batch_size=[3, 4])
  >>> td_new = TensorDict(source={'a': torch.ones(3, 4, 5), 'b': torch.ones(3, 4, 10)}, batch_size=[3, 4])
  >>> td.update(td_new)  # passes, 'a' overwritten in-place
  >>> td.update(td_new, inplace=True)  # passes, 'a' overwritten in-place (same as above)
  >>> assert 'b' in td.keys()  # passes, "b" added to td keys
  >>>
  >>> td_new = TensorDict(source={'a': torch.ones(3, 4, 1), 'b': torch.ones(3, 4, 10)}, batch_size=[3, 4])
  >>> td.update(td_new, inplace=True)  # fails, 'a' differs in shape
  >>> td.update(td_new, inplace=False)  # passes, 'a' replaced


The ``update_`` method is obviously a lot more restrictive:

.. code:: python

  >>> td = TensorDict(source={'a': torch.randn(3, 4, 5)}, batch_size=[3, 4])
  >>> td_new = TensorDict(source={'a': torch.ones(3, 4, 5)}, batch_size=[3, 4])
  >>> td.update_(td_new)  # passes
  >>>
  >>> td = TensorDict(source={'a': torch.randn(3, 4, 5)}, batch_size=[3, 4])
  >>> td_new = TensorDict(source={'a': torch.ones(3, 4, 5), 'b': torch.ones(3, 4, 10)}, batch_size=[3, 4])
  >>> td.update_(td_new)  # fails, 'b' non present
  >>>
  >>> td = TensorDict(source={'a': torch.randn(3, 4, 1), 'b': torch.zeros(3, 4, 10)}, batch_size=[3, 4])
  >>> td.update_(td_new)  # fails, 'a' shapes do not match
  >>>
  >>> td = TensorDict(source={'a': torch.randn(3, 4, 5), 'b': torch.ones(3, 4, 10)}, batch_size=[3, 4])
  >>> td_new = TensorDict(source={'a': torch.ones(3, 4, 5)}, batch_size=[3, 4])
  >>> td.update_(td_new)  # passes: all of the td_new's keys are present in td


Reshaping
---------

One can call ``tensordict.view(*new_shape)`` provided that new_shape is an
iterable with a shape that is compatible with the batch size of the tensordict.
For instance, ``td = TensorDict(source={}, batch_size=[3,4]).view(-1)`` is valid,
whereas ``td = TensorDict(source={'a': torch.randn(3,4,2)}, batch_size=[3,4]).view(24)``
will raise an error even though the shape is compatible with the number of elements in the tensor ``a``.
We can also call ``unsqueeze`` and ``squeeze`` but, again, one should keep in
mind that valid dimensions are defined by the batch size only.

As for their ``torch.Tensor`` counterparts, those operations are
memory-efficient and do not require any new storage. Moreover, they allow for
in-place modification of the original tensordict. Here are some examples:

.. code:: python

  >>> t = TensorDict({'a': torch.randn(3, 4)}, [3])
  >>> t.view(-1).set('b', torch.randn(12));
  >>> print(t)
  TensorDict(
      fields={
          a: Tensor(torch.Size([3, 4]), dtype=torch.float32),
          b: Tensor(torch.Size([3, 4]), dtype=torch.float32)},
      batch_size=torch.Size([3]),
      device=cpu,
      is_shared=False)
  >>> t.view(-1).fill_('a', 0.0);
  >>> print(t.get('a'))
  tensor([[0., 0., 0., 0.],
          [0., 0., 0., 0.],
          [0., 0., 0., 0.]])


Selecting, renaming, deleting keys
----------------------------------

It can sometimes be useful to rename a key or to delete it. A typical example
is observed during a trajectory loop: the call to ``env.step(tensordict)`` will
write a ``"next_observation"`` key into the tensordict.
For the next iteration, the ``"next_observation"`` will be renamed observation
for it to be read by the policy:

.. code:: python

  >>> def rollout():
  ...     tensordict = TensorDict(source={'observation': init_observation_tensor}, batch_size=[])
  ...     out = []
  ...     for i in range(num_steps):
  ...         policy(tensordict)  # reads the 'observation' key and produces a resulting action
  ...         env.step(tensordict)  # reads the `action` key and produces "next_observation", "reward", "done" etc.
  ...         out.append(tensordict.clone())
  ...         tensordict.del_('observation')
  ...         tensordict.rename_key('next_observation', 'observation')
  ...     return torch.stack(out, 0)


Notice that the ordering of calling the policy and ``env.step`` is important as
we want to 'tie together' the original observation with the associated action and reward ( an {S,A,S',R} tuple).

Sometimes, we might want to view only a part of a tensordict.
For instance, we might want to return only the key-value pairs that have been affected by an operation:

.. code:: python

  >>> def call_policy(shared_tensordict: _TensorDict, policy: nn.Module):
  ...     in_keys = policy.in_keys  # keys of the tensordict that are needed for the execution of the policy
  ...     out_keys = policy.out_keys  # keys of the tensordict that are written to by the policy
  ...     device = policy.device  # the policy may be on any device
  ...     policy_tensordict = shared_tensordict.select(*in_keys)
  ...     policy_tensordict = policy_tensordict.to(
  ...         device)  # no op if devices match. Often shared_tensordict will be on CPU, policy on cuda.
  ...     policy(policy_tensordict)  # writes the 'action' key
  ...     policy_tensordict = policy_tensordict.select(*out_keys)  # get rid of keys that are unchanged
  ...     shared_tensordict.update(policy_tensordict, inplace=True)  # update the shared_tensordict inplace, if possible


In this example, we have sent as few information from one device to another as
possible, by just casting the tensors that are read by the policy object
(this is possible because those keys are known in advance and 'hard-coded' in
the policy).
When sending the result back to the ``shared_tensordict``, we avoid sending also
key-value pairs that have not been changed in the process, which would result in
unnecessary I/O overhead.

Changing device or dtype
------------------------

We have just seen that a ``TensorDict`` can be sent to another device by calling
the ``.to()`` method. As do ``torch.Tensor`` objects, the ``TensorDict`` classes
support strings, integer and ``torch.device`` inputs.
``tensordict.to(...)`` also supports other ``_TensorDict`` subclasses (such as ``SavedTensorDict``).
As noted, casting a tensordict to a certain ``dtype`` is not permitted.
If this is wanted, one has to do it via regular for loop:

.. code:: python

  >>> for key, item in tensordict.items():
  ...   tensordict.set(key, item.to(dtype), inplace=False)


It is important not to specify ``inplace=True`` (default being ``False``), as
otherwise the copy will take place over the existing tensor, which won't change
its dtype.

Accessing tensor properties
---------------------------

In some instances, it might be useful to access tensor information without
directly accessing the tensor. This is typically the case for ``SavedTensorDicts``
or ``TensorDict`` instances populated with ``MemmapTensors``. To handle such
cases, we rely on a ``MetaTensor`` class, which is a no-content pseudo-tensor
class that only stores meta-data such as memory location, dtype or shape.
Each key in a ``TensorDict`` is associated with a ``torch.Tensor`` (or ``MemmapTensor``)
instance as well as a corresponding ``MetaTensor``.
When querying information about the contained tensors, ``TensorDict`` methods
will preferentially use the ``MetaTensor`` to collect this information.
Take the following examples, where we compare the speed of querying an information
through the ``get(key)`` method vs ``_get_meta(key)`` method:

.. code:: python

  >>> import timeit
  >>> from torchrl.data import TensorDict, SavedTensorDict
  >>> import torch
  >>> d = TensorDict(source={"a": torch.randn(10000,1000)}, batch_size=[])
  >>>
  >>> print(d.get("a")) # prints tensor
  >>> print(d._get_meta("a")) # MetaTensor(shape=torch.Size([10000, 1000]), device=cpu, dtype=torch.float32)
  >>>
  >>> print(timeit.timeit('d.get("a").shape', globals=globals())) # 0.320720541
  >>> print(timeit.timeit('d._get_meta("a").shape', globals=globals())) # 0.20476514300000037
  >>>
  >>> d = torch.stack([TensorDict(source={"a": torch.randn(10,10)}, batch_size=[]) for _ in range(100)], 0)
  >>>
  >>> print(timeit.timeit('d.get("a").shape', globals=globals(), number=1000)) # 0.17184535900003084
  >>> print(timeit.timeit('d._get_meta("a").shape', globals=globals(), number=1000)) # 0.04958068500002355
  >>>
  >>> d = TensorDict(source={"a": torch.randn(10, 10)}, batch_size=[]).to(SavedTensorDict)
  >>>
  >>> print(timeit.timeit('d.get("a").shape', globals=globals(), number=1000)) # 0.40540933799999834
  >>> print(timeit.timeit('d._get_meta("a").shape', globals=globals(), number=1000)) # 0.0002552720000039699


Relying on ``MetaTensor`` essentially allows us to conduct several
operations without loading the tensors in memory. One can iterate through the
content of the ``MetaTensor`` dictionary by calling ``tensordict._items_meta()``.
``MetaTensor`` name is inspired by the PyTorch official meta-tensors
(e.g. ``torch.zeros(1, device='meta')``), with the notalbe differences that it supports
more operations and also that it isn't a ``torch.Tensor`` instance.

