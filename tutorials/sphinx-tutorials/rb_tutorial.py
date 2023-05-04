# -*- coding: utf-8 -*-
"""
Using Replay Buffers
====================

**Author**: `Vincent Moens <https://github.com/vmoens>`_

"""
######################################################################
# Replay buffers are a central piece of any RL or control algorithm. 
# Supervised learning methods are usually characterized by a training loop
# where data is randomly pulled from a static dataset and fed successively
# to the model and loss function.
# In RL, things are often slightly different: the data is gathered using the
# model, then temporarily stored in a dynamic structure (the experience
# replay buffer), which serves as dataset for the loss module.
#
# As always, the context in which the buffer is used drastically conditions
# how it is built: some may wish to store trajectories when others will want
# to store single transitions. Specific sampling strategies may be preferable
# in contexts: some items can have a higher priority than others, or it can
# be important to sample with or without replacement.
# Computational factors may also come into play, such as the size of the buffer
# which may exceed the available RAM storage.
#
# For these reasons, TorchRL's replay buffers are fully composable: although
# they come with "batteries included", requiring a minimal effort to be built,
# they also support many customizations such as storage type,
# sampling strategy or data tranforms.
#
# 
# In this tutorial, you will learn:
# - How to build a Replay Buffer (RB) and use it with any datatype;
# - How to use RBs with TensorDict;
# - How to sample from or iterate over a replay buffer, and how to define the
#   sampling strategy;
# - How to use prioritized replay buffers;
# - How to transform data coming in and out from the buffer;
# - How to store trajectories in the buffer.
#
#
# Basics: building a vanilla replay buffer
# ----------------------------------------
# 
# TorchRL's replay buffers are designed to prioritize modularity,
# composability, efficiency, and simplicity. For instance, creating a basic
# replay buffer is a straightforward process, as shown in the following
# example:
#

from torchrl.data import ReplayBuffer

buffer = ReplayBuffer()


######################################################################
# By default, this replay buffer will have a size of 1000. Let's check this
# by populating our buffer using the :meth:`torchrl.data.ReplayBuffer.extend`
# method:
#

print("length before adding elements:", len(buffer))

buffer.extend(range(2000))

print("length after adding elements:", len(buffer))

######################################################################
# We have used the :meth:`torchrl.data.ReplayBuffer.extend` method which is
# designed to add multiple items at a time. If the object that is passed
# to ``extend`` has more than one dimension, its first dimension is
# considered to be the one to be split in separate elements in the buffer.
# This essentially means that when adding multidimensional tensors or
# tensordicts to the buffer, the buffer will only look at the first dimension
# when counting the elements it holds in memory.
#
# To add items one at a time, the :meth:`torchrl.data.ReplayBuffer.add` method
# should be used instead.
#
# Customizing the storage
# ~~~~~~~~~~~~~~~~~~~~~~~
# 
# We see that the buffer has been capped to the first 1000 elements that we
# passed to it.
# To change the size, we need to customize our storage.
#
# TorchRL proposes three types of storages:
#
# - The `ListStorage` stores elements independently in a list. It supports
#   any data type, but this flexibility comes at the cost of efficiency;
# - The `LazyTensorStorage` stores tensors or TensorDict (or tensorclass) objects. The storage is contiguous, meaning that sampling will be efficient, but the implicit restriction is that any data passed to it must have the same basic properties as the first batch of data that was used to instantiate the buffer. Passing data that does not match this requirement will either raise an exception or lead to some undefined behaviours.
# - The `LazyMemmapStorage` works as the `LazyTensorStorage` in that it is lazy (ie. it expects the first batch of data to be instantiated), and it requires data that match in shape and dtype for each batch stored. The big difference of this storage type is that it is stored on disk, meaning that it can support very large datasets while still storing data in a contiguous manner.
# 
# Let us see how we can use each of these storages:

# In[4]:


from torchrl.data import ListStorage, LazyMemmapStorage, LazyTensorStorage
from tensordict import TensorDict, tensorclass
import torch

size = 10_000


# In[5]:


# A buffer with a list storage buffer can store any kind of data (but we must change the collate_fn):
buffer_list = ReplayBuffer(storage=ListStorage(size), collate_fn=lambda x: x)
buffer_list.extend(["a", 0, "b"])
buffer_list.sample(3)


# In[6]:


# A LazyTensorStorage can store data contiguously. This is a great option when dealing with complicated data structures of medium size
buffer_lazytensor = ReplayBuffer(storage=LazyTensorStorage(size))

# Let us create a batch of data of size 3 with 2 tensors stored in it
data = TensorDict({
    "a": torch.arange(12).view(3, 4),
    ("b", "c"): torch.arange(15).view(3, 5),
}, batch_size=[3])
print(data)

# The first call to `extend` will instantiate the storage. The first dimension of the data is chunked and cosidered as independent:
buffer_lazytensor.extend(data)
print(f"The buffer has {len(buffer_lazytensor)} elements")


# In[7]:


# Let us sample from the buffer, and print the data:
sample = buffer_lazytensor.sample(5)
print("samples", sample['a'], sample['b', 'c'])


# In[8]:


# A LazyMemmapStorage is created in the same manner.

buffer_lazymemmap = ReplayBuffer(storage=LazyMemmapStorage(size))
buffer_lazymemmap.extend(data)
print(f"The buffer has {len(buffer_lazymemmap)} elements")
sample = buffer_lazytensor.sample(5)
print("samples: a=", sample['a'], "\n('b', 'c'):", sample['b', 'c'])


# In[9]:


# We can also customize the storage location on disk:
buffer_lazymemmap = ReplayBuffer(storage=LazyMemmapStorage(size, scratch_dir="/tmp/memmap/"))
buffer_lazymemmap.extend(data)
print(f"The buffer has {len(buffer_lazymemmap)} elements")
print(f"the 'a' tensor is stored in", buffer_lazymemmap._storage._storage['a'].filename)
print(f"the ('b', 'c') tensor is stored in", buffer_lazymemmap._storage._storage['b', 'c'].filename)


# The tensor location follows the same structure as the TensorDict that contains them: this makes it easy to save and load buffers during training.
# 
# When using `TensorDict` as a data carrier, we can take advantage of the `TensorDictReplayBuffer` class. The main advantage is that it takes care of packing the sampled data with extra information (such as sample indices) if needed. It is constructed in the exact same way as a regular ReplayBuffer and should be interchangeable with it:

# In[10]:


from torchrl.data import TensorDictReplayBuffer

buffer_lazymemmap = TensorDictReplayBuffer(storage=LazyMemmapStorage(size, scratch_dir="/tmp/memmap/"), batch_size=12)
buffer_lazymemmap.extend(data)
print(f"The buffer has {len(buffer_lazymemmap)} elements")
sample = buffer_lazymemmap.sample()
print("sample:", sample)


# In[11]:


sample["index"]


# Our sample now has an extra "index" key that indicates what indices were sampled.
# 
# The ReplayBuffer class and associated subclasses also work natively with `tensorclass` classes, which can convinently be used to encode datasets in a more explicit manner:

# In[12]:


from tensordict import tensorclass

@tensorclass
class MyData:
    images: torch.Tensor
    labels: torch.Tensor

data = MyData(
    images=torch.randint(255, (1000, 64, 64, 3), ), 
    labels=torch.randint(100, (1000,)), 
    batch_size=[1000],
)

buffer_lazymemmap = TensorDictReplayBuffer(storage=LazyMemmapStorage(size, scratch_dir="/tmp/memmap/"), batch_size=12)
buffer_lazymemmap.extend(data)
print(f"The buffer has {len(buffer_lazymemmap)} elements")
sample = buffer_lazymemmap.sample()
print("sample:", sample)


# As expected. the data has the proper class and shape!
# 
# ## Sampling and iterating over buffers
# 
# RBs support multiple sampling strategies:
# - If the batch-size is fixed and can be defined at construction time, it can be passed as keyword argument to the buffer;
# - With a fixed batch-size, the replay buffer can be iterated over to gather samples;
# - If the batch-size is dynamic, it can be passed to the `sample` method on-the-fly.
# 
# Sampling can be done using multithreading, but this is incompatible with the last option (at it requires the buffer to know in advance the size of the next batch).
# 
# Let us see a few examples:
# 
# ### Fixed batch-size

# In[13]:


buffer_lazymemmap = ReplayBuffer(storage=LazyMemmapStorage(size), batch_size=128)
buffer_lazymemmap.extend(data)
buffer_lazymemmap.sample()


# To enable multithreaded sampling, just pass a positive integer to the `prefetch` keyword argument during construction:

# In[14]:


buffer_lazymemmap = ReplayBuffer(storage=LazyMemmapStorage(size), batch_size=128, prefetch=10)  # creates a queue of 10 elements to be prefetched in the background
buffer_lazymemmap.extend(data)
buffer_lazymemmap.sample()


# This batch of data has the size that we wanted it to have (128)
# 
# ### Fixed batch-size, iterating over the buffer
# 
# We can iterate over the buffer like we would do with a regular dataloader:

# In[15]:


for i, data in enumerate(buffer_lazymemmap):
    if i == 3:
        print(data)
        break


# Since our sampling strategy is completely random and does not prevent repetition, this iterator is never ending. We can, however, use the `SamplerWithoutReplacement` which will turn our buffer in a finite iterator:

# In[16]:


from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement

buffer_lazymemmap = ReplayBuffer(storage=LazyMemmapStorage(size), batch_size=32, sampler=SamplerWithoutReplacement())
# we create a data that is big enough to get a couple of samples
data = TensorDict({
    "a": torch.arange(512).view(128, 4),
    ("b", "c"): torch.arange(1024).view(128, 8),
}, batch_size=[128])

buffer_lazymemmap.extend(data)
for i, data in enumerate(buffer_lazymemmap):
    continue
print(f"A total of {i+1} batches have been collected")


# ### Dynamic batch-size
# 
# As we have seen earlier, the `batch_size` keyword argument can be ommitted and passed directly to the `sample` method:

# In[17]:


buffer_lazymemmap = ReplayBuffer(storage=LazyMemmapStorage(size), sampler=SamplerWithoutReplacement())
buffer_lazymemmap.extend(data)
print("sampling 3 elements:", buffer_lazymemmap.sample(3))
print("sampling 5 elements:", buffer_lazymemmap.sample(5))


# ## Prioritized Replay buffers
# 
# We provide an interface for prioritized replay buffers. This buffer class samples data according to a priority signal that is passed through the data.
# 
# Although this tool is compatible with non-tensordict data, we encourage using TensorDict instead as it makes it possible to carry meta-data in and out of the buffer with little effort.
# 
# Let us first see how to build a prioritized replay buffer in the generic case:

# In[18]:


from torchrl.data.replay_buffers.samplers import PrioritizedSampler

size = 1000

rb = ReplayBuffer(storage=ListStorage(size), sampler=PrioritizedSampler(max_capacity=size, alpha=0.8, beta=1.1), collate_fn=lambda x: x)

indices = rb.extend(["a", 1, None])

# the sampler expects to have a priority for each element. This is done via the `update_priority` method.
# We assign an artifically high priority to the second sample in the dataset
rb.update_priority(index=indices, priority=torch.tensor([0, 1_000, 0.1]))

# sampling should return mostly the second sample (1)
sample, info = rb.sample(10, return_info=True)
print(sample)


# In[19]:


# The info contains the relative weights of the items as well as the indices.
print(info)


# We see that using a prioritized replay buffer requires a series of extra steps in the training loop compared with a regular buffer:
# - After collecting data and extending the buffer, the priority of the items must be updated;
# - After computing the loss and getting a "priority signal" from it, we must update again the priority of the items in the buffer. This requires us to keep track of the indices.
# 
# This drastically hampers the reusability of the buffer: if one is to write a training script where both a prioritized and a regular buffer can be created, she must add a considerable amount of control flow to make sure that the appropriate methods are called at the appropriate place, if and only if a prioritized buffer is being used.
# 
# Let us see how we can solve this with TensorDict. We saw that the `TensorDictReplayBuffer` returns data augmented with storage indices. This class also ensures that the priority signal is automatically parsed to the prioritized sampler if present. 
# 
# The combination of these features simplifies things in several ways:
# - When extending the buffer, the priority signal will automatically be parsed if present and the priority will accurately be assigned;
# - The indices will be stored in the samples, making it easy to update the priority after the loss computation.
# 
# The following code illustrates these concepts:

# In[20]:


rb = TensorDictReplayBuffer(storage=ListStorage(size), sampler=PrioritizedSampler(size, alpha=0.8, beta=1.1), priority_key="td_error", batch_size=1024)

data["td_error"] = torch.arange(data.numel())

rb.extend(data)

sample = rb.sample()
# higher indices should occur more frequently:
from matplotlib import pyplot as plt
plt.hist(sample["index"].numpy())


# In[21]:


# once we have worked with our sample, we update the priority key using the `update_tensordict_priority` method.
# For the sake of showing how this works, let us assign a very low probability to the sampled items
sample = rb.sample()
sample["td_error"] = (data.numel()-sample["index"]).exp()
rb.update_tensordict_priority(sample)

# higher indices should occur less frequently:
sample = rb.sample()
from matplotlib import pyplot as plt
plt.hist(sample["index"].numpy())


# ## Using transforms
# 
# The data stored in a replay buffer may not be ready to be presented to a loss module. 
# In some cases, the data produced by a collector can be too heavy to be saved as-is. Examples of this include converting images from uint8 to floating point tensors, or concatenating successive frames when using decision transformers. 
# 
# Data can be processed in and out of a buffer just by appending the appropriate transform to it.
# Here are a few examples:
# 
# ### Saving raw images
# 
# `uint8`-typed tensors are comparatively much less memory expensive than the floating point tensors we usually feed to our models. For this reason, it can be useful to save the raw images.

# In[22]:


from torchrl.collectors import SyncDataCollector, RandomPolicy
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import TransformedEnv, ToTensorImage, Resize, GrayScale, Compose

env = TransformedEnv(
    GymEnv("CartPole-v1", from_pixels=True), 
    Compose(
        ToTensorImage(in_keys=["pixels"], out_keys=["pixels_trsf"]),
        Resize(in_keys=["pixels_trsf"], w=64, h=64),
        GrayScale(in_keys=["pixels_trsf"]),
    )
)

# let us have a look at a rollout:

print(env.rollout(3))   


# We have just created an environment that produces pixels. These images are processed to be fed to a policy.
# We would like to store the raw images, and not their transforms.
# To do this, we will append a transform to the collector to select the keys we want to see appearing:

# In[23]:


from torchrl.envs import ExcludeTransform
collector = SyncDataCollector(
    env,
    RandomPolicy(env.action_spec),
    frames_per_batch=10,
    total_frames=1000,
    postproc=ExcludeTransform("pixels_trsf", ("next", "pixels_trsf"), "collector")
)


# Let us have a look at a batch of data, and control that the "pixels_trsf" have been discarded:

# In[24]:


for data in collector:
    print(data)
    break


# We create a replay buffer with the same transform as the environment.
# There is, however, a detail that needs to be addressed: transforms used without environments are oblivious to the data structure. Our data comes with a nested "next" tensordict that will be ignored by our transform. We manually add these keys to the transform:

# In[25]:


t = Compose(
        ToTensorImage(in_keys=["pixels", ("next", "pixels")], out_keys=["pixels_trsf", ("next", "pixels_trsf")]),
        Resize(in_keys=["pixels_trsf", ("next", "pixels_trsf")], w=64, h=64),
        GrayScale(in_keys=["pixels_trsf", ("next", "pixels_trsf")]),
    )
rb = TensorDictReplayBuffer(storage=LazyMemmapStorage(1000), transform=t, batch_size=16)
rb.extend(data)


# Let us check that a sample sees the transformed images reappear:

# In[26]:


print(rb.sample())


# ### A more complex examples: using CatFrames

# In[27]:


from torchrl.envs import UnsqueezeTransform, CatFrames

env = TransformedEnv(
    GymEnv("CartPole-v1", from_pixels=True), 
    Compose(
        ToTensorImage(in_keys=["pixels"], out_keys=["pixels_trsf"]),
        Resize(in_keys=["pixels_trsf"], w=64, h=64),
        GrayScale(in_keys=["pixels_trsf"]),
        UnsqueezeTransform(-4, in_keys=["pixels_trsf"]),
        CatFrames(dim=-4, N=4, in_keys=["pixels_trsf"]),
    )
)
collector = SyncDataCollector(
    env,
    RandomPolicy(env.action_spec),
    frames_per_batch=10,
    total_frames=1000,
    # postproc=ExcludeTransform("pixels_trsf", ("next", "pixels_trsf"), "collector")
)
for data in collector:
    print(data)
    break


# In[28]:


t = Compose(
        ToTensorImage(in_keys=["pixels", ("next", "pixels")], out_keys=["pixels_trsf", ("next", "pixels_trsf")]),
        Resize(in_keys=["pixels_trsf", ("next", "pixels_trsf")], w=64, h=64),
        GrayScale(in_keys=["pixels_trsf", ("next", "pixels_trsf")]),
        CatFrames(dim=-4, N=4, in_keys=["pixels_trsf", ("next", "pixels_trsf")]),
    )
rb = TensorDictReplayBuffer(storage=LazyMemmapStorage(1000), transform=t, batch_size=16)
data_exclude = data.exclude("pixels_trsf", ("next", "pixels_trsf"))
rb.add(data_exclude)


# In[30]:


s = rb.sample(1) # the buffer has only one element
s


# After a bit of processing (excluding non-used keys etc), we see that the data generated online and offline match!

# In[40]:


(data.exclude("collector")==s.squeeze(0).exclude("index", "collector")).all()

