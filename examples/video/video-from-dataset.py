"""Video from dataset example.

This example shows how to save a video from a dataset.

To run it, you will need to install the openx requirements as well as torchvision.
"""

from torchrl.data.datasets import OpenXExperienceReplay
from torchrl.record import CSVLogger, VideoRecorder

# Create a logger that saves videos as mp4
logger = CSVLogger("./dump", video_format="mp4")


# We use the VideoRecorder transform to save register the images coming from the batch.
t = VideoRecorder(
    logger=logger, tag="pixels", in_keys=[("next", "observation", "image")]
)
# Each batch of data will have 10 consecutive videos of 200 frames each (maximum, since strict_length=False)
dataset = OpenXExperienceReplay(
    "cmu_stretch",
    batch_size=2000,
    slice_len=200,
    download=True,
    strict_length=False,
    transform=t,
)

# Get a batch of data and visualize it
for _ in dataset:
    # The transform has seen the data since it's in the replay buffer
    t.dump()
    break

# Alternatively, we can build the dataset without the VideoRecorder and call it manually:
dataset = OpenXExperienceReplay(
    "cmu_stretch",
    batch_size=2000,
    slice_len=200,
    download=True,
    strict_length=False,
)

# Get a batch of data and visualize it
for data in dataset:
    t(data)
    t.dump()
    break
