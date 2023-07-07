from torch import multiprocessing as mp
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement


def run_collector(data_collector, qout):
    assert isinstance(data_collector, TensorDictReplayBuffer)
    qout.put("succeeded")
    print('succeeded')


if __name__ == "__main__":

    sampler = SamplerWithoutReplacement()
    buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(200),
        sampler=sampler,
        batch_size=200,
    )

    qout = mp.Queue(1)
    p = mp.Process(target=run_collector, args=(buffer, qout))
    p.start()
    assert qout.get() == "succeeded"
    p.join()

