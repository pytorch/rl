# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import os
import pickle

import pytest
import torch
from tensordict import TensorDict

from torchrl.data import (
    LazyMemmapStorage,
    LazyTensorStorage,
    ReplayBuffer,
    SliceSampler,
    VideoClipRef,
)
from torchrl.data.video import (
    _has_torchcodec,
    clear_video_decoder_cache,
    set_video_decoder_cache_size,
)
from torchrl.envs.transforms import DecodeVideoTransform

pytestmark = pytest.mark.skipif(
    not _has_torchcodec, reason="torchcodec is required for video frame decoding"
)

# Intensity assigned to frame ``i`` so frames stay identifiable after lossy encoding.
_STEP = 11


def _intensity(i: int) -> int:
    return min(i * _STEP, 255)


def _write_video(path: str, num_frames: int = 20, height: int = 8, width: int = 12):
    from torchcodec.encoders import VideoEncoder

    frames = torch.zeros(num_frames, 3, height, width, dtype=torch.uint8)
    for i in range(num_frames):
        frames[i] = _intensity(i)
    VideoEncoder(frames=frames, frame_rate=10).to_file(path)
    return num_frames


@pytest.fixture
def video_path(tmp_path):
    clear_video_decoder_cache()
    path = os.path.join(str(tmp_path), "clip.mp4")
    _write_video(path)
    yield path
    clear_video_decoder_cache()


def _aligned(frames: torch.Tensor, indices, tol: int = 14) -> bool:
    means = (
        frames.float().mean(dim=tuple(range(1, frames.ndim))).round().long().tolist()
    )
    return all(abs(m - _intensity(i)) <= tol for m, i in zip(means, indices))


class TestVideoClipRef:
    def test_from_file_metadata(self, video_path):
        ref = VideoClipRef.from_file(video_path)
        assert isinstance(ref, VideoClipRef)
        assert ref.batch_size == torch.Size([20])
        assert ref.frame_index.dtype == torch.long
        assert ref.frame_index.tolist() == list(range(20))
        assert ref.source == video_path

    def test_from_file_explicit_indices_no_metadata_read(self, video_path):
        ref = VideoClipRef.from_file(video_path, frame_index=[3, 1, 9])
        assert ref.batch_size == torch.Size([3])
        assert ref.frame_index.tolist() == [3, 1, 9]

    def test_bare_constructor_reads_all_frames(self, video_path):
        # VideoClipRef(path) auto-fills frame_index from the file metadata.
        ref = VideoClipRef(video_path)
        assert isinstance(ref, VideoClipRef)
        assert ref.batch_size == torch.Size([20])
        assert ref.frame_index.tolist() == list(range(20))
        assert _aligned(ref[4:8].decode(), [4, 5, 6, 7])

    def test_bare_constructor_explicit_frame_index(self, video_path):
        # An explicit frame_index sets the batch size and skips the metadata read.
        ref = VideoClipRef(video_path, frame_index=torch.tensor([3, 1, 9]))
        assert ref.batch_size == torch.Size([3])
        assert ref.frame_index.tolist() == [3, 1, 9]
        assert _aligned(ref.decode(), [3, 1, 9])

    def test_slicing_is_lazy(self, video_path):
        ref = VideoClipRef.from_file(video_path)
        clip = ref[4:8]
        # indexing returns a (still lazy) reference, not a tensor
        assert isinstance(clip, VideoClipRef)
        assert clip.batch_size == torch.Size([4])
        assert clip.frame_index.tolist() == [4, 5, 6, 7]
        assert clip.source == video_path

    def test_decode_shape_dtype_alignment(self, video_path):
        ref = VideoClipRef.from_file(video_path)
        decoded = ref[4:8].decode()
        assert decoded.shape == torch.Size([4, 3, 8, 12])
        assert decoded.dtype == torch.uint8
        assert _aligned(decoded, [4, 5, 6, 7])

    def test_frames_property(self, video_path):
        ref = VideoClipRef.from_file(video_path)
        torch.testing.assert_close(ref[2:5].frames, ref[2:5].decode())

    def test_scalar_ref_decodes_single_frame(self, video_path):
        ref = VideoClipRef.from_file(video_path)
        single = ref[7].decode()
        assert single.shape == torch.Size([3, 8, 12])
        assert _aligned(single.unsqueeze(0), [7])

    def test_stack_preserves_per_element_order(self, video_path):
        ref = VideoClipRef.from_file(video_path)
        order = [9, 2, 5, 13]
        stacked = torch.stack([ref[i] for i in order])
        assert isinstance(stacked, VideoClipRef)
        assert stacked.frame_index.tolist() == order
        decoded = stacked.decode()
        assert decoded.shape == torch.Size([4, 3, 8, 12])
        # non-contiguous indices must still map back to the requested order
        assert _aligned(decoded, order)

    def test_decode_dtype_override(self, video_path):
        ref = VideoClipRef.from_file(video_path)
        decoded = ref[0:3].decode(dtype=torch.float32)
        assert decoded.dtype == torch.float32

    def test_out_dtype_default_on_object(self, video_path):
        ref = VideoClipRef.from_file(video_path, dtype=torch.float32)
        assert ref[0:3].decode().dtype == torch.float32

    def test_auto_decode_eager_indexing(self, video_path):
        ref = VideoClipRef.from_file(video_path, auto_decode=True)
        out = ref[1:4]
        assert torch.is_tensor(out)
        assert out.shape == torch.Size([3, 3, 8, 12])
        assert _aligned(out, [1, 2, 3])

    def test_auto_decode_default_is_lazy(self, video_path):
        ref = VideoClipRef.from_file(video_path)
        assert isinstance(ref[1:4], VideoClipRef)

    def test_from_timestamps(self, video_path):
        ref = VideoClipRef.from_timestamps(video_path, [0.0, 0.5, 1.0])
        # 10 fps -> seconds * fps
        assert ref.frame_index.tolist() == [0, 5, 10]

    def test_pickle_roundtrip_no_decoder_state(self, video_path):
        ref = VideoClipRef.from_file(video_path)
        # warm the (module-level, per-process) decoder cache
        ref[0:2].decode()
        reloaded = pickle.loads(pickle.dumps(ref))
        assert isinstance(reloaded, VideoClipRef)
        # cleared cache must be transparently rebuilt after unpickling
        clear_video_decoder_cache()
        decoded = reloaded[5:9].decode()
        assert decoded.shape == torch.Size([4, 3, 8, 12])
        assert _aligned(decoded, [5, 6, 7, 8])

    def test_decoder_cache_size(self, video_path):
        clear_video_decoder_cache()
        set_video_decoder_cache_size(1)
        try:
            VideoClipRef.from_file(video_path)[0:2].decode()
            VideoClipRef.from_file(video_path)[2:4].decode()
        finally:
            set_video_decoder_cache_size(8)


class TestDecodeVideoTransform:
    def test_requires_in_keys(self):
        with pytest.raises(TypeError):
            DecodeVideoTransform(in_keys=None)

    def test_out_keys_default_to_in_keys(self):
        t = DecodeVideoTransform(in_keys=["frame"])
        assert t.out_keys == t.in_keys

    def test_forward_decodes_leaf(self, video_path):
        ref = VideoClipRef.from_file(video_path)
        td = TensorDict({"frame": ref[2:6]}, batch_size=[4])
        out = DecodeVideoTransform(in_keys=["frame"], out_keys=["pixels"])(td)
        assert out["pixels"].shape == torch.Size([4, 3, 8, 12])
        assert _aligned(out["pixels"], [2, 3, 4, 5])

    def test_forward_rejects_non_ref(self, video_path):
        td = TensorDict({"frame": torch.zeros(4, 3, 8, 12)}, batch_size=[4])
        with pytest.raises(TypeError):
            DecodeVideoTransform(in_keys=["frame"], out_keys=["pixels"])(td)

    @pytest.mark.parametrize("storage_cls", [LazyTensorStorage, LazyMemmapStorage])
    def test_replay_buffer_slice_sampler(self, video_path, storage_cls):
        ref = VideoClipRef.from_file(video_path)
        data = TensorDict(
            {"frame": ref, "episode": torch.zeros(20, dtype=torch.long)},
            batch_size=[20],
        )
        rb = ReplayBuffer(
            storage=storage_cls(20),
            sampler=SliceSampler(slice_len=4, traj_key="episode"),
            batch_size=8,
            transform=DecodeVideoTransform(in_keys=["frame"], out_keys=["pixels"]),
        )
        rb.extend(data)
        sample = rb.sample()
        assert sample["pixels"].shape == torch.Size([8, 3, 8, 12])
        assert sample["pixels"].dtype == torch.uint8
        # the lazy reference still travels alongside the decoded frames
        assert isinstance(sample["frame"], VideoClipRef)
        assert _aligned(sample["pixels"], sample["frame"].frame_index.tolist())

    def test_replay_buffer_nested_key(self, video_path):
        ref = VideoClipRef.from_file(video_path)
        data = TensorDict(
            {
                "obs": TensorDict({"cam": ref}, batch_size=[20]),
                "episode": torch.zeros(20, dtype=torch.long),
            },
            batch_size=[20],
        )
        rb = ReplayBuffer(
            storage=LazyTensorStorage(20),
            sampler=SliceSampler(slice_len=5, traj_key="episode"),
            batch_size=10,
            transform=DecodeVideoTransform(
                in_keys=[("obs", "cam")], out_keys=[("obs", "pixels")]
            ),
        )
        rb.extend(data)
        sample = rb.sample()
        assert sample["obs", "pixels"].shape == torch.Size([10, 3, 8, 12])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
class TestVideoClipRefCuda:
    def test_decode_on_cuda(self, video_path):
        ref = VideoClipRef.from_file(video_path)
        decoded = ref[3:7].decode(device="cuda")
        assert decoded.device.type == "cuda"
        assert decoded.shape == torch.Size([4, 3, 8, 12])

    def test_out_device_on_object(self, video_path):
        ref = VideoClipRef.from_file(video_path, device="cuda")
        assert ref[0:2].decode().device.type == "cuda"

    def test_transform_device(self, video_path):
        ref = VideoClipRef.from_file(video_path)
        td = TensorDict({"frame": ref[0:4]}, batch_size=[4])
        out = DecodeVideoTransform(
            in_keys=["frame"], out_keys=["pixels"], device="cuda"
        )(td)
        assert out["pixels"].device.type == "cuda"


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
