# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import importlib.util

import time

import pytest
import torch
from mocking_classes_llm import DummyStrDataLoader
from torchrl import logger as torchrl_logger
from torchrl.collectors.llm import LLMCollector
from torchrl.collectors.llm.weight_update.vllm import vLLMUpdater
from torchrl.data import LazyStackStorage, ReplayBuffer
from torchrl.envs import AsyncEnvPool, StepCounter
from torchrl.envs.llm import LLMEnv
from torchrl.modules.llm import TransformersWrapper, vLLMWrapper

from torchrl.modules.llm.backends.vllm import make_vllm_worker

_has_transformers = importlib.util.find_spec("transformers") is not None
_has_vllm = importlib.util.find_spec("vllm") is not None


@pytest.mark.skipif(not _has_transformers, reason="missing transformers dependencies")
@pytest.mark.skipif(not _has_vllm, reason="missing vllm dependencies")
class TestLLMCollector:
    @pytest.fixture(scope="module")
    def vllm_instance(self):
        try:
            import vllm
        except ImportError:
            pytest.skip(reason="missing vllm")

        llm_model = vllm.LLM("gpt2")
        tokenizer = llm_model.get_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        return llm_model

    @pytest.fixture(scope="module")
    def vllm_instance_opt(self):
        try:
            import vllm
        except ImportError:
            pytest.skip(reason="missing vllm")

        llm_model = vllm.LLM("facebook/opt-125m")
        tokenizer = llm_model.get_tokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        return llm_model

    @pytest.fixture(scope="module")
    def transformers_instance(self):
        from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel(GPT2Config()).eval()
        # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        # model = OPTModel(OPTConfig("facebook/opt-125m"))
        # tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        # model = OPTForCausalLM(OPTConfig())

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        return model, tokenizer

    @pytest.mark.slow
    @pytest.mark.skip_if_nightly
    @pytest.mark.parametrize("rb,queue", [[True, False], [False, True], [False, False]])
    @pytest.mark.parametrize("total_steps", [1, 10, 20])
    def test_llm_collector_with_vllm(self, rb, queue, total_steps, vllm_instance):
        # NOTE: if VLLM fails with CUDA multiprocessing, try setting
        # `export VLLM_WORKER_MULTIPROC_METHOD=spawn`
        policy = vLLMWrapper(vllm_instance)
        tokenizer = vllm_instance.get_tokenizer()
        self._run_collector_test(total_steps, rb, queue, policy, tokenizer)

    @pytest.mark.slow
    @pytest.mark.parametrize("rb,queue", [[True, False], [False, True], [False, False]])
    @pytest.mark.parametrize("total_steps", [1, 10, 20])
    def test_llm_collector_with_transformers(
        self, rb, queue, total_steps, transformers_instance
    ):
        model, tokenizer = transformers_instance
        policy = TransformersWrapper(
            model,
            tokenizer=tokenizer,
            from_text=True,
            generate=True,
            return_log_probs=True,
        )
        self._run_collector_test(total_steps, rb, queue, policy, tokenizer)

    def _run_collector_test(self, total_steps, rb, queue, policy, tokenizer):
        bsz = 4
        dataloader = DummyStrDataLoader(bsz)

        env = LLMEnv.from_dataloader(
            dataloader=dataloader,
            from_text=True,
            batch_size=bsz,
            group_repeats=True,
            eos_token_id=tokenizer.eos_token_id,
        )
        queue = None
        if rb:
            rb = ReplayBuffer(storage=LazyStackStorage(max_size=total_steps * 2))
        elif queue:
            queue = queue.Queue(max_size=total_steps * 2)
        else:
            rb = None
        collector = LLMCollector(
            env=env,
            policy_factory=lambda: policy,
            dialog_turns_per_batch=env.batch_size[0],
            replay_buffer=rb,
            total_dialog_turns=total_steps,
            queue=queue,
        )

        stack = []
        for data in collector:
            # Should be moved to replay buffer
            if rb is not None:
                assert data is None
            else:
                stack.append(data)

        if rb is not None:
            # Now check the buffer
            assert len(rb) >= total_steps
            sample = rb.sample(4)
            assert sample.shape == (4,)
            assert not sample._has_exclusive_keys
            # Should match length
            assert len(sample["text"]) == 4
            # assert len(sample["text"][0]) == 10, sample["text"][0]
            # Should be non-empty
            assert sample["text_response"] is not None
            for i in range(4):
                # Check that there are more chars in the next step
                assert len(sample["text"][i]) < len(sample["next", "text"][i])
        else:
            stack = torch.cat(stack)
            assert not stack._has_exclusive_keys
            assert stack.numel() == max(-(total_steps // -4) * 4, 4)
            stack = stack.view(-1)
            for i in range(stack.numel()):
                # Check that there are more chars in the next step
                assert len(stack["text"][i]) < len(stack["next", "text"][i])
        assert collector._frames >= total_steps

    @pytest.mark.slow
    @pytest.mark.skip_if_nightly
    def test_llm_collector_start(self, vllm_instance):
        total_steps = 20
        policy = vLLMWrapper(vllm_instance)
        vllm_instance.get_tokenizer()
        bsz = 4
        dataloader = DummyStrDataLoader(bsz)

        env = LLMEnv.from_dataloader(
            dataloader=dataloader,
            from_text=True,
            batch_size=bsz,
            group_repeats=True,
        )

        rb = ReplayBuffer(storage=LazyStackStorage(max_size=total_steps * 2))
        collector = LLMCollector(
            env=env,
            policy_factory=lambda: policy,
            dialog_turns_per_batch=env.batch_size[0],
            replay_buffer=rb,
            total_dialog_turns=total_steps,
        )
        torchrl_logger.info("starting")
        try:
            collector.start()

            j = 0
            while True:
                if not len(rb):
                    time.sleep(1)  # Use asyncio.sleep instead of time.sleep
                sample = rb.sample(10)
                assert sample.ndim == 1
                for i in range(10):
                    # Check that there are more chars in the next step
                    assert len(sample["text"][i]) < len(sample["next", "text"][i])
                assert not sample._has_exclusive_keys, sample
                j += 1
                if rb.write_count >= total_steps:
                    break
            assert collector._frames >= total_steps
        finally:
            collector.async_shutdown(timeout=10)

    @pytest.mark.slow
    @pytest.mark.parametrize("rb", [False, True])
    @pytest.mark.parametrize("yield_only_last_steps", [False, True])
    def test_llm_collector_completed(
        self, vllm_instance_opt, rb, yield_only_last_steps
    ):
        torch.manual_seed(0)
        policy = vLLMWrapper(vllm_instance_opt)
        tokenizer = vllm_instance_opt.get_tokenizer()
        bsz = 4
        total_steps = 20
        max_steps = 20
        dataloader = DummyStrDataLoader(bsz)

        env = LLMEnv.from_dataloader(
            dataloader=dataloader,
            from_text=True,
            batch_size=bsz,
            group_repeats=True,
            eos_token_id=tokenizer.eos_token_id,
        )
        # To make sure the env breaks at some point
        env = env.append_transform(StepCounter(max_steps=max_steps))

        if rb:
            rb = ReplayBuffer(storage=LazyStackStorage(max_size=total_steps * 2))
        else:
            rb = None
        collector = LLMCollector(
            env=env,
            policy_factory=lambda: policy,
            dialog_turns_per_batch=env.batch_size[0],
            replay_buffer=rb,
            total_dialog_turns=total_steps,
            yield_completed_trajectories=True,
            yield_only_last_steps=yield_only_last_steps,
        )
        assert collector.yield_completed_trajectories
        assert collector.yield_only_last_steps is yield_only_last_steps

        cur_total_steps = 0
        has_found_one_with_more_steps = False
        for data in collector:
            if rb is None:
                assert data.ndim == 1
                # assert (data["next", "step_count"] < max_steps-1).all()
                cur_total_steps += data.numel()
                for i in range(data.numel()):
                    if data[i]["next", "step_count"] == max_steps:
                        continue
                    if data[i]["text_response"]:
                        # Check that there are more chars in the next step
                        assert len(data["text"][i]) < len(data["next", "text"][i]), (
                            i,
                            data[i]["next", "step_count"],
                            data[i]["next", "done"],
                            data[i]["text_response"],
                        )
                    else:
                        assert len(data["text"][i]) == len(data["next", "text"][i]), (
                            i,
                            data[i]["next", "step_count"],
                            data[i]["next", "done"],
                            data[i]["text_response"],
                        )

                if yield_only_last_steps:
                    assert data.shape == (1,)
                else:
                    has_found_one_with_more_steps |= data.numel() > 1
            else:
                assert data is None
                sample = rb.sample(5)
                for i in range(sample.numel()):
                    if sample[i]["next", "step_count"] == max_steps:
                        continue
                    if sample[i]["text_response"]:
                        # Check that there are more chars in the next step
                        assert len(sample["text"][i]) < len(
                            sample["next", "text"][i]
                        ), (
                            i,
                            sample[i]["next", "step_count"],
                            sample[i]["next", "done"],
                            sample[i]["text_response"],
                        )
                    else:
                        assert len(sample["text"][i]) == len(
                            sample["next", "text"][i]
                        ), (
                            i,
                            sample[i]["next", "step_count"],
                            sample[i]["next", "done"],
                            sample[i]["text_response"],
                        )

                assert sample.ndim == 1
                assert sample.shape == (5,)
                assert (sample["next", "step_count"] < 99).all()
                cur_total_steps += 1
            assert collector._frames >= cur_total_steps
        if rb is None and not yield_only_last_steps:
            assert has_found_one_with_more_steps
        assert collector._frames >= total_steps

    @pytest.mark.slow
    @pytest.mark.parametrize("rb", [False, True])
    @pytest.mark.parametrize("yield_only_last_steps", [False, True])
    def test_llm_collector_completed_async(
        self, vllm_instance_opt, rb, yield_only_last_steps
    ):
        torch.manual_seed(0)
        policy = vLLMWrapper(vllm_instance_opt)
        tokenizer = vllm_instance_opt.get_tokenizer()
        bsz = 4
        total_steps = 20
        max_steps = 20
        dataloader = DummyStrDataLoader(bsz)

        def env_maker():
            env = LLMEnv.from_dataloader(
                dataloader=dataloader,
                from_text=True,
                batch_size=(),
                group_repeats=True,
                eos_token_id=tokenizer.eos_token_id,
            )
            # To make sure the env breaks at some point
            env = env.append_transform(StepCounter(max_steps=max_steps))
            return env

        env = AsyncEnvPool([env_maker] * bsz, backend="threading", stack="lazy")

        if rb:
            rb = ReplayBuffer(storage=LazyStackStorage(max_size=total_steps * 2))
        else:
            rb = None
        collector = LLMCollector(
            env=env,
            policy_factory=lambda: policy,
            dialog_turns_per_batch=env.batch_size[0],
            replay_buffer=rb,
            total_dialog_turns=total_steps,
            yield_completed_trajectories=True,
            yield_only_last_steps=yield_only_last_steps,
        )
        assert collector.yield_completed_trajectories
        assert collector.yield_only_last_steps is yield_only_last_steps

        cur_total_steps = 0
        has_found_one_with_more_steps = False
        for data in collector:
            if rb is None:
                assert data.ndim == 1
                # assert (data["next", "step_count"] < max_steps-1).all()
                cur_total_steps += data.numel()
                for i in range(data.numel()):
                    if data[i]["next", "step_count"] == max_steps:
                        continue
                    if data[i]["text_response"]:
                        # Check that there are more chars in the next step
                        assert len(data["text"][i]) < len(data["next", "text"][i]), (
                            i,
                            data[i]["next", "step_count"],
                            data[i]["next", "done"],
                            data[i]["text_response"],
                        )
                    else:
                        assert len(data["text"][i]) == len(data["next", "text"][i]), (
                            i,
                            data[i]["next", "step_count"],
                            data[i]["next", "done"],
                            data[i]["text_response"],
                        )

                if yield_only_last_steps:
                    assert data.shape == (1,)
                else:
                    has_found_one_with_more_steps |= data.numel() > 1
            else:
                assert data is None
                sample = rb.sample(5)
                for i in range(sample.numel()):
                    if sample[i]["next", "step_count"] == max_steps:
                        continue
                    if sample[i]["text_response"]:
                        # Check that there are more chars in the next step
                        assert len(sample["text"][i]) < len(
                            sample["next", "text"][i]
                        ), (
                            i,
                            sample[i]["next", "step_count"],
                            sample[i]["next", "done"],
                            sample[i]["text_response"],
                        )
                    else:
                        assert len(sample["text"][i]) == len(
                            sample["next", "text"][i]
                        ), (
                            i,
                            sample[i]["next", "step_count"],
                            sample[i]["next", "done"],
                            sample[i]["text_response"],
                        )

                assert sample.ndim == 1
                assert sample.shape == (5,)
                assert (sample["next", "step_count"] < 99).all()
                cur_total_steps += 1
            assert collector._frames >= cur_total_steps
        if rb is None and not yield_only_last_steps:
            assert has_found_one_with_more_steps
        assert collector._frames >= total_steps


class TestUpdate:
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires 2 GPUs")
    def test_vllm_update(self):
        import ray

        ray.init()

        from torchrl.envs.llm import GSM8KEnv
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen2.5-3B"
        # TODO: Simple model errors
        # model_name = "facebook/opt-125m"

        # Create train model
        train_model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="cuda:0"
        )
        train_tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Wrap
        policy_training = TransformersWrapper(
            train_model,
            # train_model.eval(),
            tokenizer=train_tokenizer,
            # We have the tokens, let's just use them
            from_text=False,
            generate=False,
            return_log_probs=True,
        )

        # Create environment
        env = GSM8KEnv(repeats=4, tokenizer=train_tokenizer, num_envs=4)

        # Get metadata
        # TODO: Simplify this: can be done by the updater if the training policy is passed
        model_metadata = {
            k: (v.dtype, v.shape) for k, v in policy_training.model.state_dict().items()
        }

        # Get inference server
        inference_server = make_vllm_worker(
            model_name,
            gpu_memory_utilization=0.5,
            devices=[1],
            make_ray_worker=True,
        )

        # Wrap
        policy = vLLMWrapper(
            inference_server,
            from_text=True,
            return_log_probs=True,
            generate_kwargs={
                "max_tokens": 1024,
                "include_stop_str_in_output": True,
                "temperature": 0.8,
            },
        )

        # Make updater
        # TODO: Could use the transformer model directly?
        updater = vLLMUpdater(
            master_address=None,
            master_port=None,
            model_metadata=model_metadata,
        )
        collector = LLMCollector(
            env,
            policy=policy,
            dialog_turns_per_batch=4,
            total_dialog_turns=1_000_000,
            weight_updater=updater,
        )
        torchrl_logger.info("Created collector")
        torchrl_logger.info("init group")

        # TODO: Could we ask the collector to do this? Or maybe automate it within the registering of
        #  the collector
        updater.maybe_init_group()
        torchrl_logger.info("Update weights")

        # TODO: If the policy training is passed to the updater, we can cache a ref to the weights
        collector.update_policy_weights_(
            policy_training.model.state_dict(), worker_ids=[0]
        )

        torchrl_logger.info("Iterate")
        for _ in collector:
            break

        torchrl_logger.info("Second update")
        collector.update_policy_weights_(
            policy_training.model.state_dict(), worker_ids=[0]
        )

        torchrl_logger.info("Shutdown")
        collector.shutdown()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
