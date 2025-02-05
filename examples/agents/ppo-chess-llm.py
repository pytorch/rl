import queue
import random
import re
import subprocess
import threading
import time

import chess

import torch
from tensordict.nn import (
    ProbabilisticTensorDictModule as Prob,
    ProbabilisticTensorDictSequential as ProbSeq,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)

from torch.distributions import Categorical

from torchrl.collectors import SyncDataCollector
from torchrl.data import NonTensor
from torchrl.data.replay_buffers.samplers import SliceSamplerWithoutReplacement
from torchrl.data.tensor_specs import Composite
from torchrl.envs import ChessEnv
from torchrl.envs.transforms import (
    ConditionalPolicySwitch,
    StepCounter,
    Tokenizer,
    Transform,
)
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from tqdm import tqdm

from transformers import pipeline


def random_policy(td):
    legal_moves = env.get_legal_moves()
    move = random.choice(legal_moves)
    san_idx = env.san_moves.index(move)
    td["action"] = torch.tensor(san_idx)
    return td


class SanHistory(Transform):
    def __init__(self):
        super().__init__()

    def _step(self, tensordict, next_tensordict):
        import numpy as np

        try:
            history = np.append(tensordict["san"], next_tensordict["san"])
        except:
            history = tensordict["san"] + [next_tensordict["san"]]
        next_tensordict["san"] = history
        return next_tensordict

    def _reset(self, tensordict, tensordict_reset):
        return tensordict_reset


class LLMInputTransform(Transform):
    def __init__(self, san_moves):
        super().__init__()
        self.san_moves = san_moves

    def _step(self, tensordict, next_tensordict):
        legal_move_indices = next_tensordict["legal_moves"]
        legal_moves = [self.san_moves[i] for i in legal_move_indices if i != 29275]
        existing_moves = (
            ", ".join(next_tensordict["san"])
            if not isinstance(next_tensordict["san"], str)
            else next_tensordict["san"]
        )
        if len(legal_moves) > 0:
            llm_input = (
                f"You are playing a game of chess. The list of moves so far are [{existing_moves}] and the legal moves are "
                f"[{', '.join(legal_moves)}]. Please choose one of the legal moves. "
                f"Respond only with the following sentence, with no additional explanatory text. Example Answer: I choose {legal_moves[0]}!"
            )
        else:
            llm_input = "The game has ended as there are no legal moves left. "
        next_tensordict["observation"] = llm_input
        print("step", llm_input)
        return next_tensordict

    def _reset(self, tensordict, tensordict_reset):
        legal_move_indices = tensordict_reset["legal_moves"]
        legal_moves = [self.san_moves[i] for i in legal_move_indices if i != 29275]
        existing_moves = (
            ", ".join(tensordict_reset["san"])
            if not isinstance(tensordict_reset["san"], str)
            else tensordict_reset["san"]
        )
        if len(legal_moves) > 0:
            llm_input = (
                f"You are playing a game of chess. The list of moves so far are [{existing_moves}] and the legal moves are "
                f"[{', '.join(legal_moves)}]. Please choose one of the legal moves. "
                f"Respond only with the following sentence, with no additional explanatory text. Example Answer: I choose {legal_moves[0]}!"
            )
        else:
            llm_input = ""
        tensordict_reset["observation"] = llm_input
        print("reset", llm_input)
        return tensordict_reset

    def transform_observation_spec(self, observation_spec: Composite):
        if not isinstance(observation_spec, Composite):
            raise ValueError(
                f"observation_spec was expected to be of type Composite. Got {type(observation_spec)} instead."
            )
        observation_spec["observation"] = NonTensor(shape=(), example_data="any")
        return observation_spec


def run_player(input_queue, output_queue):
    # Start the subprocess (./packed.sh)
    process = subprocess.Popen(
        ["python", "../sunfish/sunfish.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        while True:
            try:
                user_input = input_queue.get(timeout=5)
                # # print(f"Got user_input={user_input}")

                if user_input == "exit":
                    # print("Exiting process...")
                    break

                fen = user_input
                in_str = (
                    f"position fen {fen}\n go wtime 20 btime 20 winc 1000 binc 1000\n"
                )

                # Send the user input to the running process
                process.stdin.write(in_str)
                process.stdin.flush()

                output = process.stdout.readline()
                if output:
                    # print(f"Output: {output.strip()}")
                    move = re.search(r"bestmove (.*)", output.strip()).group(1)
                    output_queue.put(move)

            except queue.Empty:
                continue

    except Exception as e:
        print(f"Error interacting with process: {e}")
    finally:
        # Close the process gracefully
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()
        process.wait()  # Make sure the process finishes cleanly


def setup_env(input_queue, output_queue, tokenizer):
    def policy_sunfish(td):
        input_queue.put(td["fen"])
        move = output_queue.get()
        san = env.board.san(chess.Move.from_uci(move))
        san_idx = env.san_moves.index(san)
        td["action"] = torch.tensor(san_idx)
        return td

    env = ChessEnv(
        include_san=True, include_fen=True, stateful=True, include_legal_moves=True
    )
    # sunfish plays white
    sunfish_condition = lambda td: ((td.get("step_count") % 2) == 0).all()
    env = env.append_transform(StepCounter())
    env = env.append_transform(SanHistory())
    env = env.append_transform(
        ConditionalPolicySwitch(condition=sunfish_condition, policy=policy_sunfish)
    )
    env = env.append_transform(LLMInputTransform(env.san_moves))
    env = env.append_transform(
        Tokenizer(
            in_keys=["observation"],
            out_keys=["obs_tokens"],
            # in_keys_inv=["action"],
            # out_keys_inv=["action_tokens"],
            tokenizer=tokenizer,
        )
    )
    env.reset()
    return env


def setup_llm_policy():

    from transformers import pipeline

    device = "cuda:1"
    torch.set_default_device(device)

    # model_id = "meta-llama/Llama-3.1-8B-Instruct"
    model_id = "Qwen/Qwen2.5-7B-Instruct"

    pipeline = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device=device,
    )

    tokenizer = pipeline.tokenizer
    llm = pipeline.model.eval().requires_grad_(False)

    class LLMWrapper(torch.nn.Module):
        def __init__(self, llm, tokenizer, mode: str = "data"):
            super().__init__()
            self.llm = llm
            self.tokenizer = tokenizer
            assert mode in ["data", "policy"]
            self.mode = mode

        def forward(self, tokenized_observation):
            if len(tokenized_observation.shape) == 1:
                tokenized_observation = tokenized_observation.unsqueeze(0)
            max_new_tokens = 7
            exclamation_mark = self.tokenizer.encode("!", return_tensors="pt")

            if self.mode == "policy":
                input_length = tokenized_observation.shape[1]
                for i in range(max_new_tokens):
                    output = self.llm(
                        input_ids=tokenized_observation, output_hidden_states=True
                    )
                    logits = output.logits[
                        :, -1, :
                    ]  # [B, len(tokenized_observation) + 1, vocab_size] -> [B, vocab_size]
                    probs = torch.nn.functional.softmax(
                        logits, dim=-1
                    )  # [B, vocab_size] -> [B, vocab_size]
                    next_token = torch.argmax(probs, dim=-1).unsqueeze(
                        -1
                    )  # [B, vocab_size] -> [B] -> [B, 1]
                    tokenized_observation = torch.cat(
                        (tokenized_observation, next_token), dim=1
                    )
                log_prob = output.logits[:, input_length - 1 :, :]
                hidden = output.hidden_states[-1][:, input_length - 1, :]
                return log_prob, hidden
            else:
                while True:
                    generated_tokens = tokenized_observation
                    input_length = generated_tokens.shape[1]
                    for _ in range(max_new_tokens):
                        output = self.llm(
                            input_ids=generated_tokens, output_hidden_states=True
                        )
                        logits = output.logits[:, -1, :]
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        # FIXME: should this not be max to promote exploration?
                        next_token = torch.argmax(probs, dim=-1).unsqueeze(-1)
                        generated_tokens = torch.cat(
                            (generated_tokens, next_token), dim=1
                        )
                        if torch.equal(next_token, exclamation_mark):
                            break

                    # truncate the prompt from the output
                    output_tokens = generated_tokens.squeeze()[input_length:]
                    decoded_output = tokenizer.decode(
                        output_tokens.squeeze(), skip_special_tokens=True
                    ).strip()

                    try:
                        # print(decoded_output)
                        chosen_move = re.search(
                            r"I choose (.*)!", decoded_output
                        ).group(1)
                        # verify legal
                        output_idx = env.get_legal_moves().index(chosen_move)
                        break
                    except:
                        # Continue generating tokens if no valid move is found
                        continue

                # FIXME: Perhaps san_idx can be a transform
                san_idx = env.san_moves.index(chosen_move)
                # FIXME: might need to do softmax on logits
                log_prob = torch.gather(
                    output.logits[:, input_length:, :],
                    -1,
                    output_tokens.unsqueeze(0).unsqueeze(0),
                )

                # pad output tokens if it was less than max_new_tokens
                if output_tokens.shape[0] < max_new_tokens:
                    output_tokens = torch.nn.functional.pad(
                        output_tokens,
                        (0, max_new_tokens - output_tokens.shape[0]),
                        value=tokenizer.pad_token_id,
                    )

                hidden = output.hidden_states[-1][:, input_length - 1, :]

                return san_idx, output_tokens, log_prob, hidden

    def remove_logits(td):
        td.pop("logits")
        return td

    data_llm_policy = Seq(
        Mod(
            LLMWrapper(llm, tokenizer, mode="data"),
            in_keys=["obs_tokens"],
            out_keys=["action", "tokenized_action", "logits", "hidden"],
        ),
        lambda td: td.set("sample_log_prob", td.get("logits").sum(td.ndim - 1)),
        remove_logits,
    )

    prob_module = Prob(
        in_keys=["logits"],
        out_keys=["tokenized_action"],
        distribution_class=Categorical,
        return_log_prob=True,
    )

    class AggregateProb(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            log_prob = x.sum(dim=-1)
            return log_prob

    actor_llm_policy = ProbSeq(
        Mod(
            LLMWrapper(llm, tokenizer, mode="policy"),
            in_keys=["obs_tokens"],
            out_keys=["logits", "hidden"],
        ),
        prob_module,
        # if use lambda: 'function' object has no attribute 'in_keys'
        Mod(AggregateProb(), in_keys=["sample_log_prob"], out_keys=["sample_log_prob"]),
        return_composite=True,
    )

    class CriticHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.m = torch.nn.Linear(3584, 1, dtype=torch.bfloat16)

        def forward(self, hidden):
            return self.m(hidden).squeeze(-1)

    critic_llm_policy = Seq(
        Mod(CriticHead(), in_keys=["hidden"], out_keys=["state_value"]),
    )

    return actor_llm_policy, data_llm_policy, critic_llm_policy, tokenizer


def play(env, data_llm_policy, actor_llm_policy, tokenizer):
    from torchrl.data import LazyStackStorage, ReplayBuffer

    rb = ReplayBuffer(
        storage=LazyStackStorage(100),
        batch_size=32,
        sampler=SliceSamplerWithoutReplacement(slice_len=8, end_key=("next", "done")),
    )
    # layout=torch.jagged errors with Qwen
    # File "/home/mg1998/.conda/envs/rl/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py", line 859, in forward
    # cache_position = torch.arange(
    #             past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
    #         )
    # AttributeError: 'ConstantIntNode' object has no attribute 'add'
    # After attempting to fix this there were other issues like NJT can't unsqueeze dim=0
    # rb.append_transform(lambda td: td.densify(layout=torch.jagged))
    # rb.append_transform(
    #     lambda td: td.set(
    #         "obs_tokens", td.get("obs_tokens").to_padded_tensor(tokenizer.pad_token_id)
    #     )
    # )

    # rb.append_transform(lambda td: td.set("tokenized_action", td.get("tokenized_action").to_padded_tensor(tokenizer.pad_token_id)))

    collector = SyncDataCollector(
        env, data_llm_policy, frames_per_batch=100, total_frames=10000
    )
    loss_module = ClipPPOLoss(
        actor_network=actor_llm_policy,
        critic_network=critic_llm_policy,
    )

    # FIXME: is this the right way to do this?
    loss_module.tensor_keys.action = "tokenized_action"

    optim = torch.optim.Adam(loss_module.parameters())

    gae = GAE(
        value_network=Seq(actor_llm_policy[0], *critic_llm_policy),
        gamma=0.99,
        lmbda=0.95,
        shifted=True,
    )

    import torch.nn.functional as F

    def pad_tensors_to_same_shape(tensor1, tensor2):
        size1 = tensor1.shape
        size2 = tensor2.shape

        if size1[-1] < size2[-1]:
            pad1 = size2[-1] - size1[-1]
            tensor1_padded = F.pad(tensor1, (0, pad1))
            tensor2_padded = tensor2
        elif size1[-1] > size2[-1]:
            pad2 = size1[-1] - size2[-1]
            tensor2_padded = F.pad(tensor2, (0, pad2))
            tensor1_padded = tensor1
        else:
            tensor1_padded = tensor1
            tensor2_padded = tensor2

        return tensor1_padded, tensor2_padded

    for data in tqdm(collector):
        breakpoint()
        print(
            "win rate",
            data["next", "reward"].sum() / data["next", "done"].sum().clamp_min(1e-6),
        )
        rb.empty()
        # FIXME: what is the right way to do this?
        data = data.densify(layout=torch.jagged)
        data.set(
            "obs_tokens",
            data.get("obs_tokens").to_padded_tensor(tokenizer.pad_token_id),
        )
        data.get("next").set(
            "obs_tokens",
            data.get("next").get("obs_tokens").to_padded_tensor(tokenizer.pad_token_id),
        )

        obs_tokens = data.get("obs_tokens")
        next_obs_tokens = data.get("next").get("obs_tokens")
        obs_tokens, next_obs_tokens = pad_tensors_to_same_shape(
            obs_tokens, next_obs_tokens
        )
        data.set("obs_tokens", obs_tokens)
        data.get("next").set("obs_tokens", next_obs_tokens)

        data = gae(data)
        rb.extend(data)

        for data in tqdm(rb):
            loss = loss_module(data)
            loss.sum(reduce=True).backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 100.0)
            optim.step()
            optim.zero_grad()


if __name__ == "__main__":
    input_queue = queue.Queue()
    output_queue = queue.Queue()

    player_thread = threading.Thread(
        target=run_player, args=(input_queue, output_queue)
    )
    player_thread.start()
    actor_llm_policy, data_llm_policy, critic_llm_policy, tokenizer = setup_llm_policy()
    env = setup_env(input_queue, output_queue, tokenizer)

    play(env, data_llm_policy, actor_llm_policy, tokenizer)

    input_queue.put("exit")
    player_thread.join()
