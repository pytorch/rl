import math
import os
import pickle
from contextlib import nullcontext
from pathlib import Path

import torch
from model import GPT, GPTConfig
from torch.utils.data import DataLoader

HERE = Path(__file__).parent


def setup(config):
    os.makedirs(config["out_dir"], exist_ok=True)

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    # for later use in torch.autocast
    device_type = "cuda" if "cuda" in config["device"] else "cpu"
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config["dtype"]]

    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )
    return ctx


def create_infinite_dataloader(data, config, collate_fn):
    """
    Creates a dataloader and yields batches from it indefinitely, so that we can request
    batches whenever we like with next.
    """
    dl = DataLoader(
        data,
        batch_size=config["batch_size"],
        shuffle=True,  # TODO: perhaps validation set shouldn't be shuffled?
        collate_fn=collate_fn,
        drop_last=True,
    )
    while True:
        yield from dl


def load_checkpoint(config):
    ckpt_path = os.path.join(config["out_dir"], "ckpt.pt")
    return torch.load(ckpt_path, map_location=config["device"])


def create_lr_scheduler(config):
    # learning rate decay scheduler (cosine with warmup)
    def scheduler(it):
        # 1) linear warmup for warmup_iters steps
        if it < config["warmup_iters"]:
            return config["learning_rate"] * it / config["warmup_iters"]
        # 2) if it > lr_decay_iters, return min learning rate
        if it > config["lr_decay_iters"]:
            return config["min_lr"]
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - config["warmup_iters"]) / (
            config["lr_decay_iters"] - config["warmup_iters"]
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return config["min_lr"] + coeff * (config["learning_rate"] - config["min_lr"])

    return scheduler


def init_model_scratch(config, model_kwargs):
    # attempt to derive vocab_size from the dataset
    meta_path = HERE / "nanoGPT" / "data" / config["dataset"] / "meta.pkl"
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print(
            "defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)"
        )
    model_kwargs["vocab_size"] = (
        meta_vocab_size if meta_vocab_size is not None else 50304
    )
    gptconf = GPTConfig(**model_kwargs)
    return GPT(gptconf)


def init_model_resume(config, model_kwargs):
    print(f"Resuming training from {config['out_dir']}")
    # resume training from a checkpoint.
    checkpoint = load_checkpoint(config)
    checkpoint_model_kwargs = checkpoint["model_kwargs"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_kwargs[k] = checkpoint_model_kwargs[k]
    # create the model
    gptconf = GPTConfig(**model_kwargs)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefixes = ["_orig_mod.", "model."]
    for unwanted_prefix in unwanted_prefixes:
        for k in list(state_dict):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    config["iter_num"] = checkpoint["iter_num"]
    config["best_val_loss"] = checkpoint["best_val_loss"]
    return model


def init_model_gpt2(config, model_kwargs):
    print(f"Initializing from OpenAI GPT-2 weights: {config['init_from']}")
    # initialize from OpenAI GPT-2 weights
    override_args = {"dropout": config["dropout"]}
    model = GPT.from_pretrained(config["init_from"], override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_kwargs[k] = getattr(model.config, k)

    return model


def init_model(config):
    model_kwargs = {
        "n_layer": config["n_layer"],
        "n_head": config["n_head"],
        "n_embd": config["n_embd"],
        "block_size": config["block_size"],
        "bias": config["bias"],
        "vocab_size": None,
        "dropout": config["dropout"],
    }

    if config["init_from"] == "scratch":
        model = init_model_scratch(config, model_kwargs)
    elif config["init_from"] == "resume":
        model = init_model_resume(config, model_kwargs)
    elif config["init_from"].startswith("gpt2"):
        model = init_model_gpt2(config, model_kwargs)

    # crop down the model block size if desired, using model surgery
    if config["block_size"] < model.config.block_size:
        model.crop_block_size(config["block_size"])
        # so that the checkpoint will have the right value
        model_kwargs["block_size"] = config["block_size"]

    return model, model_kwargs
