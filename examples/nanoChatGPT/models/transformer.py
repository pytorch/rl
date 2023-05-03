import pickle
from pathlib import Path

import torch
from tensordict.nn import TensorDictModule

from .nanoGPT.model import GPT, GPTConfig
from .utils import _remove_state_dict_prefixes, load_checkpoint

HERE = Path(__file__).parent

DEFAULT_VOCAB_SIZE = 50_304


def init_transformer_scratch(config, model_kwargs):
    # attempt to derive vocab_size from the dataset
    meta_path = HERE / "nanoGPT" / "data" / config["dataset"] / "meta.pkl"
    meta_vocab_size = None
    if meta_path.exists():
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
        meta_vocab_size if meta_vocab_size is not None else DEFAULT_VOCAB_SIZE
    )
    gptconf = GPTConfig(**model_kwargs)
    return GPT(gptconf)


def init_transformer_resume(config, model_kwargs):
    print(f"Resuming training from {config['out_dir']}")
    # resume training from a checkpoint.
    checkpoint = load_checkpoint(config["out_dir"], device=config["device"])
    checkpoint_model_kwargs = checkpoint["model_kwargs"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_kwargs[k] = checkpoint_model_kwargs[k]
    # create the model
    gptconf = GPTConfig(**model_kwargs)
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    _remove_state_dict_prefixes(state_dict)
    model.load_state_dict(state_dict)
    config["iter_num"] = checkpoint["iter_num"]
    config["best_val_loss"] = checkpoint["best_val_loss"]
    return model


def init_transformer_gpt2(config, model_kwargs):
    print(f"Initializing from OpenAI GPT-2 weights: {config['init_from']}")
    # initialize from OpenAI GPT-2 weights
    override_args = {"dropout": config["dropout"]}
    model = GPT.from_pretrained(config["init_from"], override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
        model_kwargs[k] = getattr(model.config, k)

    return model


def init_transformer(config, as_tensordictmodule=True, skip_compilation=False):
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
        model = init_transformer_scratch(config, model_kwargs)
    elif config["init_from"] == "resume":
        model = init_transformer_resume(config, model_kwargs)
    elif config["init_from"].startswith("gpt2"):
        model = init_transformer_gpt2(config, model_kwargs)

    # crop down the model block size if desired, using model surgery
    if config["block_size"] < model.config.block_size:
        model.crop_block_size(config["block_size"])
        # so that the checkpoint will have the right value
        model_kwargs["block_size"] = config["block_size"]

    model.to(config["device"])
    # compile the model
    if not skip_compilation and config["compile"]:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # requires PyTorch 2.0

    if as_tensordictmodule:
        model = TensorDictModule(
            model, in_keys=["prompt", "target"], out_keys=["logits", "loss"]
        )
    return model, model_kwargs


def init_optimizer(model, config):
    # optimizer
    optimizer = model.configure_optimizers(
        config["weight_decay"],
        config["learning_rate"],
        (config["beta1"], config["beta2"]),
        "cuda" if "cuda" in config["device"] else "cpu",
    )
    if config["init_from"] == "resume":
        checkpoint = load_checkpoint(config["out_dir"], device=config["device"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    return optimizer
