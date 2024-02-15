# RLHF example

This example uses RLHF (Reinforcement Learning with Human Feedback) to train a 
language model to summarize Reddit posts.

## Note:
This example is not included in the benchmarked results of the current release (v0.3). The intention is to include it in the
benchmarking of future releases, to ensure that it can be successfully run with the release code and that the
results are consistent. For now, be aware that this additional check has not been performed in the case of this
specific example.

## Getting started

Make sure you have PyTorch>=2.0 installed. You can find installation instructions
[here](https://pytorch.org/get-started/locally/).

From this directory, you can install extra requirements for running these
examples with

```sh
pip install -r requirements.txt
```

## Training the models
### Training the transformer

Once the data has been prepared, you can train the GPT model.

```sh
python train.py
```

Default configuration can be found in `config/train.yaml`, and any option can
be overridden with command-line arguments, for example to run the training
script with a different batch size:

```sh
python train.py --batch_size=128
```
> **_NOTE:_**  Apple Silicon Macbooks users make sure to use `--device=mps`
> and prepend all commands with `PYTORCH_ENABLE_MPS_FALLBACK=1` to enable CPU fallback

### Training the reward model

Once you have completed supervised fine-tuning, copy the desired model
checkpoint to `./out` or update the config to point `model.name_or_path` at
the relevant checkpoint in the timestamped working directory created by Hydra.
You can then train the reward model with:

```sh
python train_reward.py
```

### Training the final model with RLHF

Once again, make sure you have either updated the configuration to point
`reward_model.name_or_path` at the relevant timestamped working directory, or
copy the checkpoint to `./out_reward`.
You can then train the final model by running

```sh
python train_rlhf.py
```
