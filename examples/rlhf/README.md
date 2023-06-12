# RLHF example

This example uses RLHF (Reinforcement Learning with Human Feedback) to train a language model to summarize Reddit posts.

## Getting started

Make sure you have PyTorch 2.0 installed. You can find installation instructions [here](https://pytorch.org/get-started/locally/).

From this directory, you can install extra requirements for running these examples with

```sh
pip install -r requirements.txt
```

## Training the models
### Training the transformer

Once the data has been prepared, you can train the GPT model.

```sh
python train.py
```

Default configuration can be found in `config/train.yaml`, and any option can be overridden with command-line arguments, for example to run the training script with a different batch size

```sh
python train.py --batch_size=128
```
> **_NOTE:_**  Apple Silicon Macbooks users make sure to use `--device=mps` and prepend all commands with `PYTORCH_ENABLE_MPS_FALLBACK=1` to enable CPU fallback

### Training the reward model

Next you can train the reward model with

```sh
python train_reward.py
```

### Training the final model with RLHF

To train the final model run

```sh
python train_rlhf.py
```
