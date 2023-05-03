# nanoChatGPT

This is a toy implementation of a chatGPT-like model built on top of [nanoGPT][nanoGPT].

## Getting started

You will need to pull the code for nanoGPT which we have included here as a Git submodule. From inside your local copy of the TorchRL repo, run

```sh
git submodule init
git submodule update
```

Make sure you have PyTorch 2.0 installed. You can find installation instructions [here](https://pytorch.org/get-started/locally/).

You will also need to install the latest versions of tensordict and TorchRL.

```sh
pip install torchrl-nightly
pip uninstall -y tensordict
pip install tensordict-nightly
```

**Note**: `torchrl-nightly` has `tensordict` as a dependency, not `tensordict-nightly`, so after installing `torchrl-nightly` we uninstall `tensordict` before installing `tensordict-nightly` to avoid a clash.

From this directory, you can install extra requirements for running these examples with

```sh
pip install -r requirements.txt
```

## Training the models

### Preparing the data

First you must prepare the data you wish to train on. For example

```sh
python models/nanoGPT/data/shakespeare/prepare.py
```

### Training the transformer

Once the data has been prepared, you can train the GPT model.

```sh
python train.py
```

Default configuration can be found in `config/train.yaml`, and any option can be overridden with command-line arguments, for example to run the training script with a different batch size

```sh
python train.py --batch_size=128
```

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

[nanoGPT]: https://github.com/karpathy/nanoGPT
