# nanoChatGPT

This is a toy implementation of a chatGPT-like model built on top of [nanoGPT][nanoGPT].

## Getting started

You will need to pull the code for nanoGPT which we have included here as a Git submodule. From inside your local copy of the TorchRL repo, run

```sh
git submodule init
git submodule update
```

From this directory, you can install extra requirements for running these examples with

```sh
pip install -r requirements.txt
```

## Training the models

First you must prepare the data you wish to train on. For example

```sh
python nanoGPT/data/shakespeare/prepare.py
```

Once the data has been prepared, you can train the GPT model.

```sh
python train.py
```

Default configuration can be found in `config/train.yaml`, and any option can be overridden with command-line arguments, for example to run the training script with a different batch size

```sh
python train.py --batch_size=128
```

[nanoGPT]: https://github.com/karpathy/nanoGPT
