# Versioning Issues

## Pytorch version
This issue is related to https://github.com/pytorch/rl/issues/689. Using PyTorch versions <1.13 and installing stable package leads to undefined symbol errors. For example:
```
ImportError: /usr/local/lib/python3.7/dist-packages/torchrl/_torchrl.so: undefined symbol: _ZN8pybind116detail11type_casterIN2at6TensorEvE4loadENS_6handleEb
```

### How to reproduce
1. Create an Colab Notebook (at 24/11/2022 Colab enviroment has Python 3.7 and Pytorch 1.12 installed by default).
2. ``` !pip install torchrl ```
3. ``` import torchrl ```

In Colab you can solve the issue by running:
``` 
!pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu -U 
```
before the ```!pip install torchrl``` command. This will install the latest pytorch. Instructions can be found [here](https://pytorch.org/get-started/locally/).

### Workarounds
There are two workarounds to this issue
1. Install/upgrade to the latest pytorch release before installing torchrl.
2. If you need to use a previous pytorch relase: Install functorch version related to your torch distribution: e.g. ``` pip install functorch==0.2.0 ``` and install library from source ``` pip install git+https://github.com/pytorch/rl@<lib_version_here> ```.
