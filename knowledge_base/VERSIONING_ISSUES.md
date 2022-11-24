# Versioning Issues

## Pytorch version
This issue is related to https://github.com/pytorch/rl/issues/689. Using PyTorch versions lower < 1.13 and installing stable package leads to the following error:
```
ImportError: /usr/local/lib/python3.7/dist-packages/torchrl/_torchrl.so: undefined symbol: _ZN8pybind116detail11type_casterIN2at6TensorEvE4loadENS_6handleEb
```
This is probably due to some incompatibility (tensor casting?) of torch C++ bindings between version 1.12 and 1.13. 


### How to reproduce
1. Create an Colab Notebook (at 24/11/2022 Colab enviroment has Python 3.7 and Pytorch 1.12 installed by default).
2. ``` !pip install torchrl ```
3. ``` import torchrl ```

### Workarounds
1. Install torch 1.13 or above.
2. Install latest version of functorch. This version depends on torch 1.13 and will install it for you. 

      ``` pip install functorch ```
3. If you need to keep you current torch version, install from source. 

      ``` pip install git+https://github.com/pytorch/rl@<put_version_here> ```


