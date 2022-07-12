import copy
import os
from typing import Sequence
from typing import Optional

import hydra
import omegaconf
import torch
import tempfile
from torchvision.datasets.utils import download_url

from .transforms import Transform

from r3m import load_r3m

# VALID_ARGS = ["_target_", "device", "lr", "hidden_dim", "size", "l2weight", "l1weight", "langweight", "tcnweight", "l2dist", "bs"]
#
# def remove_language_head(state_dict):
#     keys = state_dict.keys()
#     ## Hardcodes to remove the language head
#     ## Assumes downstream use is as visual representation
#     for key in list(keys):
#         if ("lang_enc" in key) or ("lang_rew" in key):
#             del state_dict[key]
#     return state_dict
#
# def cleanup_config(cfg, device):
#     config = copy.deepcopy(cfg)
#     keys = config.agent.keys()
#     for key in list(keys):
#         if key not in VALID_ARGS:
#             del config.agent[key]
#     config.agent["_target_"] = "r3m.R3M"
#     config["device"] = device
#
#     ## Hardcodes to remove the language head
#     ## Assumes downstream use is as visual representation
#     config.agent["langweight"] = 0
#     return config.agent
#
# def load_r3m(modelid, device, store_dir=None):
#     store_dir_obj = None
#     if store_dir is None:
#         store_dir_obj = tempfile.TemporaryDirectory()
#         store_dir = str(store_dir_obj.name)
#     if modelid == "resnet50":
#         foldername = "r3m_50"
#         modelurl = 'https://drive.google.com/uc?id=1Xu0ssuG0N1zjZS54wmWzJ7-nb0-7XzbA'
#         configurl = 'https://drive.google.com/uc?id=10jY2VxrrhfOdNPmsFdES568hjjIoBJx8'
#     elif modelid == "resnet34":
#         foldername = "r3m_34"
#         modelurl = 'https://drive.google.com/uc?id=15bXD3QRhspIRacOKyWPw5y2HpoWUCEnE'
#         configurl = 'https://drive.google.com/uc?id=1RY0NS-Tl4G7M1Ik_lOym0b5VIBxX9dqW'
#     elif modelid == "resnet18":
#         foldername = "r3m_18"
#         modelurl = 'https://drive.google.com/uc?id=1A1ic-p4KtYlKXdXHcV2QV0cUzI4kn0u-'
#         configurl = 'https://drive.google.com/uc?id=1nitbHQ-GRorxc7vMUiEHjHWP5N11Jvc6'
#     else:
#         raise NameError('Invalid Model ID')
#
#     if not os.path.exists(os.path.join(store_dir, foldername)):
#         os.makedirs(os.path.join(store_dir, foldername))
#     foldername = os.path.join(store_dir, foldername)
#     modelpath = os.path.join(foldername, "model.pt")
#     configpath = os.path.join(foldername, "config.yaml")
#     print("downloading model")
#     download_url(modelurl, foldername, "model.pt")
#     print("downloading config")
#     download_url(configurl, foldername, "config.yaml")
#     print("done")
#
#     modelcfg = omegaconf.OmegaConf.load(configpath)
#     cleancfg = cleanup_config(modelcfg, device)
#     rep = hydra.utils.instantiate(cleancfg)
#     # rep = torch.nn.DataParallel(rep)
#     r3m_state_dict = remove_language_head(
#         torch.load(modelpath, map_location=torch.device(device))['r3m'])
#     rep.load_state_dict(r3m_state_dict)
#     del store_dir_obj
#     return rep


class R3MTransform(Transform):
    def __init__(
        self,
        keys_in: Sequence[str],
        keys_out: Optional[Sequence[str]] = None,
        modelid="resnet50",
    ):
        super().__init__(keys_in=keys_in, keys_out=keys_out)
        self.model = load_r3m(modelid, device="cpu")
        print(self.model)
