from typing import Any, Dict, Mapping, cast
import os
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import re
from compressai.registry import OPTIMIZERS, register_optimizer

def net_non_aux_optimizer(
    net: nn.Module, conf: Mapping[str, Any]
) -> Dict[str, optim.Optimizer]:
    """Returns optimizer for net loss."""
    parameters = {
        "net": {name
                 for name, param in net.named_parameters() 
                if param.requires_grad and not name.endswith(".quantiles")
                },
    }

    params_dict = dict(net.named_parameters())

    def make_optimizer(key):
        kwargs = dict(conf[key])
        del kwargs["type"]
        params = (params_dict[name] for name in sorted(parameters[key]))
        return OPTIMIZERS[conf[key]["type"]](params, **kwargs)

    optimizer = {key: make_optimizer(key) for key in ["net"]}

    return cast(Dict[str, optim.Optimizer], optimizer)

class TriplaneDataset(Dataset):
    def __init__(self, root_dir, split='train', target_lamda='0.002'):
        assert split in ['train', 'test']
        self.root_dir = root_dir
        self.split = split
        self.target_lamda = str(target_lamda)
        self.file_list = []
        self.meta_list = []
        self.shapes = []

        all_files = sorted([
            f for f in os.listdir(root_dir)
            if f.endswith('.pt') and re.match(r'^.+_.+_.+\.pt$', f)
        ])

        for fname in all_files:
            dataset, scene, lamda = fname[:-3].split('_')
            if (split == 'test' and lamda == self.target_lamda) or \
               (split == 'train' and lamda != self.target_lamda):
                self.file_list.append(fname)
                self.meta_list.append({
                    'filename': fname,
                    'dataset': dataset,
                    'scene': scene,
                    'lamda': lamda
                })
                if not self.shapes:
                    sample = torch.load(os.path.join(root_dir, fname))
                    assert sample.ndim == 4 and sample.shape[0] == 15
                    N, C, H, W = sample.shape
                    self.shapes = (N, C, H, W)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        path = os.path.join(self.root_dir, fname)
        tensor = torch.load(path)  # [15, C, H, W]
        return tensor

class RateLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, output, target):
        N, C, H, W = target
        out = {}
        num_params = N * C * H * W

        out['bit_loss'] = sum(
            (-torch.log2(likelihoods).sum() / num_params)
            for likelihoods in output["likelihoods"].values()
        )
        return out
