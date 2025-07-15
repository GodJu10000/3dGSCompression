# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import random
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
from model import Cheng2020AnchorCheckerboard, Elic2022Official
from utils import net_non_aux_optimizer, TriplaneDataset, RateLoss

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

def get_triplane_dataloader(root_dir, split='train', target_lamda='0.003',
                            batch_size=15, shuffle=False, num_workers=4):
    dataset = TriplaneDataset(root_dir, split=split, target_lamda=target_lamda)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=shuffle, num_workers=num_workers)
    # dataloader 输出 shape: [1, 15, C, H, W]
    return dataloader, dataset.meta_list, dataset.shapes


def configure_optimizers(net, args):
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate}
    }
    optimizer = net_non_aux_optimizer(net, conf)
    return optimizer["net"]


def train_one_epoch(
    model, criterion, train_dataloader, metas, shapes, optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device
    bit_losses = []
    bit2MB_scale = 8 * 1024 * 1024
    for i, d in enumerate(train_dataloader):
        d = d.to(device).squeeze(0) # [N, C, H, W]

        optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d.shape)
        out_criterion["bit_loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        print(
            f"Train epoch {epoch}: ["
            f"{i*1}/{len(train_dataloader.dataset)}"
            f" ({100. * i / len(train_dataloader):.0f}%)]"
            f'\tLoss: {out_criterion["bit_loss"].item():.3f} \n'
        )
        with torch.no_grad():
            bit_losses.append[out_criterion["bit_loss"].item()]

    print("Train dataset size:\n")
    for bit_loss, meta, shape in zip(bit_losses, metas, shapes):
        dataset = meta['dataset']
        scene = meta['scene']
        lamda = meta['lamda']
        N, C, H, W = shape 
        num_params = N*C*H*W
        size = bit_loss*num_params/bit2MB_scale
        print(
            f"{dataset}_{scene}_{lamda} triplanes size: {size}MB |"
        )

def test_epoch(epoch, test_dataloader, metas, shapes, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()

    bit_losses = []
    bit2MB_scale = 8 * 1024 * 1024
    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device).squeeze(0) # [N, C, H, W]
            out_net = model(d)
            out_criterion = criterion(out_net, d.shape)

            loss.update(out_criterion["bit_loss"])

        bit_losses.append[out_criterion["bit_loss"].item()]

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} \n"

    )
    
    print("Test dataset size:\n")
    for bit_loss, meta, shape in zip(bit_losses, metas, shapes):
        dataset = meta['dataset']
        scene = meta['scene']
        lamda = meta['lamda']
        N, C, H, W = shape 
        num_params = N*C*H*W
        size = bit_loss*num_params/bit2MB_scale
        print(
            f"{dataset}_{scene}_{lamda} triplanes size(MB): {size} |"
        )

    return loss.avg


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "checkpoint_best_loss.pth.tar")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")

    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num_workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=15, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=15,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument('--channel_dim', type=int, default=16, help='Number of channels (default: 16)')
    parser.add_argument('--data_source', type=str, default='../triplanes', help='Path to data source (default: ../triplanes)')
    parser.add_argument('--target_lamda', type=str, default='0.002', help='Lamda value used to split test set (default: 0.002)')
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)


    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_loader, train_meta, train_shape = get_triplane_dataloader(
    root_dir=args.data_source, split='train', target_lamda=args.target_lamda, shuffle=True, num_workers=args.num_workers)

    test_loader, test_meta, test_shape = get_triplane_dataloader(
    root_dir=args.data_source, split='test', target_lamda=args.target_lamda, shuffle=False, num_workers=args.num_workers)

    net = Cheng2020AnchorCheckerboard(N=args.channel_dim)
    net = net.to(device)
    
    # parallelism
    # if args.cuda and torch.cuda.device_count() > 1:
    #     net = CustomDataParallel(net)

    optimizer= configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateLoss()

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_loader,
            train_meta,
            train_shape,
            optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = test_epoch(
            epoch, 
            test_loader, 
            test_meta,
            test_shape,
            net, 
            criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

    updated = net.update(update_quantiles=True)
    if updated:
        print("aux parameters have been updated successfully")
    else:
        print("aux parameters havn't been updated")

    if args.save:
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "loss": loss,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            },
            is_best,
        )



if __name__ == "__main__":
    main(sys.argv[1:])