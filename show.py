#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/28 

import os
from argparse import ArgumentParser

import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt

from data import get_dataloader, DATASETS
from util import ucp_expand_batch, ucp_norm


def print_stats(ucp:np.ndarray):
  print('   min:',  ucp.min())
  print('   max:',  ucp.max())
  print('   rng:',  ucp.max() - ucp.min())
  print('   avg:',  ucp.mean())
  print('   var:',  ucp.var())
  print('   L0:',   (np.abs(ucp) > 0).sum())
  print('   L1:',   np.abs(ucp).mean())
  print('   L2:',   np.linalg.norm(ucp))
  print('   Linf:', np.abs(ucp).max())


def show(args):
  # quick skip
  save_fp = os.path.join(args.img_path, os.path.splitext(os.path.basename(args.ucp))[0] + '.png')
  if args.silent and os.path.exists(save_fp) and not args.overwrite:
    return

  # load ucp
  ucp = np.load(args.ucp)       # [C, H, W]
  if not args.silent:
    print('ucp.shape:', ucp.shape)
    print_stats(ucp)

  # show & save ucp
  if True:
    plt.figure(figsize=(3, 3))

    # as RGB image
    plt.subplot(2, 2, 1)
    plt.axis('off') ; plt.xticks([]) ; plt.yticks([])
    plt.imshow(ucp_norm(ucp, norm=True).transpose([1, 2, 0]))  # [H, W, C]

    # split each channel
    for i, (ch, cmap) in enumerate(zip(ucp, ['Reds', 'Greens', 'Blues']), 2):
      if not args.silent:
        print(f'[{cmap}]')
        print_stats(ch)

      plt.subplot(2, 2, i)
      plt.axis('off') ; plt.xticks([]) ; plt.yticks([])
      plt.imshow(ucp_norm(ch), cmap=cmap)

    plt.margins(0, 0)
    plt.tight_layout()
    plt.subplots_adjust(top=0.975, bottom=0.025, left=0.025, right=0.975, hspace=0.05, wspace=0.05)

    # save ucp
    if not os.path.exists(save_fp) or args.overwrite:
      plt.savefig(save_fp, bbox_inches='tight', pad_inches=0.1)

    # show ucp
    if not args.silent:
      plt.show()

  # show clean image `x` and `x+ucp`
  if args.dataset and not args.silent:
    n_sample = 8
    dataloader = get_dataloader(args.dataset, n_sample, args.data_path, 'test', shuffle=True, n_worker=0)

    ucp = torch.from_numpy(ucp)
    X, _ = iter(dataloader).next()
    DX = ucp_expand_batch(ucp, X.shape, resizer=args.resizer)
    try:
      for X, _ in dataloader:
        AX = (X + DX).clip(0.0, 1.0)
        
        X_AX = torch.cat([X, AX])                               # [16, 3, 32, 32]
        grid = make_grid(X_AX, nrow=n_sample, normalize=False)  # [3, 274, 70]

        plt.imshow(grid.numpy().transpose([1, 2, 0]))
        plt.show()
    except KeyboardInterrupt:
      pass


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--ucp', required=True, help='path to ucp.npy file')
  parser.add_argument('--resizer', default='tile', choices=['tile', 'interpolate'], help='resize UCP when shape mismatch')
  parser.add_argument('-D', '--dataset', default='imagenet-1k', choices=DATASETS, help='show when ucp is added onto a dataset')

  parser.add_argument('--overwrite', action='store_true', help='force overwrite if exists when plt.save_fig(ucp)')
  parser.add_argument('--silent', action='store_true', help='only plt.save_fig(ucp), no plt.show()')

  parser.add_argument('--data_path', default='data', help='folder path to downloaded dataset')
  parser.add_argument('--img_path', default='img', help='folder path to demo images')
  args = parser.parse_args()

  os.makedirs(args.img_path, exist_ok=True)

  show(args)
