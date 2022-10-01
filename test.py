#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

import os
from argparse import ArgumentParser

import numpy as np

from model import *
from data import *
from util import *


def test(args):
  ''' Ckpt '''
  if args.load:
    print('[Ckpt] use local trained weights')
    args.name = args.load
    args.model, args.train_dataset = args.load.split('_')
    args.ckpt_fp = os.path.join(args.log_path, args.name, 'model-best.pth')
    if not os.path.exists(args.ckpt_fp):
      raise ValueError(f'You have not trained the local model {args.ckpt_fp} yet')
  else:
    print('[Ckpt] use pretrained weights from torchvision/torchhub')
    args.train_dataset = 'imagenet'   # NOTE: currently all models in MODELS are pretrained on `imagenet`
    args.name = f'{args.model}_{args.train_dataset}'
    args.ckpt_fp = None

  ''' Model '''
  model = get_model(args.model, ckpt_fp=args.ckpt_fp).to(device)
  model.eval()

  ''' Data '''
  dataloader = get_dataloader(args.atk_dataset, args.batch_size, args.data_path, split=args.split)

  ''' Test '''
  # Try testing clean accuracy
  if not args.ucp:
    if chk_dataset_compatible(args.train_dataset, args.atk_dataset):
      print(f'[Accuracy]')
      acc = test_acc(model, dataloader)
      print(f'   clean: {acc:.3%}')
      for mag in [1e-3, 1e-2, 1e-1]:
        ret = test_acc(model, dataloader, noise=mag)
        print(f'   clean-noise({float_to_str(mag)}): {ret:.3%}')
    else:
      print('atk_dataset is not compatible is with train_dataset')
  
  # Try attacked accuracy
  if args.ucp:
    ucp = torch.from_numpy(np.load(args.ucp)).to(device)

    # Try testing remnet accuracy (:= 1 - misclf rate) after adding UCP
    if chk_dataset_compatible(args.train_dataset, args.atk_dataset):
      acc = test_acc(model, dataloader, ucp, resizer=args.resizer)
      print(f'   ucp: {acc:.3%}')

      if args.ex:
        test_ucp_ex(ucp, model, dataloader, test_fn=test_acc, resizer=args.resizer)
      
    # Try testing predction changing rate after adding UCP
    if True:
      print(f'[Pred Change Rate]')
      pcr = test_pcr(model, dataloader, ucp, resizer=args.resizer)
      print(f'   ucp: {pcr:.3%}')

      if args.ex:
        test_ucp_ex(ucp, model, dataloader, test_fn=test_pcr, resizer=args.resizer)


def test_acc(model, dataloader, ucp=None, resizer='tile', noise=None) -> float:
  ''' Accuracy '''

  if ucp is not None:
    X, _ = iter(dataloader).next()
    DX = ucp_expand_batch(ucp, X.shape, resizer=resizer)

  total, correct = 0, 0
  with torch.no_grad():
    for X, Y in dataloader:
      X, Y = X.to(device), Y.to(device)
      if noise is not None:
        n = (torch.rand_like(X) * 2 - 1) * noise
        X = (X + n).clip(0.0, 1.0)
      if ucp is not None:
        X = (X + DX).clip(0.0, 1.0)
      X = normalize(X, args.atk_dataset)

      pred = model(X).argmax(dim=-1)

      total += len(pred)
      correct += (pred == Y).sum()
  
  return correct / total


def test_pcr(model, dataloader, ucp, resizer='tile') -> float:
  ''' Prediction Change Rate '''

  if True:
    X, _ = iter(dataloader).next()
    DX = ucp_expand_batch(ucp, X.shape, resizer=resizer)
    
  total, changed = 0, 0
  with torch.no_grad():
    for X, Y in dataloader:
      X, Y = X.to(device), Y.to(device)
      AX = (X + DX).clip(0.0, 1.0)
      X  = normalize(X,  args.atk_dataset)
      AX = normalize(AX, args.atk_dataset)

      pred1 = model(X ).argmax(dim=-1)
      pred2 = model(AX).argmax(dim=-1)

      total += len(pred1)
      changed += (pred1 != pred2).sum()
  
  return changed / total


def test_ucp_ex(ucp, model, dataloader, test_fn, resizer='tile'):
  for factor in [-2, -0.5, 0.5, 2]:
    ucp_v = ucp * factor
    ret = test_fn(model, dataloader, ucp_v, resizer=resizer)
    print(f'   ucp(x{factor}): {ret:.3%}')
  
  ucp_v = ucp_norm(ucp, norm=True)
  ret = test_fn(model, dataloader, ucp_v, resizer=resizer)
  print(f'   ucp-norm: {ret:.3%}')

  for mag in [1e-3, 1e-2, 1e-1]:
    ucp_v = ucp_noise(ucp, mag=mag)
    ret = test_fn(model, dataloader, ucp_v, resizer=resizer)
    print(f'   ucp-noise({float_to_str(mag)}): {ret:.3%}')

  for mode in ['vertical', 'horizontal']:
    ucp_v = ucp_flip(ucp, mode)
    ret = test_fn(model, dataloader, ucp_v, resizer=resizer)
    print(f'   ucp-clip({mode}): {ret:.3%}')
  
  for angle in [90, 180, 270]:
    ucp_v = ucp_rotate(ucp, angle)
    ret = test_fn(model, dataloader, ucp_v, resizer=resizer)
    print(f'   ucp-rotate({angle}): {ret:.3%}')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='resnet18', choices=MODELS, help='model to attack')
  parser.add_argument('-D', '--atk_dataset', default='imagenet-1k', choices=DATASETS, help='dataset to attack (atk_dataset can be different from train_dataset')
  parser.add_argument('--split', default='test', choices=['test', 'train'], help='split name for atk_dataset')
  parser.add_argument('--load', help='path to trained model weights folder, named like <model>_<train_dataset>')
  
  parser.add_argument('--ucp', help='path to ucp.npy file')
  parser.add_argument('--resizer', default='tile', choices=['tile', 'interpolate'], help='resize UCP when shape mismatch')
  parser.add_argument('--ex', action='store_true', help='enable extended tests on ucp stability')
  
  parser.add_argument('-B', '--batch_size', type=int, default=100)
  parser.add_argument('--data_path', default='data', help='folder path to downloaded dataset')
  parser.add_argument('--log_path', default='log', help='folder path to local trained model weights and logs')
  args = parser.parse_args()

  print(f'>> testing on dataset "{args.atk_dataset}" of split "{args.split}"')
  print(f'>> using resizer "{args.resizer}" in case of shape mismatch')

  test(args)
