#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

import os
import pickle as pkl
from argparse import ArgumentParser

import numpy as np
from model import *
from data import *
from util import *


def log(stats: dict, path: str, name:str, value: float):
  def set_dval():
    node = stats
    for seg in path.split('/'):
      if seg not in node:
        node[seg] = { }
      node = node[seg]
    node[name] = value
  
  set_dval()
  print(f'   {name}: {value:.3%}')


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

  ''' Stats '''
  # load acc/pcr records
  stats_fp = os.path.join(args.log_path, 'stats.pkl')
  if os.path.exists(stats_fp):
    with open(stats_fp, 'rb') as fh:
      stats = pkl.load(fh)
  else:
    # { 'acc|pcr': { '<ucp_name>': { '<resizer>': { '<atk_dataset>': { '<name>': float } } } } }
    # { 'clean': { '<atk_dataset>': { '<name>': float } } }
    stats = { }
  
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
      log(stats, f'clean/{args.atk_dataset}', 'clean', acc)
      for mag in [1e-3, 1e-2, 1e-1]:
        acc = test_acc(model, dataloader, noise=mag)
        log(stats, f'clean/{args.atk_dataset}', f'clean-noise({float_to_str(mag)})', acc)
    else:
      print('atk_dataset is not compatible is with train_dataset')
  
  # Try attacked accuracy
  if args.ucp:
    ucp_name = os.path.splitext(os.path.basename(args.ucp))[0]
    ucp = torch.from_numpy(np.load(args.ucp)).to(device)

    # Try testing remnet accuracy (:= 1 - misclf rate) after adding UCP
    if chk_dataset_compatible(args.train_dataset, args.atk_dataset):
      acc = test_acc(model, dataloader, ucp, resizer=args.resizer)
      log(stats, f'acc/{ucp_name}/{args.resizer}/{args.atk_dataset}', 'ucp', acc)

      if args.ex:
        test_ucp_ex(ucp, model, dataloader, test_fn=test_acc, resizer=args.resizer, stats=stats, ucp_name=ucp_name)
      
    # Try testing predction changing rate after adding UCP
    if True:
      print(f'[Pred Change Rate]')
      pcr = test_pcr(model, dataloader, ucp, resizer=args.resizer)
      log(stats, f'pcr/{ucp_name}/{args.resizer}/{args.atk_dataset}', 'ucp', pcr)

      if args.ex:
        test_ucp_ex(ucp, model, dataloader, test_fn=test_pcr, resizer=args.resizer, stats=stats, ucp_name=ucp_name)

  # save acc/pcr records
  with open(stats_fp, 'wb') as fh:
    pkl.dump(stats, fh)


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
  
  return (correct / total).item()


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
  
  return (changed / total).item()


def test_ucp_ex(ucp, model, dataloader, test_fn, resizer='tile', **kwargs):
  metric = test_fn.__name__.split('_')[-1]
  stats = kwargs.get('stats', { })
  ucp_name = kwargs.get('ucp_name', 'anonymous')

  for factor in [-2, -0.5, 0.5, 2]:
    ucp_v = ucp * factor
    ret = test_fn(model, dataloader, ucp_v, resizer=resizer)
    log(stats, f'{metric}/{ucp_name}/{args.resizer}/{args.atk_dataset}', f'ucp(x{factor})', ret)
  
  ucp_v = ucp_norm(ucp, norm=True)
  ret = test_fn(model, dataloader, ucp_v, resizer=resizer)
  log(stats, f'{metric}/{ucp_name}/{args.resizer}/{args.atk_dataset}', 'ucp-norm', ret)

  for mag in [1e-3, 1e-2, 1e-1]:
    ucp_v = ucp_noise(ucp, mag=mag)
    ret = test_fn(model, dataloader, ucp_v, resizer=resizer)
    log(stats, f'{metric}/{ucp_name}/{args.resizer}/{args.atk_dataset}', f'ucp-noise({float_to_str(mag)})', ret)

  for mode in ['vertical', 'horizontal']:
    ucp_v = ucp_flip(ucp, mode)
    ret = test_fn(model, dataloader, ucp_v, resizer=resizer)
    log(stats, f'{metric}/{ucp_name}/{args.resizer}/{args.atk_dataset}', f'ucp-flip({mode})', ret)
  
  for angle in [90, 180, 270]:
    ucp_v = ucp_rotate(ucp, angle)
    ret = test_fn(model, dataloader, ucp_v, resizer=resizer)
    log(stats, f'{metric}/{ucp_name}/{args.resizer}/{args.atk_dataset}', f'ucp-rotate({angle})', ret)


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
  if args.ucp:
    print(f'>> using ucp "{os.path.basename(args.ucp)}"')
    print(f'>> using resizer "{args.resizer}" in case of shape mismatch')

  test(args)
