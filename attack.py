#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

import logging
import os
from argparse import ArgumentParser

import torch
from tensorboardX import SummaryWriter
import numpy as np

from nannomono import NannoMono, ATK_METHODS
from model import *
from data import *
from util import *

# NOTE: perfcount
#   .to(device)   7s
#   forward       0.4s
#   backward      0.27s
#   atk()         ~20s


def attack(args):
  ''' Dirs '''
  log_dp = os.path.join(args.log_path, args.log_dn)
  os.makedirs(args.data_path, exist_ok=True)
  os.makedirs(log_dp, exist_ok=True)

  ''' Logger '''
  logger = logging.getLogger('ucp')
  logging.basicConfig(level=logging.INFO)
  logger.setLevel(logging.INFO)
  handler = logging.FileHandler(os.path.join(log_dp, "attack.log"))
  handler.setLevel(logging.INFO)
  formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S")
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  sw = SummaryWriter(log_dp)

  ''' Data'''
  dataloader = get_dataloader(args.atk_dataset, args.batch_size, args.data_path, n_worker=args.n_worker)

  ''' Model '''
  model = get_model(args.model, ckpt_fp=args.ckpt_fp).to(device)
  model.eval()

  ''' Info '''
  logger.info(f'[Attack]')
  logger.info(f'   expriment name      = {ucp_name(args.model, args.train_dataset, args.atk_dataset, args.method, args.eps, args.alpha, args.alpha_to)}')
  logger.info(f'   model               = {args.model}')
  logger.info(f'     weights ckpt      = {"local trained" if args.load else "torchub pretrained"}')
  logger.info(f'   train_dataset       = {args.train_dataset}')
  logger.info(f'   atk_dataset         = {args.atk_dataset}')
  logger.info(f'     n_examples        = {len(dataloader.dataset)}')
  logger.info(f'     n_batches         = {len(dataloader)}')
  logger.info(f'     n_epochs          = {args.steps / len(dataloader)}')
  logger.info(f'   atk_method          = {args.method}')
  logger.info(f'   atk_eps             = {args.eps}')
  logger.info(f'   atk_alpha           = {args.alpha}')
  logger.info(f'     decay             = {args.alpha_to}')
  logger.info(f'   atk_steps           = {args.steps}')
  logger.info(f'     steps_per_batch   = {args.steps_per_batch}')
  logger.info(f'     micro_steps       = {args.micro_steps}')

  atk_setting = f'{args.atk_dataset}{ucp_suffix(args.method, args.eps, args.alpha, args.alpha_to)}'

  ''' Attack '''
  kwargs = {
    'normalizer': lambda x: normalize(x, args.atk_dataset),   # delay normalize to attacker
  }
  if args.alpha_to is not None:
    kwargs.update({
      'alpha_decay': True,
      'alpha_from' : args.alpha,
      'alpha_to'   : args.alpha_to,
      'micro_steps': args.micro_steps,
    })
  atk = NannoMono(model, method=args.method, eps=args.eps, alpha=args.alpha, steps=args.steps_per_batch, **kwargs)
  with torch.no_grad():
    X, _ = iter(dataloader).next()
    B = X.shape[0]
    x = X[0].unsqueeze(0).to(device)                    # [1, C, H, W]
    atk.init_ucp(x.shape)                               # random init a pertubation
    y_hat = model(x)                                    # [1, N_CLASS]
    Y_tgt = torch.ones_like(y_hat) / y_hat.shape[-1]    # make uniform distribution
    Y_tgt = Y_tgt.repeat([B, 1])                        # [B, N_CLASS]
  
  steps = 0
  losses = ValueWindow()
  while steps <= args.steps:
    for X, _ in dataloader:
      X = X.to(device)
      
      loss, ucp = atk(X, Y_tgt)
      losses.update(loss)

      steps += 1
      if steps > args.steps: break

      for k in dir(args):
        if k.endswith('_interval'):
          if steps % getattr(args, k) == 0:
            ucp_npy = ucp.squeeze(0).detach().cpu().numpy()
            break

      if steps % args.log_interval == 0:
        sw.add_scalar(f'Loss-{atk_setting}', losses.mean, steps)
        sw.add_scalar(f'L1-{atk_setting}', np.abs   (ucp_npy).mean(), steps)
        sw.add_scalar(f'L2-{atk_setting}', np.square(ucp_npy).mean(), steps)

        log_alpha = f'alpha={atk.alpha}' if args.alpha_to else ''
        logger.info(f'[steps {steps}/{args.steps}] loss={losses.mean} {log_alpha}')
      
      if steps % args.show_interval == 0:
        sw.add_image(f'{atk_setting}', ucp_norm(ucp_npy, norm=True), steps)

      if steps % args.save_interval == 0:
        np.save(args.save_fp, ucp_npy)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='resnet18', choices=MODELS, help='victim model with pretrained weight')
  parser.add_argument('-D', '--atk_dataset', default='svhn', choices=DATASETS, help='victim dataset')
  parser.add_argument('--load', help='path to trained model weights folder, named like <model>_<train_dataset>')

  parser.add_argument('--method', default='pgd', choices=ATK_METHODS, help='base attack method')
  parser.add_argument('--eps', type=float, default=0.03, help='attack param eps, total pertubation limit')
  parser.add_argument('--alpha', type=float, default=0.001, help='attack param alpha, stepwise pertubation limit ~= learning rate')
  parser.add_argument('--alpha_to', default=None, help='enable alpha_decay mechanism if set, e.g.: 2e-5')
  parser.add_argument('--steps', type=int, default=2000, help='attack iteration steps on whole dataset, e.g.: 3000')
  parser.add_argument('--steps_per_batch', type=int, default=20, help='attack iteration steps on one batch, e.g.: 40')

  parser.add_argument('-B', '--batch_size', type=int, default=128)
  parser.add_argument('-J', '--n_worker', type=int, default=0)
  parser.add_argument('--overwrite', action='store_true', help='force retrain and overwrite existing <ucp>.npy')
  parser.add_argument('--data_path', default='data', help='folder path to downloaded dataset')
  parser.add_argument('--log_path', default='log', help='folder path to local trained model weights and logs')
  parser.add_argument('--log_interval', type=int, default=10)
  parser.add_argument('--show_interval', type=int, default=200)
  parser.add_argument('--save_interval', type=int, default=1000)
  args = parser.parse_args()

  if args.eps <= 0.0:
    raise ValueError('--eps should > 0')
  if args.alpha > args.eps:
    raise ValueError('--alpha should be smaller than --eps')
  if args.alpha_to is not None:
    args.alpha_to = float(args.alpha_to)
  args.micro_steps = args.steps * args.steps_per_batch

  if args.load:
    print('[Ckpt] use local trained weights')
    try:    args.model, args.train_dataset = args.load.split('_')
    except: breakpoint()
    args.log_dn = args.load
    args.ckpt_fp = os.path.join(args.log_path, args.log_dn, 'model-best.pth')
    if not os.path.exists(args.ckpt_fp):
      raise ValueError(f'You have not trained the local model {args.ckpt_fp} yet')
  else:
    print('[Ckpt] use pretrained weights from torchvision/torchhub')
    args.train_dataset = 'imagenet'     # NOTE: currently all `torchvision.models` are pretrained on `imagenet`
    args.log_dn = load_name(args.model, args.train_dataset)
    args.ckpt_fp = None

  ucp_fn = ucp_name(args.model, args.train_dataset, args.atk_dataset, args.method, args.eps, args.alpha, args.alpha_to) + '.npy'
  args.save_fp = os.path.join(args.log_path, args.log_dn, ucp_fn)
  if os.path.exists(args.save_fp) and not args.overwrite:
    print(f'safely ignore due to {args.save_fp} exists')
  else:
    attack(args)
