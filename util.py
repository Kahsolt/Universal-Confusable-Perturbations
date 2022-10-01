#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

from typing import Union

import torch
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cuda':
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class ValueWindow:

  def __init__(self, size:int=20):
    self.size = size
    self.data = []

  @property
  def mean(self) -> float:
    return sum(self.data) / len(self.data) if len(self.data) else 0.0

  def update(self, v:float):
    self.data.append(v)
    self.data = self.data[-self.size:]


def ucp_norm(ucp:Union[torch.Tensor, np.ndarray], norm=False) -> torch.Tensor:
  # shift to [0, 1]
  x = ucp - ucp.min()
  if x.max() > 1.0 or norm: x /= x.max()
  return x

def ucp_noise(ucp:torch.Tensor, mag=1e-5) -> torch.Tensor:
  noise = (torch.rand_like(ucp) * 2 - 1) * mag
  return ucp + noise

def ucp_expand_batch(ucp:torch.Tensor, shape:torch.Size, resizer='tile') -> torch.Tensor:
  ''' [C, h, w] -> [B, C, H, W] '''

  B, C, H, W = shape

  if resizer == 'tile':
    ucp = ucp_tile(ucp, torch.Size([C, H, W]))
  elif resizer == 'interpolate':
    ucp = ucp_interpolate(ucp, torch.Size([C, H, W]))
  else:
    raise ValueError('choose from ["tile", "interpolate"]')
  
  ucp = ucp.unsqueeze(dim=0).repeat([B, 1, 1, 1])   # expand batch

  return ucp

def ucp_tile(ucp:torch.Tensor, shape:torch.Size) -> torch.Tensor:
  ''' [C, h, w] -> [C, H, W] '''

  c, h, w = ucp.shape
  C, H, W = shape
  assert c == C

  rf = int(np.ceil(H / h))
  cf = int(np.ceil(W / w))

  ucp = ucp.tile([1, rf, cf])   # tile
  ucp = ucp[:, :H, :W]          # clip edge

  return ucp

def ucp_interpolate(ucp:torch.Tensor, shape:torch.Size, mode='bilinear') -> torch.Tensor:
  ''' [C, h, w] -> [C, H, W] '''

  c, h, w = ucp.shape
  C, H, W = shape
  assert c == C

  ucp = ucp.unsqueeze(dim=0)
  ucp = F.interpolate(ucp, (H, W), mode=mode)
  ucp = ucp.squeeze(dim=0)

  return ucp

def ucp_flip(ucp:torch.Tensor, mode='vertical') -> torch.Tensor:
  if mode in ['v', 'vertical']:
    return ucp.flip(dims=[1])
  elif mode in ['h', 'horizontal']:
    return ucp.flip(dims=[2])
  else: raise ValueError('choose mode for ["vertical"/"v", "horizontal"/"h"]')

def ucp_rotate(ucp:torch.Tensor, angle=90) -> torch.Tensor:
  if angle % 90 != 0:
    raise ValueError('angle must be divisible by 90')
  
  return ucp.rot90(k=angle//90,  dims=(1, 2))


def float_to_str(x:str, n_prec:int=3) -> str:
  # integer
  if int(x) == x: return str(x)
  
  # float
  sci = f'{x:e}'
  frac, exp = sci.split('e')
  
  frac_r = round(float(frac), n_prec)
  frac_s = f'{frac_r}'
  if frac_s.endswith('.0'):   # remove tailing '.0'
    frac_s = frac_s[:-2]
  exp_i = int(exp)
  
  if exp_i != 0:
    # '3e-5', '-1.2e+3'
    return f'{frac_s}e{exp_i}'
  else:
    # '3.4', '-1.2'
    return f'{frac_s}'
