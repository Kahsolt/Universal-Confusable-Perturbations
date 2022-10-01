#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/29 

from argparse import ArgumentParser

from attack import MODELS, DATASETS, ATK_METHODS

EPS = [
  0.1,
  0.05,
  0.03,
  0.01,
]
ALPHA = [
  0.01,
  0.005,
  0.003,
  0.001,
]


def mk(args):
  fp = f'train_{args.name}.cmd'
  with open(fp, 'w', encoding='utf-8') as fh:
    for method in ATK_METHODS:
      fh.write(f'@REM {method}\n')
      for eps in EPS:
        for alpha in ALPHA:
          fh.write(f'python attack.py -M {args.model} -D {args.atk_dataset} --method {method} --eps {eps} --alpha {alpha}\n')
        fh.write('\n')
      fh.write('\n')


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='resnet18', choices=MODELS, help='model to attack')
  parser.add_argument('-D', '--atk_dataset', default='imagenet-1k', choices=DATASETS, help='dataset to attack (atk_dataset can be different from train_dataset')
  args = parser.parse_args()

  args.train_dataset = 'imagenet'   # NOTE: currently all torchvision.models are pretrained on `imagenet`
  args.name = f'{args.model}_{args.train_dataset}-{args.atk_dataset}'

  mk(args)
