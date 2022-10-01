# Universal-Confusable-Perturbations

    When an adversarial attak is targeted for uniform distribution...

----

# CURRENTLY STILL WORKING ON ...

​    As far as I know, the concept of [Universal Adversarial Perturbations](https://arxiv.org/abs/1610.08401) stems from the DeepFool method, which optimizes one **common pertubation** (shaped like a single image, a.k.a. adversarial texture) so that most of images `x` in dataset `X` would be misclassified by the victim model `f` when this pertubation `dx` is added to as a noise signal, i.e. : `∀x∈X. P[f(x + dx) != y] -> 1`. It usually achieves this goal by leading the model to predict **another class** (either intended or not), However, this universal pertubation might still somehow contain information about this certain dataset.  

​    Now we consider of what if we **set the optimizing target to a uniform distribution**. That's to say: `∀x∈X. P[f(x + dx)] ~ U[0, 1]`. Intuitively we are searching for a **Universal Confusable Perturbation (UCP)** such that when added to a benign image, the victim classification model hesitates to make decisions. 

​    This sounds even weaker in adversarial attack scenario, but could we achieve better transferability across models/datasets by this way? <del> Even make a tool for bypassing existing AI censorshipments.</del>


### Experiments

We format our experiment name to indicate the experimental settings like: `<model>_<train_dataset>-<atk_dataset>_<method>` which means using `<model>` pretrained on `<train_dataset>` to search for UCPs on `<atk_dataset>` by `<method>` attack.

Here are currently avaiable choices for each component:

```txt
model (all availables in `pytorch.models`)
  resnet18
  resnet50
  densenet121
  ...

train_dataset
  imagenet           // currently `pytorch.models` only provides weights for `imagenet`
  ...                // if you refined on any other dataset by yourself (use `attack.py --load <ckpt.pth>`)

atk_dataset (available through `pytorch.datasets`)
  mnist              // 28 x 28
  svhn               // 32 x 32
  cifar10            // 32 x 32
  cifar100           // 32 x 32
  tiny-imagenet (*)  // 64 x 64
  imagenet-1k (*)    // 224 x 224

method
  pgd
  pgdl2
  mifgsm

(*): see section "dataset download & preprocess" to download manually
```

#### Attack on the same dataset, i.e. : "resnet18_imagenet-imagenet-1k"

This setting means: use a resnet18 model pretrained on ImageNet to search for UCPs on ImageNet-1k.


#### Attack on anthor dataset, i.e. : "resnet18_imagenet-svhn"

This setting means: use a resnet18 model pretrained on ImageNet but to search for UCPs on SVHN. 

⚪ resnet18_imagenet-svhn_pgd

generated UCP picture is like:

| eps \ alpha | 0.01 | 0.005 | 0.003 | 0.001 |
| :-: | :-: | :-: | :-: |
| 0.1  |  |  |  |  |
| 0.05 |  |  |  |  |
| 0.03 |  |  |  |  |
| 0.01 |  |  |  |  |

prediction change rate (PCR) tests:

| svhn \ cifar10 | 0.01 | 0.005 | 0.003 | 0.001 |
| :-: | :-: | :-: | :-: |
| 0.1  |  |  |  |  |
| 0.05 |  |  |  |  |
| 0.03 |  |  |  |  |
| 0.01 |  |  |  |  |

⚪ resnet18_imagenet-svhn_pgdl2


⚪ resnet18_imagenet-svhn_mifgsm




#### generate UCP on pretrained models

| Model | Dataset (pretrain on) | Dataset (attack on) |
| :-: | :-: | :-: |
| resnet18 | imagenet | svhn          |
| resnet18 | imagenet | cifar10       |
| resnet50 | imagenet | cifar10       |
| resnet50 | imagenet | cifar100      |
| resnet50 | imagenet | tiny-imagenet |

```shell
# load pretrained resnet18 to generate a UCP on cifar10
# => generated UCP saves at `log/<model>_<train_dataset>/<model>_<train_dataset>-<atk_dataset>.npy`
#     e.g. `log/resnet18_imagenet/resnet18_imagenet-cifar10.npy`
python attack.py -M resnet18 -D cifar10 --steps 3000
# show the generated UCP
python show.py log/resnet18_imagenet/resnet18_imagenet_cifar10.npy
```

#### generate UCP on your own models

```shell
# train a capable model
# => saved at `log/<model>_<train_dataset>`
python train.py -M resnet18 -D cifar10

# generate a UCP
# => saved at `log/<model>_<train_dataset>/<model>_<train_dataset>-<atk_dataset>.npy`
python attack.py --load resnet18_cifar10
# try attack on another dataset `mnist`
python attack.py --load resnet18_cifar10 -D mnist

# test 
python test.py --load resnet18_cifar10
python test.py -M resnet18 -D mnist --load
```

#### dataset download & preprocess

- imagenet-1k:
  - download `clean_resized_images.zip`, unzip all image files under `data\imagenet-1k\val\`
  - put index file `image_name_to_class_id_and_name.json` under `imagenet-1k`
- [tiny-imagenet-200](https://tiny-imagenet.herokuapp.com)
  - download & unzip `tiny-imagenet-200.zip` under `data`

```
├── imagenet-1k
│   ├── image_name_to_class_id_and_name.json    <- index file
│   └── val
│       └── ILSVRC2012_val_*.png
│── tiny-imagenet-200
│   ├── test
│   ├── train
│   ├── val
│   ├── val_annotations.txt
│   ├── wnids.txt
│   └── words.txt
│── ...
```

#### troubleshoot

Q: It seems that loss is very high, and not decrease.
A: Don't worry, because Cross-Entropy Loss on a uniform distribution is naturally high, theorical lower bound for a uniform distribution lengthed `1000` like ImageNet is the constant `6.9078`, hence loss `~= 7.1` is ok.


#### references

- Universal adversarial perturbations: [https://arxiv.org/abs/1610.08401](https://arxiv.org/abs/1610.08401)
  - PyTorch implementation: [https://github.com/NetoPedro/Universal-Adversarial-Perturbations-Pytorch][https://github.com/NetoPedro/Universal-Adversarial-Perturbations-Pytorch] 
- DeepFool: a simple and accurate method to fool deep neural networks: [https://arxiv.org/abs/1511.04599v3](https://arxiv.org/abs/1511.04599v3)
- TinyImageNet-Benchmarks: [https://github.com/meet-minimalist/TinyImageNet-Benchmarks](https://github.com/meet-minimalist/TinyImageNet-Benchmarks)

----

by Armit
2022/09/27 
