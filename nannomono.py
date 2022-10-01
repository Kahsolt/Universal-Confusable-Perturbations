#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/28 

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchattacks.attack import Attack
import numpy as np

# 不依赖 true_label 的极性
ATK_METHODS = [
  # Linf
  'pgd', 
  'mifgsm',
  # L2
  'pgdl2', 
]

class AlphaScheduler:

  ''' quadra-decay from `from_` to `to` in `decay_micro_steps` steps '''

  def __init__(self, attacker, decay_micro_steps:int, from_:float=1e-3, to:float=2e-5):
    self.attacker = attacker

    self.steps = 0
    self.attacker.alpha = from_

    self.f = lambda steps: (from_ - to) / (decay_micro_steps ** 2) * (steps - decay_micro_steps) ** 2 + to

  def step(self):
    self.steps += 1
    self.attacker.alpha = self.f(self.steps)


# なんのもの
class NannoMono(Attack):

  def __init__(self, model, method='pgd', eps=8/255, alpha=1/255, steps=40, **kwargs):
    super().__init__("NannoMono", model)
    self.eps = eps
    self.alpha = alpha
    self.steps = steps
    self.method = method
    self._supported_mode = ['targeted']
    self.kwargs = kwargs

    # alpha scheduler
    self.alpha_sched = ('alpha_decay' in kwargs) and AlphaScheduler(self, kwargs['micro_steps'], kwargs['alpha_from'], kwargs['alpha_to'])

    # set for compatibility
    self._attack_mode = 'targeted'
    self._targeted = True
    self._target_map_function = lambda x: x

    self.kl_loss = nn.KLDivLoss(reduce='batchmean')
    self.ce_loss = nn.CrossEntropyLoss()
    self.mse_loss = nn.MSELoss(reduction='none')
    self.eps_for_division = 1e-9

  def init_ucp(self, shape: torch.Size):
    if self.method.endswith('l2'):
      mag = self.eps / shape.numel()
    else:
      mag = self.eps
    self.ucp = torch.empty(shape).uniform_(-mag, mag).to(self.device)

  def forward(self, images, labels):
    images = images.clone()   # avoid inplace grad overwrite
    labels = labels.clone()
    return getattr(self, self.method)(images, labels)

  def pgd(self, images: torch.Tensor, labels: torch.Tensor):
    ''' modified from torchattacks.attacks.PGD '''

    B = len(images)
    images = images.detach().to(self.device)
    labels = labels.detach().to(self.device)

    adv_images = images.detach() + self.ucp.repeat([B, 1, 1, 1])  # broadcast, [B, C, H, W]
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(self.steps):
      adv_images.requires_grad = True
      outputs = self.model(adv_images)

      # Calculate loss (targeted)
      #loss = self.kl_loss(F.log_softmax(outputs), labels)
      loss = self.ce_loss(outputs, labels)

      # Update adversarial images
      grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
      adv_images = adv_images.detach() - self.alpha * grad.sign()     # 朝梯度负向走一步
      delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
      adv_images = torch.clamp(images + delta, min=0, max=1).detach()

      if self.alpha_sched: self.alpha_sched.step()

    self.ucp = delta.mean(axis=0)   # aggregate, [C, H, W]
    return loss, self.ucp

  def pgdl2(self, images: torch.Tensor, labels: torch.Tensor):
    ''' modified from torchattacks.attacks.PGDL2 '''

    B = len(images)
    images = images.detach().to(self.device)
    labels = labels.detach().to(self.device)

    adv_images = images.detach() + self.ucp.repeat([B, 1, 1, 1])  # broadcast, [B, C, H, W]
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(self.steps):
      adv_images.requires_grad = True
      outputs = self.model(adv_images)

      # Calculate loss
      #loss = self.kl_loss(F.log_softmax(outputs), labels)
      loss = self.ce_loss(outputs, labels)

      # Update adversarial images
      grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
      grad_norms = torch.norm(grad.reshape(B, -1), p=2, dim=1) + self.eps_for_division
      grad = grad / grad_norms.reshape(B, 1, 1, 1)
      adv_images = adv_images.detach() - self.alpha * grad     # 朝梯度负向走一步

      delta = adv_images - images
      delta_norms = torch.norm(delta.view(B, -1), p=2, dim=1)
      factor = self.eps / delta_norms
      factor = torch.min(factor, torch.ones_like(delta_norms))
      delta = delta * factor.reshape(-1, 1, 1, 1)
      adv_images = torch.clamp(images + delta, min=0, max=1).detach()

      if self.alpha_sched: self.alpha_sched.step()

    self.ucp = delta.mean(axis=0)   # aggregate, [C, H, W]
    return loss, self.ucp

  def mifgsm(self, images: torch.Tensor, labels: torch.Tensor):
    ''' modified from torchattacks.attacks.MIFGSM '''

    momentum_decay = self.kwargs.get('momentum_decay', 1.0)

    B = len(images)
    images = images.detach().to(self.device)
    labels = labels.detach().to(self.device)

    momentum = torch.zeros_like(images).detach().to(self.device)

    adv_images = images.detach() + self.ucp.repeat([B, 1, 1, 1])   # broadcast, [B, C, H, W]
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(self.steps):
      adv_images.requires_grad = True
      outputs = self.model(adv_images)

      # Calculate loss
      #loss = self.kl_loss(F.log_softmax(outputs), labels)
      loss = self.ce_loss(outputs, labels)

      # Update adversarial images
      grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
      grad = grad / torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
      grad = grad + momentum * momentum_decay
      momentum = grad

      adv_images = adv_images.detach() - self.alpha * grad.sign()
      delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
      adv_images = torch.clamp(images + delta, min=0, max=1).detach()

      if self.alpha_sched: self.alpha_sched.step()

    self.ucp = delta.mean(axis=0)  # aggregate, [C, H, W]
    return loss, self.ucp

  # FIXME: need to refactor
  def deepfool(image, net, num_classes, overshoot, max_iter):
    ''' https://github.com/BXuan694/Universal-Adversarial-Perturbation/blob/master/deepfool.py '''
    """
      :param image:
      :param net: network (input: images, output: values of activation **BEFORE** softmax).
      :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
      :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
      :param max_iter: maximum number of iterations for deepfool (default = 50)
      :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """

    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = f_image.argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    k_i = label

    while k_i == label and loop_i < max_iter:
      pert = np.inf
      fs[0, I[0]].backward(retain_graph=True)
      grad_orig = x.grad.data.cpu().numpy().copy()

      for k in range(1, num_classes):
        #zero_gradients(x)    # FIXME: API changed, how to fix??

        fs[0, I[k]].backward(retain_graph=True)
        cur_grad = x.grad.data.cpu().numpy().copy()

        # set new w_k and new f_k
        w_k = cur_grad - grad_orig
        f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

        pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

        # determine which w_k to use
        if pert_k < pert:
          pert = pert_k
          w = w_k

      # compute r_i and r_tot
      # Added 1e-4 for numerical stability
      r_i =  (pert+1e-4) * w / np.linalg.norm(w)
      r_tot = np.float32(r_tot + r_i)

      pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

      x = Variable(pert_image, requires_grad=True)
      # print(image.shape)
      # print(x.view(1,1,image.shape[0],-1).shape)
      fs = net.forward(x.view(1,1,image.shape[1],-1))
      k_i = np.argmax(fs.data.cpu().numpy().flatten())

      loop_i += 1

    return (1+overshoot)*r_tot, loop_i, label, k_i, pert_image
