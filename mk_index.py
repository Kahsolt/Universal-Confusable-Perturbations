#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/30 

# Designed corresponding to file:
#   'attack_resnet18_imagenet-svhn_gridsearch.cmd'
#   'attack_resnet18_imagenet-svhn_ablations.cmd'

import os
import pickle as pkl
import numpy as np

from attack import ATK_METHODS
from util import *

img_dp  = 'img'
log_dp  = 'log'
html_fn = 'index.html'

# for `gridsearch`
ALPHA = [ 0.01, 0.005, 0.001, 0.0005 ]
EPS_Linf = [ 0.1, 0.05, 0.03, 0.01 ]
EPS_L2 = [ 3, 1, 0.5, 0.3 ]
EPS = {
  'pgd'   : EPS_Linf,
  'mifgsm': EPS_Linf,
  'pgdl2' : EPS_L2,
}

# for `ablations`
ALPHA_DECAYS = {
  'const': [
    # (method, eps, alpha)
    ('pgd', 0.03, 0.0001),
    ('pgd', 0.03, 0.00001),
  ],
  'decay': [
    # (method, eps, alpha_from, alpha_to)
    ('pgd', 0.03, 0.001, 1e-4),
    ('pgd', 0.03, 0.001, 2e-5),
  ],
}
MODELS = [
  # model
  'resnet18',
  'resnet34',
  'resnet50',
  'resnext50_32x4d',
  'wide_resnet50_2',
  'densenet121',
  'efficientnet_v2_s',
  'shufflenet_v2_x1_5',
  'squeezenet1_1',
#  'inception_v3',
  'mobilenet_v3_large',
  'regnet_y_400mf',
#  'vit_b_16',
#  'swin_t',
]
# (eps, alpha)
MODELS_E_A = (0.03, 0.001)


def mk_index():
  html_skel = '''<!DOCTYPE>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width">
  <title>Universal Confusable Pertubations Demo</title>
  <link rel="stylesheet" href="w3.css">
  <style>
    /* table, th, td { border: 1px solid black; } */
  </style>
</head>
<body class="w3-sand">
<!-- menu -->
<div class="w3-bar w3-blue">
<span class="w3-bar-item w3-large"> %s </span>
%s  <!-- buttons or span-splits -->
</div>

<!-- content -->
<div class="w3-container">
<!-- tables or divs -->
%s
</div>

<!-- control -->
<script>
function openTable(tabpage) {
  let x = document.getElementsByClassName("tabpage");
  for (let i = 0; i < x.length; i++)
    x[i].style.display = "none";
  document.getElementById(tabpage).style.display = "block";
}
</script>
</body>
</html>
'''

  mk_tr     = lambda text: f'<tr>\n{text}\n</tr>'
  mk_td     = lambda text: f'<td>\n{text}\n</td>'
  mk_img    = lambda src:  f'<img class="w3-image" src="{src}"></img>'
  mk_p      = lambda text: f'<p>{text}</p>'
  mk_span   = lambda text: f'<span>{text}</span>'
  mk_div    = lambda text, center=True: f'<div class="w3-container{" w3-center" if center else ""}">\n{text}\n</div>'
  mk_button = lambda text: f'<button class="w3-bar-item w3-button w3-large" onclick="openTable(\'{text}\')">{text}</button>'
  mk_table  = lambda id, text, display=True: f'<table id="{id}" class="w3-table tabpage" style="display:{"block" if display else "none"}">\n{text}\n</table>'
  mk_divP   = lambda id, text, display=True: f'<div id="{id}"class="w3-container tabpage" style="display:{"block" if display else "none"}">\n{text}\n</div>'

  mk_menusplit = lambda: f'<span class="w3-bar-item"> | </span>'
  mk_linetext  = lambda text: f'<span style="display:block">{text}</span>'
  mk_grid      = lambda text: f'<div class="w3-row" style="display:inline-block">\n{text}\n</div>'
  mk_row       = lambda text, width='s6': f'<div class="w3-col {width}">\n{text}\n</div>'
  mk_card      = lambda fp, title: mk_div(f'{mk_img(fp)}\n{mk_p(title)}')
  mk_info      = lambda items: mk_div('\n'.join(mk_linetext(it) for it in items), center=False)
  mk_card_ex   = lambda fp, title, items: mk_grid(f'{mk_row(mk_card(fp, title), width="s8 m8 l8")}\n{mk_row(mk_info(items), width="s4 m4 l4")}')

  # test settings
  if 'test settings':
    model         = 'resnet18'
    train_dataset = 'imagenet'
    atk_dataset   = 'svhn'          # ucp genrated on this
    apply_dataset = 'imagenet-1k'   # ucp applied on this to test attack success rate
    log_dn = load_name(model, train_dataset)
    name = f'{log_dn}-{atk_dataset}'

  # test stats if available
  stats_fp = os.path.join(log_dp, 'stats.pkl')
  if os.path.exists(stats_fp):
    with open(stats_fp, 'rb') as fh:
      tstats = pkl.load(fh)
  else:
    tstats = None

  def get_stats_items(model, ucp_base):
    def get_dval(path):
      nonlocal tstats
      if tstats is None: return None
      node = tstats
      for seg in path.split('/'):
        if seg not in node:
          return None
        node = node[seg]
      return node
  
    log_dn = load_name(model, train_dataset)
    ucp_fp = os.path.join(log_dp, log_dn, ucp_base + '.npy')
    stats = ucp_stats(np.load(ucp_fp))
    items = [f'{k}: {v:g}' for k, v in stats.items()]
    try:
      items += [
        '=' * 10,
        'acc: {:.3%} / {:.3%}'.format(get_dval(f'acc/{ucp_base}/tile/{apply_dataset}/ucp'), get_dval(f'acc/{ucp_base}/interpolate/{apply_dataset}/ucp')),
        'pcr: {:.3%} / {:.3%}'.format(get_dval(f'pcr/{ucp_base}/tile/{apply_dataset}/ucp'), get_dval(f'pcr/{ucp_base}/interpolate/{apply_dataset}/ucp')),
      ]
    except Exception as e:
      breakpoint()
      print(f'>> acc/pcr not found for {ucp_base}, ignored test stats')
    return items
  
  # rendering stuff for html
  menuitems, pages = [], []

  if 'grid search':
    menuitems.append(mk_menusplit())
    for i, method in enumerate(ATK_METHODS):
      menuitems.append(mk_button(method))

      trs = []
      for eps in EPS[method]:
        tds = []
        for alpha in ALPHA:
          ucp_base = ucp_name(model, train_dataset, atk_dataset, method, eps, alpha)
          img_fp = os.path.join(img_dp, ucp_base + '.png')
          subtitle = f'{method} eps={eps} alpha={alpha}'
          if os.path.exists(img_fp):
            try:
              items = get_stats_items(model, ucp_base)
              tds.append(mk_td(mk_card_ex(img_fp, subtitle, items)))
            except Exception as e:
              print(f'>> {ucp_base} not found, ignore stats')
              tds.append(mk_td(mk_card(img_fp, subtitle)))
          else:
            tds.append(mk_td(mk_p(f'>> missing {img_fp}')))
        
        trs.append(mk_tr('\n'.join(tds)))

      pages.append(mk_table(method, '\n'.join(trs), display=(i == 0)))
  
  if 'ablation study':
    menuitems.append(mk_menusplit())

    ablas = 'alpha-decay'
    if True:
      menuitems.append(mk_button(ablas))

      trs = []
      if 'const':
        tds = []
        for method, eps, alpha in ALPHA_DECAYS['const']:
          ucp_base = ucp_name(model, train_dataset, atk_dataset, method, eps, alpha)
          img_fp = os.path.join(img_dp, ucp_base + '.png')
          if os.path.exists(img_fp):
            subtitle = f'{method} eps={eps} alpha={alpha}'
            try:
              items = get_stats_items(model, ucp_base)
              tds.append(mk_td(mk_card_ex(img_fp, subtitle, items)))
            except Exception as e:
              print(f'>> {ucp_base} not found, ignore stats')
              tds.append(mk_td(mk_card(img_fp, subtitle)))
          else:
            tds.append(mk_td(mk_p(f'>> missing {img_fp}')))
        trs.append(mk_tr('\n'.join(tds)))
      if 'decay':
        tds = []
        for method, eps, alpha_from, alpha_to in ALPHA_DECAYS['decay']:
          ucp_base = ucp_name(model, train_dataset, atk_dataset, method, eps, alpha_from, alpha_to)
          img_fp = os.path.join(img_dp, ucp_base + '.png')
          if os.path.exists(img_fp):
            subtitle = f'{method} eps={eps} alpha_from={alpha_from} alpha_to={alpha_to}'
            try:
              items = get_stats_items(model, ucp_base)
              tds.append(mk_td(mk_card_ex(img_fp, subtitle, items)))
            except Exception as e:
              print(f'>> {ucp_base} not found, ignore stats')
              tds.append(mk_td(mk_card(img_fp, subtitle)))
          else:
            tds.append(mk_td(mk_p(f'>> missing {img_fp}')))
        trs.append(mk_tr('\n'.join(tds)))
      pages.append(mk_table(ablas, '\n'.join(trs), display=False))

    ablas = 'models'
    if True:
      menuitems.append(mk_button(ablas))

      eps, alpha = MODELS_E_A
      cards = []
      for model in MODELS:
        ucp_base = ucp_name(model, train_dataset, atk_dataset, method, eps, alpha)
        img_fp = os.path.join(img_dp, ucp_base + '.png')
        if os.path.exists(img_fp):
          subtitle = f'{model} {method} eps={eps} alpha={alpha}'
          try:
            items = get_stats_items(model, ucp_base)
            cards.append(mk_card_ex(img_fp, subtitle, items))
          except Exception as e:
            print(f'>> {ucp_base} not found, ignore stats')
            cards.append(mk_card(img_fp, subtitle))
        else:
          cards.append(mk_grid(mk_p(f'>> missing {img_fp}')))
      pages.append(mk_divP(ablas, '\n'.join(cards), display=False))

  html = html_skel % (name, '\n'.join(menuitems), '\n'.join(pages))
  with open(html_fn, 'w', encoding='utf-8') as fh:
    fh.write(html)


if __name__ == '__main__':
  mk_index()
