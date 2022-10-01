#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/30 

import os

from mk_train_script import EPS, ALPHA
from attack import ATK_METHODS
from util import float_to_str 


img_dp = 'img'
html_fn = 'index.html'
name = 'resnet18_imagenet-svhn'

if __name__ == '__main__':
  html_skel = '''<!DOCTYPE>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width">
  <title>Show Universal Confusable Pertubations</title>
  <link rel="stylesheet" href="w3.css">
  <style>
    //table, th, td { border: 1px solid black; }
  </style>
</head>
<body>
<div class="w3-bar w3-blue">
<span class="w3-bar-item w3-large"> %s </span>
<span class="w3-bar-item"> | </span>
%s
</div>
<div class="w3-container w3-sand">
%s
</div>
<script>
function openTable(method) {
  let x = document.getElementsByClassName("method");
  for (let i = 0; i < x.length; i++)
    x[i].style.display = "none";
  document.getElementById(method).style.display = "block";
}
</script>
</body>
</html>
'''

  mk_tr   = lambda text: f'<tr>\n{text}\n</tr>'
  mk_td   = lambda text: f'  <td>{text}</td>'
  mk_img  = lambda src:  f'<img class="w3-image" src="{src}"></img>'
  mk_p    = lambda text: f'<p>{text}</p>'
  mk_div  = lambda text: f'<div class="w3-container w3-hover-border-red w3-center">{text}</div>'
  
  mk_button = lambda page: f'<button class="w3-bar-item w3-button w3-large" onclick="openTable(\'{page}\')">{page}</button>'
  mk_table = lambda id, content, display='none': f'<table id="{id}" class="w3-table method" style="display:{display}">{content}</table>'
  mk_card = lambda fp, title: mk_div(f'{mk_img(fp)}{mk_p(title)}')

  buttons, tables = [], []
  for i, method in enumerate(ATK_METHODS):
    button = mk_button(method)
    buttons.append(button)

    trs = []
    for eps in EPS:
      tds = []

      for alpha in ALPHA:
        suffix = f'{method}_e{float_to_str(eps)}_a{(float_to_str(alpha))}'
        fp = f'{img_dp}/{name}_{suffix}.png'

        if os.path.exists(fp):
          tds.append(mk_td(mk_card(fp, suffix)))
        else:
          tds.append(mk_td(mk_p(f'missing {suffix}')))

      trs.append(mk_tr('\n'.join(tds)))

    table = mk_table(method, '\n'.join(trs), display='block' if i == 0 else 'none')
    tables.append(table)
  
  html = html_skel % (name, '\n'.join(buttons), '\n'.join(tables))
  with open(html_fn, 'w', encoding='utf-8') as fh:
    fh.write(html)
