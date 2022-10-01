@REM all attacks are on `svhn`

@REM baseline
python attack.py -M resnet18 -D svhn --method pgd --eps 0.03  --alpha 0.001

@REM ablation on atk_method
python attack.py -M resnet18 -D svhn --method mifgsm --eps 0.03 --alpha 0.001
python attack.py -M resnet18 -D svhn --method pgdl2  --eps 1    --alpha 0.01 

@REM ablation on eps
python attack.py -M resnet18 -D svhn --method pgd --eps 0.05 --alpha 0.001
python attack.py -M resnet18 -D svhn --method pgd --eps 0.01 --alpha 0.001

@REM albation on alhpa
python attack.py -M resnet18 -D svhn --method pgd --eps 0.03 --alpha 0.0001 
python attack.py -M resnet18 -D svhn --method pgd --eps 0.03 --alpha 0.00001

@REM alabtiom on alpha decay
python attack.py -M resnet18 -D svhn --method pgd --eps 0.03 --alpha 0.001 --alpha_decay 1e-4
python attack.py -M resnet18 -D svhn --method pgd --eps 0.03 --alpha 0.001 --alpha_decay 2e-5

@REM ablation on model
python attack.py -M resnet18           -D svhn --method pgd --eps 0.03 --alpha 0.001
python attack.py -M resnet34           -D svhn --method pgd --eps 0.03 --alpha 0.001
python attack.py -M resnet50           -D svhn --method pgd --eps 0.03 --alpha 0.001
python attack.py -M resnext50_32x4d    -D svhn --method pgd --eps 0.03 --alpha 0.001
python attack.py -M wide_resnet50_2    -D svhn --method pgd --eps 0.03 --alpha 0.001
python attack.py -M densenet121        -D svhn --method pgd --eps 0.03 --alpha 0.001
python attack.py -M efficientnet_v2_s  -D svhn --method pgd --eps 0.03 --alpha 0.001
python attack.py -M shufflenet_v2_x1_5 -D svhn --method pgd --eps 0.03 --alpha 0.001
python attack.py -M squeezenet1_1      -D svhn --method pgd --eps 0.03 --alpha 0.001
python attack.py -M inception_v3       -D svhn --method pgd --eps 0.03 --alpha 0.001
python attack.py -M mobilenet_v3_large -D svhn --method pgd --eps 0.03 --alpha 0.001
python attack.py -M regnet_y_400mf     -D svhn --method pgd --eps 0.03 --alpha 0.001
python attack.py -M vit_b_16           -D svhn --method pgd --eps 0.03 --alpha 0.001
python attack.py -M swin_t             -D svhn --method pgd --eps 0.03 --alpha 0.001

@REM albation on steps
python attack.py -M resnet18 -D svhn --method pgd --eps 0.03 --alpha 0.001 --steps 1000 --steps_per_batch 20
python attack.py -M resnet18 -D svhn --method pgd --eps 0.03 --alpha 0.001 --steps 2000 --steps_per_batch 10
python attack.py -M resnet18 -D svhn --method pgd --eps 0.03 --alpha 0.001 --steps 3000 --steps_per_batch 40

@REM albation on batch_size
python attack.py -M resnet18 -D svhn --method pgd --eps 0.03 --alpha 0.001 -B 64
python attack.py -M resnet18 -D svhn --method pgd --eps 0.03 --alpha 0.001 -B 256
